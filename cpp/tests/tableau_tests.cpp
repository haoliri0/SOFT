#include "core/tableau.hpp"
#include "core/frames.hpp"

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

void require_canonical(const symft::PlanningTableau& tableau, const std::string& context) {
    using namespace symft;
    const int n = tableau.nqubits();
    for (int q = 0; q < n; ++q) {
        require(pauli_squares_to_identity(tableau.stabilizer(q)), context + " stabilizer is Hermitian");
        require(pauli_squares_to_identity(tableau.destabilizer(q)), context + " destabilizer is Hermitian");
        for (int r = 0; r < n; ++r) {
            require(
                !pauli_anticommutes(tableau.stabilizer(q), tableau.stabilizer(r)),
                context + " stabilizers commute");
            require(
                !pauli_anticommutes(tableau.destabilizer(q), tableau.destabilizer(r)),
                context + " destabilizers commute");
            require(
                pauli_anticommutes(tableau.stabilizer(q), tableau.destabilizer(r)) == (q == r),
                context + " generator pairs have canonical commutation");
        }
    }
}

void require_random_roundtrips(
    const symft::PlanningTableau& tableau,
    std::mt19937_64& rng,
    int count,
    const std::string& context) {
    using namespace symft;
    for (int sample = 0; sample < count; ++sample) {
        PauliString coordinates(tableau.nqubits());
        for (int q = 0; q < tableau.nqubits(); ++q) {
            coordinates.set_xbit(q, (rng() & 1u) != 0);
            coordinates.set_zbit(q, (rng() & 1u) != 0);
        }
        coordinates.set_phase(static_cast<int>(rng() & 3u));
        const PauliString physical = tableau.reconstruct(coordinates);
        require(tableau.decompose(physical) == coordinates, context + " coordinate round trip");
    }
}

void require_matches_clifford_frame(
    const symft::PlanningTableau& tableau,
    std::mt19937_64& rng,
    int count,
    const std::string& context) {
    using namespace symft;
    CliffordFrame frame(tableau.nqubits());
    for (int q = 0; q < tableau.nqubits(); ++q) {
        frame.copy_pauli_to_row(frame.xrow(q), tableau.destabilizer(q));
        frame.copy_pauli_to_row(frame.zrow(q), tableau.stabilizer(q));
    }
    for (int sample = 0; sample < count; ++sample) {
        PauliString coordinates(tableau.nqubits());
        for (int q = 0; q < tableau.nqubits(); ++q) {
            coordinates.set_xbit(q, (rng() & 1u) != 0);
            coordinates.set_zbit(q, (rng() & 1u) != 0);
        }
        coordinates.set_phase(static_cast<int>(rng() & 3u));
        const PauliString physical = tableau.reconstruct(coordinates);
        require(
            preimage(frame, coordinates) == physical,
            context + " reconstructs the same full Pauli phase");
        require(
            coordinates_in_frame(frame, physical) == coordinates,
            context + " Clifford-frame coordinates preserve the full phase");
        require(
            tableau.decompose(physical) == coordinates,
            context + " planning coordinates preserve the real generator signs");
    }
}

void test_identity_decomposition() {
    using namespace symft;
    std::mt19937_64 rng(1234);
    for (int n : {0, 1, 2, 63, 64, 65}) {
        PlanningTableau tableau(n);
        require_canonical(tableau, "identity tableau");
        require_random_roundtrips(tableau, rng, 32, "identity tableau");
    }
}

void test_dormant_rotation_promotion() {
    using namespace symft;
    PlanningTableau tableau(4);
    const PauliString coordinates = pauli_string("XYZX");
    const PauliString physical = tableau.reconstruct(coordinates);

    tableau.promote_dormant_rotation(coordinates, 1, 2);

    require_canonical(tableau, "dormant rotation promotion");
    require(tableau.decompose(physical) == pauli_x(4, 1), "promoted Pauli becomes the next active X");
    require(tableau.stabilizer(1) == pauli_z(4, 3), "promotion swaps the dormant pivot stabilizer into the active block");
    require(tableau.destabilizer(1) == physical, "promotion stores the physical Pauli as the new destabilizer");
    require(
        tableau.stabilizer(3) == pauli_z(4, 1) * pauli_z(4, 3),
        "promotion updates the displaced stabilizer by the pivot stabilizer");
    require(
        tableau.destabilizer(3) == pauli_x(4, 1) * pauli_z(4, 3),
        "promotion updates the displaced destabilizer by the pivot stabilizer");

    std::mt19937_64 rng(5678);
    require_random_roundtrips(tableau, rng, 128, "promoted tableau");
}

void test_dormant_measurement_replacement() {
    using namespace symft;
    PlanningTableau tableau(4);
    const PauliString coordinates = pauli_string("YZXX");
    const PauliString physical = tableau.reconstruct(coordinates);
    const PauliString old_pivot_stabilizer = tableau.stabilizer(3);

    const PauliString correction = tableau.replace_dormant_measurement(coordinates, 2, 1);

    require_canonical(tableau, "dormant measurement replacement");
    require(correction == old_pivot_stabilizer, "dormant measurement correction is the old pivot stabilizer");
    require(tableau.decompose(physical) == pauli_z(4, 3), "measured Pauli becomes the pivot stabilizer");

    std::mt19937_64 rng(9012);
    require_random_roundtrips(tableau, rng, 128, "dormant measurement tableau");
}

void test_active_diagonal_measurement() {
    using namespace symft;
    PlanningTableau tableau(4);
    const PauliString coordinates = pauli_string("ZZ_Z");
    const PauliString physical = tableau.reconstruct(coordinates);
    const PauliString old_pivot_destabilizer = tableau.destabilizer(1);

    const PauliString correction = tableau.remove_active_measurement(coordinates, 3, 1, true);

    require_canonical(tableau, "active diagonal measurement");
    require(correction == old_pivot_destabilizer, "diagonal correction uses the pivot destabilizer");
    require(tableau.decompose(physical) == pauli_z(4, 2), "diagonal measured Pauli moves to the dormant boundary");
    require(
        tableau.destabilizer(3) == pauli_x(4, 3) * old_pivot_destabilizer,
        "diagonal update includes the dormant stabilizer component");

    std::mt19937_64 rng(3456);
    require_random_roundtrips(tableau, rng, 128, "active diagonal tableau");
}

void test_active_nondiagonal_measurement() {
    using namespace symft;
    PlanningTableau tableau(4);
    const PauliString coordinates = pauli_string("XZ_Z");
    const PauliString physical = tableau.reconstruct(coordinates);
    const PauliString old_pivot_stabilizer = tableau.stabilizer(0);

    const PauliString correction = tableau.remove_active_measurement(coordinates, 3, 0, false);

    require_canonical(tableau, "active nondiagonal measurement");
    require(correction == old_pivot_stabilizer, "nondiagonal correction uses the pivot stabilizer");
    require(tableau.decompose(physical) == pauli_z(4, 2), "nondiagonal measured Pauli moves to the dormant boundary");
    require(
        tableau.destabilizer(3) == pauli_x(4, 3) * old_pivot_stabilizer,
        "nondiagonal update includes the dormant stabilizer component");

    std::mt19937_64 rng(7890);
    require_random_roundtrips(tableau, rng, 128, "active nondiagonal tableau");
}

void test_packed_generator_sign_update() {
    using namespace symft;
    PlanningTableau tableau(2);

    // The first promotion creates destabilizer XX in coordinate 0 and
    // stabilizer ZZ in coordinate 1.
    tableau.promote_dormant_rotation(pauli_string("XX"), 0, 1);
    require(tableau.destabilizer(0) == pauli_string("XX"), "first promotion creates XX");
    require(tableau.stabilizer(1) == pauli_string("ZZ"), "first promotion creates ZZ");

    // The second promotion multiplies those commuting generators. Their
    // bodies combine to YY, but XX * ZZ = -YY, exercising the packed sign
    // update independently of the transposed body update.
    tableau.promote_dormant_rotation(pauli_string("ZX"), 1, 0);
    require(tableau.destabilizer(0) == neg(pauli_string("YY")), "packed update keeps XX * ZZ = -YY");

    std::mt19937_64 rng(13579);
    require_canonical(tableau, "packed generator sign update");
    require_random_roundtrips(tableau, rng, 128, "packed generator sign update");
}

void test_shared_tableau_body_core_preserves_distinct_phase_policies() {
    using namespace symft;
    PlanningTableau tableau(4);
    tableau.promote_dormant_rotation(pauli_string("XYZX"), 1, 2);

    bool has_odd_internal_phase = false;
    for (int q = 0; q < tableau.nqubits(); ++q) {
        has_odd_internal_phase |= (tableau.stabilizer(q).phase_exponent() & 1) != 0;
        has_odd_internal_phase |= (tableau.destabilizer(q).phase_exponent() & 1) != 0;
    }
    require(
        has_odd_internal_phase,
        "shared body test includes a Clifford row with an odd Pauli phase exponent");

    std::mt19937_64 rng(97531);
    require_matches_clifford_frame(tableau, rng, 256, "shared tableau body core");
}

void test_sequential_updates_keep_transposed_coordinates_synchronized() {
    using namespace symft;
    PlanningTableau tableau(70);
    std::mt19937_64 rng(24680);
    const auto hermitian = [](PauliString coordinates) {
        coordinates.set_phase(pauli_body_y_count(coordinates));
        return coordinates;
    };
    const auto verify = [&](const std::string& context) {
        require_canonical(tableau, context);
        require_random_roundtrips(tableau, rng, 32, context);
    };

    PauliString first_promotion(70);
    first_promotion.set_xbit(69);
    first_promotion.set_zbit(2);
    first_promotion.set_xbit(65);
    first_promotion.set_zbit(65);
    tableau.promote_dormant_rotation(hermitian(first_promotion), 0, 69);
    verify("first sequential promotion");

    PauliString second_promotion(70);
    second_promotion.set_xbit(0);
    second_promotion.set_zbit(0);
    second_promotion.set_xbit(64);
    second_promotion.set_zbit(66);
    tableau.promote_dormant_rotation(hermitian(second_promotion), 1, 63);
    verify("second sequential promotion");

    PauliString dormant_measurement(70);
    dormant_measurement.set_zbit(0);
    dormant_measurement.set_xbit(68);
    dormant_measurement.set_zbit(69);
    tableau.replace_dormant_measurement(hermitian(dormant_measurement), 2, 66);
    verify("sequential dormant measurement");

    PauliString diagonal_measurement(70);
    diagonal_measurement.set_zbit(0);
    diagonal_measurement.set_zbit(1);
    diagonal_measurement.set_zbit(65);
    tableau.remove_active_measurement(hermitian(diagonal_measurement), 2, 1, true);
    verify("sequential active diagonal measurement");

    PauliString third_promotion(70);
    third_promotion.set_zbit(0);
    third_promotion.set_xbit(67);
    third_promotion.set_zbit(68);
    tableau.promote_dormant_rotation(hermitian(third_promotion), 1, 66);
    verify("third sequential promotion");

    PauliString nondiagonal_measurement(70);
    nondiagonal_measurement.set_xbit(0);
    nondiagonal_measurement.set_zbit(1);
    nondiagonal_measurement.set_zbit(66);
    tableau.remove_active_measurement(hermitian(nondiagonal_measurement), 2, 0, false);
    verify("sequential active nondiagonal measurement");
}

} // namespace

int main() {
    test_identity_decomposition();
    test_dormant_rotation_promotion();
    test_dormant_measurement_replacement();
    test_active_diagonal_measurement();
    test_active_nondiagonal_measurement();
    test_packed_generator_sign_update();
    test_shared_tableau_body_core_preserves_distinct_phase_policies();
    test_sequential_updates_keep_transposed_coordinates_synchronized();
    std::cout << "tableau_tests passed\n";
    return 0;
}
