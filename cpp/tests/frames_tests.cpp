#include "core/frames.hpp"
#include "frontend/stim.hpp"
#include "sampler/single_shot.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

symft::PauliString expected_pauli_preimage(std::string preimage) {
    bool negative = false;
    if (!preimage.empty() && preimage[0] == '-') {
        negative = true;
        preimage = preimage.substr(1);
    }
    symft::PauliString out = symft::pauli_string(preimage);
    if (negative) {
        out.phase_shift(2);
    }
    return out;
}

void test_clifford_frame() {
    using namespace symft;
    CliffordFrame cf(2);
    left_CX(cf, 0, 1);
    require(preimage(cf, pauli_x(2, 0)) == pauli_string("XX"), "CX maps Xc");
    require(preimage(cf, pauli_z(2, 1)) == pauli_string("ZZ"), "CX maps Zt");
    const auto stored_body = pauli_z(2, 1);
    const auto current_correction = pauli_x(2, 0);
    require(
        pauli_anticommutes(stored_body, preimage(cf, current_correction)) ==
            pauli_anticommutes(coordinates_in_frame(cf, stored_body), current_correction),
        "deferred frame preserves commutation");

    CliffordFrame h(1);
    left_H(h, 0);
    left_S(h, 0);
    require(preimage(h, pauli_x(1, 0)) == pauli_y(1, 0), "left composition order");

    CliffordFrame wide(65);
    auto wide_pauli = pauli_identity(65);
    wide_pauli.set_xbit(0);
    wide_pauli.set_zbit(64);
    require(coordinates_in_frame(wide, wide_pauli) == wide_pauli, "sparse frame coordinates");
    require(preimage(wide, wide_pauli) == wide_pauli, "sparse frame preimage");

    CliffordFrame columns(1024);
    for (int q = 1; q < 1024; ++q) {
        left_CX(columns, 0, q);
    }
    const auto column_query = pauli_z(1024, 0);
    const auto column_first = coordinates_in_frame(columns, column_query);
    const auto column_second = coordinates_in_frame(columns, column_query);
    const auto column_third = coordinates_in_frame(columns, column_query);
    require(column_first == column_second && column_second == column_third, "column frame coordinate cache");

    CliffordFrame cached_support(2);
    require(
        preimage(cached_support, pauli_x(2, 0)) == pauli_x(2, 0),
        "initial sparse support cache");
    left_CX(cached_support, 0, 1);
    require(
        preimage(cached_support, pauli_x(2, 0)) == pauli_string("XX"),
        "tableau mutation invalidates sparse support automatically");

    CliffordFrame cached_coordinates(1);
    require(
        coordinates_in_frame(cached_coordinates, pauli_z(1, 0)) == pauli_z(1, 0),
        "initial coordinate index");
    left_H(cached_coordinates, 0);
    require(
        coordinates_in_frame(cached_coordinates, pauli_z(1, 0)) == pauli_x(1, 0),
        "tableau mutation rebuilds the coordinate index lazily");

    CliffordFrame phaseful(1);
    left_S(phaseful, 0);
    require(
        (phaseful.generator(phaseful.xrow(0)).phase_exponent() & 1) != 0,
        "Clifford frame retains an odd Pauli phase exponent");
    require(
        coordinates_in_frame(phaseful, pauli_y(1, 0)) == neg(pauli_x(1, 0)),
        "body decomposition reconstructs an odd Clifford-frame phase");
    left_Z(phaseful, 0);
    require(
        coordinates_in_frame(phaseful, pauli_y(1, 0)) == pauli_x(1, 0),
        "phase-only frame updates reuse the body index without losing the sign");
}

void test_extended_clifford_frame_preimages() {
    using namespace symft;
    using SingleGate = void (*)(CliffordFrame&, int);
    struct SingleCase {
        const char* name;
        SingleGate apply;
        const char* x_preimage;
        const char* z_preimage;
    };
    const std::vector<SingleCase> single_cases{
        {"H_NXY", static_cast<SingleGate>(&left_H_NXY), "-Y", "-Z"},
        {"H_NXZ", static_cast<SingleGate>(&left_H_NXZ), "-Z", "-X"},
        {"H_NYZ", static_cast<SingleGate>(&left_H_NYZ), "-X", "-Y"},
        {"H_XY", static_cast<SingleGate>(&left_H_XY), "Y", "-Z"},
        {"H_YZ", static_cast<SingleGate>(&left_H_YZ), "-X", "Y"},
        {"C_NXYZ", static_cast<SingleGate>(&left_C_NXYZ), "-Z", "Y"},
        {"C_NZYX", static_cast<SingleGate>(&left_C_NZYX), "Y", "-X"},
        {"C_XNYZ", static_cast<SingleGate>(&left_C_XNYZ), "Z", "-Y"},
        {"C_XYNZ", static_cast<SingleGate>(&left_C_XYNZ), "-Z", "-Y"},
        {"C_XYZ", static_cast<SingleGate>(&left_C_XYZ), "Z", "Y"},
        {"C_ZNYX", static_cast<SingleGate>(&left_C_ZNYX), "-Y", "X"},
        {"C_ZYNX", static_cast<SingleGate>(&left_C_ZYNX), "-Y", "-X"},
        {"C_ZYX", static_cast<SingleGate>(&left_C_ZYX), "Y", "X"},
        {"SQRT_X", static_cast<SingleGate>(&left_SQRT_X), "X", "Y"},
        {"SQRT_X_DAG", static_cast<SingleGate>(&left_SQRT_X_DAG), "X", "-Y"},
        {"SQRT_Y", static_cast<SingleGate>(&left_SQRT_Y), "Z", "-X"},
        {"SQRT_Y_DAG", static_cast<SingleGate>(&left_SQRT_Y_DAG), "-Z", "X"},
        {"Y", static_cast<SingleGate>(&left_Y), "-X", "-Z"},
    };
    for (const auto& c : single_cases) {
        CliffordFrame frame(1);
        c.apply(frame, 0);
        require(
            preimage(frame, pauli_x(1, 0)) == expected_pauli_preimage(c.x_preimage),
            std::string(c.name) + " X generator preimage");
        require(
            preimage(frame, pauli_z(1, 0)) == expected_pauli_preimage(c.z_preimage),
            std::string(c.name) + " Z generator preimage");
    }

    using TwoGate = void (*)(CliffordFrame&, int, int);
    struct TwoCase {
        const char* name;
        TwoGate apply;
        const char* xa_preimage;
        const char* za_preimage;
        const char* xb_preimage;
        const char* zb_preimage;
    };
    const std::vector<TwoCase> two_cases{
        {"CY", static_cast<TwoGate>(&left_CY), "XY", "Z_", "ZX", "ZZ"},
        {"CXSWAP", static_cast<TwoGate>(&left_CXSWAP), "_X", "ZZ", "XX", "Z_"},
        {"CZSWAP", static_cast<TwoGate>(&left_CZSWAP), "ZX", "_Z", "XZ", "Z_"},
        {"ISWAP", static_cast<TwoGate>(&left_ISWAP), "-ZY", "_Z", "-YZ", "Z_"},
        {"ISWAP_DAG", static_cast<TwoGate>(&left_ISWAP_DAG), "ZY", "_Z", "YZ", "Z_"},
        {"SQRT_XX", static_cast<TwoGate>(&left_SQRT_XX), "X_", "YX", "_X", "XY"},
        {"SQRT_XX_DAG", static_cast<TwoGate>(&left_SQRT_XX_DAG), "X_", "-YX", "_X", "-XY"},
        {"SQRT_YY", static_cast<TwoGate>(&left_SQRT_YY), "ZY", "-XY", "YZ", "-YX"},
        {"SQRT_YY_DAG", static_cast<TwoGate>(&left_SQRT_YY_DAG), "-ZY", "XY", "-YZ", "YX"},
        {"SQRT_ZZ", static_cast<TwoGate>(&left_SQRT_ZZ), "-YZ", "Z_", "-ZY", "_Z"},
        {"SQRT_ZZ_DAG", static_cast<TwoGate>(&left_SQRT_ZZ_DAG), "YZ", "Z_", "ZY", "_Z"},
        {"SWAPCX", static_cast<TwoGate>(&left_SWAPCX), "XX", "_Z", "X_", "ZZ"},
        {"XCX", static_cast<TwoGate>(&left_XCX), "X_", "ZX", "_X", "XZ"},
        {"XCY", static_cast<TwoGate>(&left_XCY), "X_", "ZY", "XX", "XZ"},
        {"XCZ", static_cast<TwoGate>(&left_XCZ), "X_", "ZZ", "XX", "_Z"},
        {"YCX", static_cast<TwoGate>(&left_YCX), "XX", "ZX", "_X", "YZ"},
        {"YCY", static_cast<TwoGate>(&left_YCY), "XY", "ZY", "YX", "YZ"},
        {"YCZ", static_cast<TwoGate>(&left_YCZ), "XZ", "ZZ", "YX", "_Z"},
    };
    for (const auto& c : two_cases) {
        CliffordFrame frame(2);
        c.apply(frame, 0, 1);
        require(
            preimage(frame, pauli_x(2, 0)) == expected_pauli_preimage(c.xa_preimage),
            std::string(c.name) + " X_ generator preimage");
        require(
            preimage(frame, pauli_z(2, 0)) == expected_pauli_preimage(c.za_preimage),
            std::string(c.name) + " Z_ generator preimage");
        require(
            preimage(frame, pauli_x(2, 1)) == expected_pauli_preimage(c.xb_preimage),
            std::string(c.name) + " _X generator preimage");
        require(
            preimage(frame, pauli_z(2, 1)) == expected_pauli_preimage(c.zb_preimage),
            std::string(c.name) + " _Z generator preimage");
    }
}

void test_sqrt_gate_directions() {
    using namespace symft;
    struct DirectionCase {
        const char* name;
        const char* circuit;
        bool expected_measurement;
    };
    const std::vector<DirectionCase> cases{
        {"SQRT_X", "SQRT_X 0\nMY 0\n", true},
        {"SQRT_X_DAG", "SQRT_X_DAG 0\nMY 0\n", false},
        {"SQRT_Y", "SQRT_Y 0\nH 0\nM 0\n", false},
        {"SQRT_Y_DAG", "SQRT_Y_DAG 0\nH 0\nM 0\n", true},
    };
    for (const auto& c : cases) {
        const auto parsed = parse_stim_text(c.circuit);
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        const auto records = sample_measurements(program, 8, 31);
        for (const auto& shot : records) {
            require(
                packed_bit(shot, 0) == c.expected_measurement,
                std::string(c.name) + " uses the Stim gate direction");
        }
    }

    const auto sample = [](const char* circuit) {
        const auto parsed = parse_stim_text(circuit);
        PendingFactoredState pending(parsed.state);
        return sample_measurements(plan_factored_updates(pending), 8, 37);
    };
    require(
        sample("SQRT_X 0\nMY 0\n") == sample("H 0\nS 0\nH 0\nMY 0\n"),
        "SQRT_X agrees with H S H");
    require(
        sample("SQRT_X_DAG 0\nMY 0\n") == sample("H 0\nS_DAG 0\nH 0\nMY 0\n"),
        "SQRT_X_DAG agrees with H S_DAG H");
}

void test_extended_clifford_gate_directions() {
    using namespace symft;
    struct GateCase {
        const char* name;
        const char* native_circuit;
        const char* reference_circuit;
    };
    const std::vector<GateCase> cases{
        {"C_NXYZ", "C_NXYZ 0\n", "H 0\nS 0\nH 0\nS_DAG 0\n"},
        {"C_NZYX", "C_NZYX 0\n", "S 0\nS 0\nH 0\nS_DAG 0\n"},
        {"C_XNYZ", "C_XNYZ 0\n", "S 0\nH 0\n"},
        {"C_XYNZ", "C_XYNZ 0\n", "H 0\nS_DAG 0\nH 0\nS 0\n"},
        {"C_XYZ", "C_XYZ 0\n", "S_DAG 0\nH 0\n"},
        {"C_ZNYX", "C_ZNYX 0\n", "H 0\nS_DAG 0\n"},
        {"C_ZYNX", "C_ZYNX 0\n", "S 0\nH 0\nS_DAG 0\nH 0\n"},
        {"C_ZYX", "C_ZYX 0\n", "H 0\nS 0\n"},
        {"CXSWAP", "CXSWAP 0 1\n", "CX 0 1\nSWAP 0 1\n"},
        {"SWAPCX", "SWAPCX 0 1\n", "SWAP 0 1\nCX 0 1\n"},
        {"ISWAP",
         "ISWAP 0 1\n",
         "CX 1 0\n"
         "CX 0 1\n"
         "CX 1 0\n"
         "S 0\n"
         "H 1\n"
         "CX 0 1\n"
         "H 1\n"
         "S 1\n"},
        {"ISWAP_DAG",
         "ISWAP_DAG 0 1\n",
         "S_DAG 1\n"
         "H 1\n"
         "CX 0 1\n"
         "H 1\n"
         "S_DAG 0\n"
         "CX 1 0\n"
         "CX 0 1\n"
         "CX 1 0\n"},
    };
    for (const auto& c : cases) {
        const auto native = parse_stim_text(c.native_circuit);
        const auto reference = parse_stim_text(c.reference_circuit);
        require(
            native.state.pending_operations.empty() && reference.state.pending_operations.empty(),
            std::string(c.name) + " direction test remains Clifford-only");
        require(
            native.state.clifford == reference.state.clifford,
            std::string(c.name) + " agrees with its primitive Clifford decomposition");
    }
}

} // namespace

int main() {
    test_clifford_frame();
    test_extended_clifford_frame_preimages();
    test_sqrt_gate_directions();
    test_extended_clifford_gate_directions();
    std::cout << "All frame tests passed\n";
    return 0;
}
