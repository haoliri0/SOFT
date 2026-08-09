#include "frontend/stim.hpp"

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

void require_equivalent_factorizations(const std::string& text, const std::string& label) {
    using namespace symft;
    const auto circuit = parse_stim_circuit_text(text);
    const auto framed = lower_circuit_to_factored(circuit, FactorizationStrategy::CliffordFrames);
    const auto direct = lower_circuit_to_factored(circuit, FactorizationStrategy::DirectPullback);
    require(
        framed.state.pending_operations == direct.state.pending_operations,
        label + " direct pullback agrees with frame factorization");
    require(
        framed.measurement_records == direct.measurement_records,
        label + " direct pullback preserves measurement records");
    require(
        framed.instruction_pending_operation_counts == direct.instruction_pending_operation_counts,
        label + " direct pullback preserves pending-operation positions");
    require(
        framed.state.context->next_condition == direct.state.context->next_condition &&
            framed.state.context->bernoulli_probabilities == direct.state.context->bernoulli_probabilities &&
            framed.state.context->condition_to_categorical == direct.state.context->condition_to_categorical &&
            framed.state.context->categorical_distributions.size() ==
                direct.state.context->categorical_distributions.size(),
        label + " direct pullback preserves symbolic condition allocation");
    for (std::size_t index = 0;
         index < framed.state.context->categorical_distributions.size();
         ++index) {
        const auto& framed_distribution =
            framed.state.context->categorical_distributions[index];
        const auto& direct_distribution =
            direct.state.context->categorical_distributions[index];
        require(
            framed_distribution.nbits == direct_distribution.nbits &&
                framed_distribution.conditions == direct_distribution.conditions &&
                framed_distribution.assignments == direct_distribution.assignments &&
                framed_distribution.probabilities == direct_distribution.probabilities,
            label + " direct pullback preserves categorical distributions");
    }
}

void test_direct_pullback_factorization() {
    using namespace symft;
    const std::vector<std::string> single_qubit_gates{
        "H",          "H_NXY",     "H_NXZ",     "H_NYZ", "H_XY",      "H_YZ",
        "C_NXYZ",     "C_NZYX",    "C_XNYZ",    "C_XYNZ", "C_XYZ",    "C_ZNYX",
        "C_ZYNX",     "C_ZYX",     "S",          "S_DAG", "SQRT_X",    "SQRT_X_DAG",
        "SQRT_Y",     "SQRT_Y_DAG", "X",         "Y",     "Z",
    };
    for (const auto& gate : single_qubit_gates) {
        for (const char axis : std::string("XYZ")) {
            require_equivalent_factorizations(
                gate + " 0\nMPP " + axis + "0\n",
                gate + " on " + axis);
        }
    }

    const std::vector<std::string> two_qubit_gates{
        "CX",          "CY",          "CZ",          "SWAP",        "CXSWAP",
        "CZSWAP",      "ISWAP",       "ISWAP_DAG",   "SQRT_XX",     "SQRT_XX_DAG",
        "SQRT_YY",     "SQRT_YY_DAG", "SQRT_ZZ",     "SQRT_ZZ_DAG", "SWAPCX",
        "XCX",         "XCY",         "XCZ",         "YCX",          "YCY",
        "YCZ",
    };
    const std::string axes = "IXYZ";
    for (const auto& gate : two_qubit_gates) {
        for (char a : axes) {
            for (char b : axes) {
                if (a == 'I' && b == 'I') {
                    continue;
                }
                std::string product;
                if (a != 'I') {
                    product += a;
                    product += '0';
                }
                if (b != 'I') {
                    if (!product.empty()) {
                        product += '*';
                    }
                    product += b;
                    product += '1';
                }
                require_equivalent_factorizations(
                    gate + " 0 1\nMPP " + product + "\n",
                    gate + " on " + product);
            }
        }
    }

    require_equivalent_factorizations(
        "H 0\n"
        "X_ERROR(0.125) 0\n"
        "CX 0 1\n"
        "DEPOLARIZE1(0.25) 1\n"
        "M(0.1) !0\n"
        "CX rec[-1] 1\n"
        "S 1\n"
        "MR(0.2) !1\n"
        "R 0\n"
        "PAULI_CHANNEL_2("
        "0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,"
        "0.01,0.01,0.01,0.01,0.01,0.01,0.01) 0 1\n"
        "R_PAULI(0.25) X0*Y1\n"
        "EXP_VAL Z0*X1\n",
        "mixed noise, reset, feedback, rotation, and expectation circuit");

    std::string staggered_batch;
    for (int repeat = 0; repeat < 160; ++repeat) {
        const int q = repeat & 7;
        const int next = (q + 1) & 7;
        staggered_batch += "H " + std::to_string(q) + "\n";
        staggered_batch +=
            "CX " + std::to_string(q) + " " + std::to_string(next) + "\n";
        staggered_batch += "X_ERROR(0.01) " + std::to_string(next) + "\n";
        staggered_batch +=
            "EXP_VAL X" + std::to_string(q) + "*Z" + std::to_string(next) + "\n";
    }
    require_equivalent_factorizations(
        staggered_batch,
        "staggered batched pullback circuit");

    std::string noisy;
    for (int repeat = 0; repeat < 32; ++repeat) {
        noisy += "X_ERROR(0.01) 0 1 2 3 4 5 6 7\n";
    }
    noisy += "M 0\n";
    const auto noisy_circuit = parse_stim_circuit_text(noisy);
    const auto noisy_estimate = estimate_circuit_factorization(noisy_circuit);
    require(noisy_estimate.clifford_operations == 0, "noise-heavy estimate Clifford count");
    require(noisy_estimate.conditional_pauli_operations == 256, "noise-heavy estimate Pauli count");
    require(noisy_estimate.pending_pauli_operations == 1, "noise-heavy estimate pending count");
    require(noisy_estimate.pullback_event_qubit_touches == 256, "noise-heavy estimate touch count");
    require(
        noisy_estimate.preferred_strategy == FactorizationStrategy::DirectPullback,
        "noise-heavy circuit selects direct pullback");
    require(
        lower_circuit_to_factored(noisy_circuit).factorization_strategy ==
            FactorizationStrategy::DirectPullback,
        "automatic lowering uses direct pullback for noise-heavy circuit");

    std::string noisy_many_targets = noisy;
    for (int repeat = 1; repeat < 64; ++repeat) {
        noisy_many_targets += "M 0\n";
    }
    const auto noisy_many_target_estimate =
        estimate_circuit_factorization(parse_stim_circuit_text(noisy_many_targets));
    require(
        noisy_many_target_estimate.direct_pullback_work <
            noisy_many_target_estimate.frame_work,
        "raw asymptotic estimate favors direct pullback in deep noisy circuit");
    require(
        noisy_many_target_estimate.preferred_strategy ==
            FactorizationStrategy::CliffordFrames,
        "locality guard retains packed frames for many pending Paulis");

    std::string clifford_heavy = "H 1\n";
    for (int repeat = 0; repeat < 100; ++repeat) {
        clifford_heavy += "H 0\n";
    }
    for (int repeat = 0; repeat < 20; ++repeat) {
        clifford_heavy += "M 0\n";
    }
    const auto clifford_heavy_circuit = parse_stim_circuit_text(clifford_heavy);
    const auto clifford_heavy_estimate = estimate_circuit_factorization(clifford_heavy_circuit);
    require(
        clifford_heavy_estimate.preferred_strategy == FactorizationStrategy::CliffordFrames,
        "Clifford-heavy circuit retains frame factorization");

    std::string shared_clifford_prefix = "H 127\n";
    for (int repeat = 0; repeat < 512; ++repeat) {
        shared_clifford_prefix += (repeat & 1) ? "S 0\n" : "H 0\n";
    }
    for (int repeat = 0; repeat < 512; ++repeat) {
        shared_clifford_prefix += "EXP_VAL Z0\n";
    }
    const auto shared_prefix_estimate = estimate_circuit_factorization(
        parse_stim_circuit_text(shared_clifford_prefix));
    require(
        shared_prefix_estimate.preferred_strategy ==
            FactorizationStrategy::CliffordFrames,
        "shared Clifford prefix retains frame factorization");

    std::string interleaved_cliffords = "H 127\n";
    for (int repeat = 0; repeat < 512; ++repeat) {
        const int q = repeat & 127;
        interleaved_cliffords += "H " + std::to_string(q) + "\n";
        interleaved_cliffords += "EXP_VAL Z" + std::to_string(q) + "\n";
    }
    const auto interleaved_estimate = estimate_circuit_factorization(
        parse_stim_circuit_text(interleaved_cliffords));
    require(
        interleaved_estimate.preferred_strategy ==
            FactorizationStrategy::DirectPullback,
        "interleaved pending Paulis select batched direct pullback");

    constexpr int n = 32;
    FrameFactoredState framed(n, 0, FactorizationStrategy::CliffordFrames);
    FrameFactoredState direct(n, 0, FactorizationStrategy::DirectPullback);
    std::mt19937_64 rng(0x5eed1234ULL);
    for (int step = 0; step < 3000; ++step) {
        const int q0 = static_cast<int>(rng() % n);
        int q1 = static_cast<int>(rng() % (n - 1));
        if (q1 >= q0) {
            ++q1;
        }
        switch (rng() % 9) {
        case 0:
            left_H(framed, q0);
            left_H(direct, q0);
            break;
        case 1:
            left_S(framed, q0);
            left_S(direct, q0);
            break;
        case 2:
            left_SQRT_Y_DAG(framed, q0);
            left_SQRT_Y_DAG(direct, q0);
            break;
        case 3:
            left_CX(framed, q0, q1);
            left_CX(direct, q0, q1);
            break;
        case 4:
            left_CZ(framed, q0, q1);
            left_CZ(direct, q0, q1);
            break;
        case 5:
            left_SWAP(framed, q0, q1);
            left_SWAP(direct, q0, q1);
            break;
        case 6: {
            PauliString correction(n);
            for (int item = 0; item < 1 + static_cast<int>(rng() % 4); ++item) {
                const int q = static_cast<int>(rng() % n);
                const unsigned axis = 1 + static_cast<unsigned>(rng() % 3);
                correction.set_xbit(q, (axis & 1u) != 0);
                correction.set_zbit(q, (axis & 2u) != 0);
            }
            correction.set_phase(pauli_body_y_count(correction));
            const int condition = 1 + static_cast<int>(rng() % 64);
            apply_pauli(framed, correction, condition);
            apply_pauli(direct, correction, condition);
            break;
        }
        case 7: {
            const unsigned axis = 1 + static_cast<unsigned>(rng() % 3);
            const int condition = 1 + static_cast<int>(rng() % 64);
            apply_single_qubit_pauli(framed, q0, (axis & 1u) != 0, (axis & 2u) != 0, condition);
            apply_single_qubit_pauli(direct, q0, (axis & 1u) != 0, (axis & 2u) != 0, condition);
            break;
        }
        default: {
            PauliString probe(n);
            for (int item = 0; item < 1 + static_cast<int>(rng() % 4); ++item) {
                const int q = static_cast<int>(rng() % n);
                const unsigned axis = 1 + static_cast<unsigned>(rng() % 3);
                probe.set_xbit(q, (axis & 1u) != 0);
                probe.set_zbit(q, (axis & 2u) != 0);
            }
            probe.set_phase(pauli_body_y_count(probe));
            const auto framed_probe = apply_pauli_measurement(framed, probe);
            const auto direct_probe = apply_pauli_measurement(direct, probe);
            require(
                framed_probe == direct_probe,
                "indexed direct pullback agrees throughout randomized history");
            break;
        }
        }
    }
    require(
        framed.pending_operations == direct.pending_operations,
        "indexed direct pullback preserves randomized pending sequence");
}

} // namespace

int main() {
    test_direct_pullback_factorization();
    std::cout << "All factorization tests passed\n";
    return 0;
}
