#include "frontend/stim.hpp"
#include "frontend/stim_prepared_sampler.hpp"
#include "sampler/batch_internal.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/component_plan.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/single_shot.hpp"
#include "simd/simd.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

void require_same_counts(
    const symft::CircuitSamplingCounts& lhs,
    const symft::CircuitSamplingCounts& rhs,
    const std::string& context) {
    require(lhs.shots == rhs.shots, context + " sampled count");
    require(lhs.discarded == rhs.discarded, context + " detector count");
    require(lhs.accepted == rhs.accepted, context + " accepted count");
    require(lhs.logical_errors == rhs.logical_errors, context + " logical count");
}

template <typename Fn>
void require_throws(Fn&& fn, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const symft::Error&) {
        threw = true;
    }
    require(threw, message);
}

template <typename Fn>
void require_throws_with_message(Fn&& fn, const std::string& expected, const std::string& message) {
    bool threw = false;
    try {
        fn();
    } catch (const symft::Error& ex) {
        threw = true;
        require(std::string(ex.what()).find(expected) != std::string::npos, message + " error text");
    }
    require(threw, message);
}

bool approx(symft::Complex a, symft::Complex b, double eps = 1e-10) {
    return std::abs(a - b) <= eps;
}

symft::PauliString expected_pauli_image(std::string image) {
    bool negative = false;
    if (!image.empty() && image[0] == '-') {
        negative = true;
        image = image.substr(1);
    }
    symft::PauliString out = symft::pauli_string(image);
    if (negative) {
        out.phase_shift(2);
    }
    return out;
}

symft::Complex deterministic_amplitude(std::size_t basis, int shot = 0) {
    const double re = 0.001 * static_cast<double>((basis % 97) + 1 + 3 * shot);
    const double im = -0.0015 * static_cast<double>(((basis + 5 * static_cast<std::size_t>(shot)) % 89) + 1);
    return {re, im};
}

std::vector<symft::Complex> deterministic_alpha(int k, int shot = 0) {
    const std::size_t dim = std::size_t{1} << k;
    std::vector<symft::Complex> alpha(dim);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        alpha[basis] = deterministic_amplitude(basis, shot);
    }
    return alpha;
}

symft::FactoredInstructionProgram active_only_program(int k) {
    return symft::FactoredInstructionProgram(
        k,
        k,
        std::vector<symft::FactoredInstruction>{},
        k);
}

symft::ApplyPrecomputedActivePauliRotation high_pivot_rotation_instruction(
    const symft::PauliString& pauli,
    double theta,
    bool sign = false) {
    symft::ActivePauliAction action(pauli);
    symft::PrecomputedActivePauliRotationKernel kernel(action, theta);
    const symft::SymbolicBool sign_expr(sign);
    return symft::ApplyPrecomputedActivePauliRotation{
        pauli,
        action,
        kernel,
        theta,
        sign_expr,
        symft::SymbolicBoolEvaluationPlan(sign_expr),
    };
}

symft::PromoteDormantRotation component_test_promotion(double angle) {
    const symft::SymbolicBool sign(false);
    return symft::PromoteDormantRotation{
        angle,
        sign,
        symft::SymbolicBoolEvaluationPlan(sign),
    };
}

symft::ApplyPrecomputedActivePauliRotation component_test_rotation(
    const symft::PauliString& pauli,
    double angle) {
    return high_pivot_rotation_instruction(pauli, angle);
}

symft::MeasurePrecomputedActivePauli component_test_measurement(
    const symft::PauliString& pauli,
    int branch,
    int record) {
    const symft::SymbolicBool outcome = symft::symbolic_bool(branch);
    return symft::MeasurePrecomputedActivePauli{
        pauli,
        symft::PrecomputedActivePauliMeasurementKernel(pauli),
        branch,
        outcome,
        record,
        std::nullopt,
        symft::SymbolicBoolEvaluationPlan(outcome),
    };
}

symft::FactoredInstructionProgram component_test_program() {
    using namespace symft;
    std::vector<FactoredInstruction> instructions;
    instructions.emplace_back(
        component_test_rotation(PauliString(0), 0.19));
    instructions.emplace_back(component_test_promotion(0.17));
    instructions.emplace_back(component_test_promotion(-0.29));
    instructions.emplace_back(component_test_promotion(0.41));

    PauliString merge_late(3);
    merge_late.set_xbit(1);
    merge_late.set_xbit(2);
    instructions.emplace_back(component_test_rotation(merge_late, 0.23));

    PauliString merge_early_into_late(3);
    merge_early_into_late.set_xbit(0);
    merge_early_into_late.set_xbit(1);
    instructions.emplace_back(
        component_test_rotation(merge_early_into_late, -0.31));
    instructions.emplace_back(
        component_test_measurement(merge_early_into_late, 1, 1));
    const SymbolicBool no_symbolic_detector_sign(false);
    instructions.emplace_back(RecordDetector{
        no_symbolic_detector_sign,
        {1},
        1,
        SymbolicBoolEvaluationPlan(no_symbolic_detector_sign),
    });

    // Global order after the first measurement is [q0, q2], while the merged
    // component order is [q2, q0]. The global highest pivot (q2) is therefore
    // local bit zero, exercising explicit non-highest local pivots.
    PauliString reversed_single(2);
    reversed_single.set_zbit(1);
    instructions.emplace_back(
        component_test_measurement(reversed_single, 2, 2));

    PauliString final_x(1);
    final_x.set_xbit(0);
    instructions.emplace_back(
        component_test_measurement(final_x, 3, 3));

    return FactoredInstructionProgram(
        8,
        0,
        std::move(instructions),
        8);
}

void check_high_pivot_single_rotation_kernel(const symft::PauliString& pauli, double theta, int k) {
    using namespace symft;
    auto alpha = deterministic_alpha(k);
    ActiveState expected(k, alpha);
    rotate_pauli(expected, pauli, theta);

    ActivePauliAction action(pauli);
    PrecomputedActivePauliRotationKernel kernel(action, theta);
    require(kernel.pair_bit == static_cast<unsigned>(k - 1), "active rotation kernel uses highest X pivot bit");

    ActiveState aos_fast(k, alpha);
    rotate_pauli(aos_fast, kernel, false);
    for (std::size_t basis = 0; basis < alpha.size(); ++basis) {
        require(approx(aos_fast.alpha[basis], expected.alpha[basis], 1e-9), "AoS high-pivot rotation kernel matches generic rotation");
    }

    const auto program = active_only_program(k);
    FactoredExecutorState runtime(program, 123);
    for (std::size_t basis = 0; basis < alpha.size(); ++basis) {
        runtime.active_re[basis] = alpha[basis].real();
        runtime.active_im[basis] = alpha[basis].imag();
    }
    auto instruction = high_pivot_rotation_instruction(pauli, theta);
    execute_instruction_in_place(runtime, FactoredInstruction(instruction));
    for (std::size_t basis = 0; basis < alpha.size(); ++basis) {
        require(
            approx({runtime.active_re[basis], runtime.active_im[basis]}, expected.alpha[basis], 1e-9),
            "SoA high-pivot rotation kernel matches generic rotation");
    }
}

symft::PauliString uniform_xmask_pauli(int k, int pivot, std::uint64_t lower_mask) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pivot);
    for (int bit = 0; bit < pivot; ++bit) {
        if ((lower_mask & (std::uint64_t{1} << bit)) != 0) {
            pauli.set_xbit(bit);
        }
    }
    return pauli;
}

symft::PauliString real_xmask_pauli(int k, int pivot, std::uint64_t lower_mask) {
    symft::PauliString pauli = uniform_xmask_pauli(k, pivot, lower_mask);
    pauli.set_zbit(pivot);
    pauli.set_phase(1);
    return pauli;
}

symft::PauliString general_xmask_pauli(int k, int pivot, std::uint64_t lower_mask) {
    symft::PauliString pauli = uniform_xmask_pauli(k, pivot, lower_mask);
    pauli.set_zbit(pivot - 1);
    return pauli;
}

double reference_active_measurement_probability(
    const std::vector<symft::Complex>& alpha,
    const symft::PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const auto insert_zero = [](std::size_t packed, int bit) {
        const std::size_t low_mask = (std::size_t{1} << bit) - 1;
        return (packed & low_mask) | ((packed & ~low_mask) << 1);
    };
    double probability = 0.0;
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        symft::Complex value;
        const std::size_t source0 = insert_zero(idx, kernel.pivot);
        if (kernel.is_diagonal) {
            const int parity = std::popcount(static_cast<std::uint64_t>(source0) & kernel.z_without_pivot) & 1;
            const int pivot_value = kernel.diagonal_phase_bit ^ parity ^ static_cast<int>(branch);
            const std::size_t source = source0 | (std::size_t{static_cast<unsigned>(pivot_value)} << kernel.pivot);
            value = alpha[source];
        } else {
            const std::size_t source1 = source0 ^ static_cast<std::size_t>(kernel.action.xmask);
            const bool odd = (std::popcount(static_cast<std::uint64_t>(source0) & kernel.action.zmask) & 1) != 0;
            const symft::Complex eta = odd ? kernel.action.odd_phase : kernel.action.even_phase;
            const symft::Complex coefficient1 = (branch ? -1.0 : 1.0) * std::conj(eta) * inv_sqrt2;
            value = inv_sqrt2 * alpha[source0] + coefficient1 * alpha[source1];
        }
        probability += std::norm(value);
    }
    return probability;
}

std::vector<symft::Complex> reference_active_measurement_projection(
    const std::vector<symft::Complex>& alpha,
    const symft::PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const auto insert_zero = [](std::size_t packed, int bit) {
        const std::size_t low_mask = (std::size_t{1} << bit) - 1;
        return (packed & low_mask) | ((packed & ~low_mask) << 1);
    };
    const double invnorm = 1.0 / std::sqrt(probability);
    std::vector<symft::Complex> out(kernel.out_dim);
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        symft::Complex value;
        const std::size_t source0 = insert_zero(idx, kernel.pivot);
        if (kernel.is_diagonal) {
            const int parity = std::popcount(static_cast<std::uint64_t>(source0) & kernel.z_without_pivot) & 1;
            const int pivot_value = kernel.diagonal_phase_bit ^ parity ^ static_cast<int>(branch);
            const std::size_t source = source0 | (std::size_t{static_cast<unsigned>(pivot_value)} << kernel.pivot);
            value = alpha[source];
        } else {
            const std::size_t source1 = source0 ^ static_cast<std::size_t>(kernel.action.xmask);
            const bool odd = (std::popcount(static_cast<std::uint64_t>(source0) & kernel.action.zmask) & 1) != 0;
            const symft::Complex eta = odd ? kernel.action.odd_phase : kernel.action.even_phase;
            const symft::Complex coefficient1 = (branch ? -1.0 : 1.0) * std::conj(eta) * inv_sqrt2;
            value = inv_sqrt2 * alpha[source0] + coefficient1 * alpha[source1];
        }
        out[idx] = value * invnorm;
    }
    return out;
}

void check_active_measurement_projection_kernel(const symft::PauliString& pauli, int k, std::uint64_t seed) {
    using namespace symft;
    auto alpha = deterministic_alpha(k);
    double norm2 = 0.0;
    for (const auto& value : alpha) {
        norm2 += std::norm(value);
    }
    const double invnorm = 1.0 / std::sqrt(norm2);
    for (auto& value : alpha) {
        value *= invnorm;
    }
    ActivePauliAction action(pauli);
    PrecomputedActivePauliMeasurementKernel kernel(action);
    require(!kernel.is_diagonal && kernel.pivot == k - 1, "active measurement test uses highest non-diagonal pivot");
    const double prob_true = reference_active_measurement_probability(alpha, kernel, true);

    MeasurePrecomputedActivePauli instruction{
        pauli,
        kernel,
        1,
        symbolic_bool(1),
        std::nullopt,
        std::nullopt,
        SymbolicBoolEvaluationPlan(symbolic_bool(1)),
    };
    FactoredInstructionProgram program(k, k, {FactoredInstruction(instruction)}, k);
    FactoredExecutorState runtime(program, seed);
    for (std::size_t basis = 0; basis < alpha.size(); ++basis) {
        runtime.active_re[basis] = alpha[basis].real();
        runtime.active_im[basis] = alpha[basis].imag();
    }

    execute_instruction_in_place(runtime, FactoredInstruction(instruction));
    const bool branch = (runtime.value_words[0] & std::uint64_t{1}) != 0;
    const double probability = branch ? prob_true : 1.0 - prob_true;
    const auto expected = reference_active_measurement_projection(alpha, kernel, branch, probability);
    require(runtime.k == k - 1, "active measurement projection removes one active qubit");
    for (std::size_t basis = 0; basis < expected.size(); ++basis) {
        require(
            approx({runtime.active_re[basis], runtime.active_im[basis]}, expected[basis], 1e-9),
            "partner-permuted active measurement projection matches reference");
    }
}

void check_diagonal_active_measurement_projection_kernel(const symft::PauliString& pauli, int k, std::uint64_t seed) {
    using namespace symft;
    auto alpha = deterministic_alpha(k);
    double norm2 = 0.0;
    for (const auto& value : alpha) {
        norm2 += std::norm(value);
    }
    const double invnorm = 1.0 / std::sqrt(norm2);
    for (auto& value : alpha) {
        value *= invnorm;
    }
    ActivePauliAction action(pauli);
    PrecomputedActivePauliMeasurementKernel kernel(action);
    require(kernel.is_diagonal, "active diagonal measurement test uses a diagonal Pauli");
    const double prob_true = reference_active_measurement_probability(alpha, kernel, true);

    MeasurePrecomputedActivePauli instruction{
        pauli,
        kernel,
        1,
        symbolic_bool(1),
        std::nullopt,
        std::nullopt,
        SymbolicBoolEvaluationPlan(symbolic_bool(1)),
    };
    FactoredInstructionProgram program(k, k, {FactoredInstruction(instruction)}, k);
    FactoredExecutorState runtime(program, seed);
    for (std::size_t basis = 0; basis < alpha.size(); ++basis) {
        runtime.active_re[basis] = alpha[basis].real();
        runtime.active_im[basis] = alpha[basis].imag();
    }

    execute_instruction_in_place(runtime, FactoredInstruction(instruction));
    const bool branch = (runtime.value_words[0] & std::uint64_t{1}) != 0;
    const double probability = branch ? prob_true : 1.0 - prob_true;
    const auto expected = reference_active_measurement_projection(alpha, kernel, branch, probability);
    require(runtime.k == k - 1, "active diagonal measurement projection removes one active qubit");
    for (std::size_t basis = 0; basis < expected.size(); ++basis) {
        require(
            approx({runtime.active_re[basis], runtime.active_im[basis]}, expected[basis], 1e-9),
            "permuted active diagonal measurement projection matches reference");
    }
}

void check_batch_active_measurement_projection_kernel(
    const symft::PauliString& pauli,
    int k,
    std::uint64_t seed,
    int shots = 7,
    bool shot_major = false) {
    using namespace symft;
    ActivePauliAction action(pauli);
    PrecomputedActivePauliMeasurementKernel kernel(action);
    MeasurePrecomputedActivePauli instruction{
        pauli,
        kernel,
        1,
        symbolic_bool(1),
        std::nullopt,
        std::nullopt,
        SymbolicBoolEvaluationPlan(symbolic_bool(1)),
    };
    const FactoredInstructionProgram program(k, k, {FactoredInstruction(instruction)}, k);
    BatchFactoredExecutorState runtime(program, shots, seed);
    if (shot_major) {
        runtime.dense_shot_major_active = true;
        reset_batch_executor(runtime, program, shots);
    }
    require(runtime.active_pitch == padded_batch_active_pitch(shots), "batch measurement test uses expected active pitch");
    const std::size_t dim = std::size_t{1} << k;
    std::vector<std::vector<Complex>> alpha_by_shot;
    alpha_by_shot.reserve(shots);
    for (int shot = 0; shot < shots; ++shot) {
        auto alpha = deterministic_alpha(k, shot);
        double norm2 = 0.0;
        for (const auto& value : alpha) {
            norm2 += std::norm(value);
        }
        const double invnorm = 1.0 / std::sqrt(norm2);
        for (auto& value : alpha) {
            value *= invnorm;
        }
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const std::size_t offset = batch_active_offset(runtime, basis, shot);
            runtime.active_re[offset] = alpha[basis].real();
            runtime.active_im[offset] = alpha[basis].imag();
        }
        alpha_by_shot.push_back(std::move(alpha));
    }
    measure_precomputed_active_pauli_branch_batch(runtime, kernel, 1);
    require(runtime.k == k - 1, "batch active measurement projection removes one active qubit");
    const std::size_t out_dim = dim >> 1;
    for (int shot = 0; shot < shots; ++shot) {
        const bool branch =
            (runtime.value_words[batch_condition_offset(runtime, 1, batch_shot_word(shot))] & batch_shot_mask(shot)) != 0;
        const double prob_true = reference_active_measurement_probability(alpha_by_shot[static_cast<std::size_t>(shot)], kernel, true);
        const double probability = branch ? prob_true : 1.0 - prob_true;
        const auto expected =
            reference_active_measurement_projection(alpha_by_shot[static_cast<std::size_t>(shot)], kernel, branch, probability);
        for (std::size_t basis = 0; basis < out_dim; ++basis) {
            const std::size_t offset = batch_active_offset(runtime, basis, shot);
            const Complex actual{runtime.active_re[offset], runtime.active_im[offset]};
            require(
                approx(actual, expected[basis], 1e-9),
                std::string("batch active measurement projection matches reference; shots=") +
                    std::to_string(shots) +
                    " shot=" + std::to_string(shot) +
                    " basis=" + std::to_string(basis) +
                    " diagonal=" + (kernel.is_diagonal ? "true" : "false") +
                    " shot_major=" + (shot_major ? "true" : "false"));
        }
    }
}

void check_high_pivot_batch_rotation_kernel(
    const symft::PauliString& pauli,
    double theta,
    int k,
    bool mixed_signs,
    int shots = 5,
    bool shot_major = false) {
    using namespace symft;
    const auto program = active_only_program(k);
    BatchFactoredExecutorState runtime(program, shots, 321);
    if (shot_major) {
        runtime.dense_shot_major_active = true;
        reset_batch_executor(runtime, program, shots);
    }
    require(runtime.active_pitch == padded_batch_active_pitch(shots), "batch rotation test uses expected active pitch");
    const std::size_t dim = std::size_t{1} << k;
    std::vector<std::vector<Complex>> expected_by_shot;
    expected_by_shot.reserve(shots);
    for (int shot = 0; shot < shots; ++shot) {
        auto alpha = deterministic_alpha(k, shot);
        ActiveState expected(k, alpha);
        const bool sign = mixed_signs && ((shot & 1) != 0);
        rotate_pauli(expected, pauli, sign ? -theta : theta);
        expected_by_shot.push_back(std::move(expected.alpha));
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const std::size_t offset = batch_active_offset(runtime, basis, shot);
            runtime.active_re[offset] = alpha[basis].real();
            runtime.active_im[offset] = alpha[basis].imag();
        }
    }

    ActivePauliAction action(pauli);
    PrecomputedActivePauliRotationKernel kernel(action, theta);
    require(kernel.pair_bit == static_cast<unsigned>(k - 1), "batch rotation kernel uses highest X pivot bit");
    std::vector<std::uint64_t> sign_bits(runtime.batch_words, 0);
    if (mixed_signs) {
        for (int shot = 0; shot < shots; ++shot) {
            if ((shot & 1) != 0) {
                sign_bits[static_cast<std::size_t>(shot >> 6)] |= std::uint64_t{1} << (shot & 63);
            }
        }
    }
    rotate_pauli_batch(runtime, kernel, sign_bits);
    for (int shot = 0; shot < shots; ++shot) {
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const std::size_t offset = batch_active_offset(runtime, basis, shot);
            require(
                approx({runtime.active_re[offset], runtime.active_im[offset]}, expected_by_shot[static_cast<std::size_t>(shot)][basis], 1e-9),
                std::string("batch high-pivot rotation kernel matches generic rotation; shot_major=") +
                    (shot_major ? "true" : "false"));
        }
    }
}

symft::BatchDetectorPostselectionResult execute_expression_postselected_for_test(
    symft::BatchFactoredExecutorState& runtime,
    const symft::FactoredInstructionProgram& program,
    const symft::PackedPresampledExogenous& samples,
    symft::BatchDetectorPostselectionScratch& scratch,
    const symft::BatchDetectorPostselectionOptions& options = {}) {
    symft::PresampledExpressionPlan expression_plan;
    symft::prepare_presampled_expression_plan(expression_plan, program, samples);
    symft::PresampledExpressionBlock expression_block;
    symft::evaluate_presampled_expression_block(
        expression_block,
        expression_plan,
        samples);
    return symft::execute_batch_postselected_in_place(
        runtime,
        program,
        expression_plan,
        expression_block,
        0,
        scratch,
        options);
}

symft::FactoredInstructionProgram planned_stim_program(
    const symft::StimParseResult& parsed) {
    return symft::plan_stim_factored_program(parsed);
}

void test_pauli_algebra() {
    using namespace symft;
    require(pauli_x(1, 0) * pauli_x(1, 0) == pauli_identity(1), "X squares to I");
    require(pauli_y(1, 0) * pauli_y(1, 0) == pauli_identity(1), "Y squares to I");
    auto xz = pauli_y(1, 0);
    xz.phase_shift(3);
    require(pauli_x(1, 0) * pauli_z(1, 0) == xz, "packed XZ phase");
    require(pauli_anticommutes(pauli_x(1, 0), pauli_z(1, 0)), "X anticommutes with Z");
    const auto ps = pauli_string("IXYZ");
    require(!ps.xbit(0) && !ps.zbit(0), "I bits");
    require(ps.xbit(1) && !ps.zbit(1), "X bits");
    require(ps.xbit(2) && ps.zbit(2), "Y bits");
    require(!ps.xbit(3) && ps.zbit(3), "Z bits");
}

void test_clifford_frame() {
    using namespace symft;
    CliffordFrame cf(2);
    left_CX(cf, 0, 1);
    require(preimage(cf, pauli_x(2, 0)) == pauli_string("XX"), "CX maps Xc");
    require(preimage(cf, pauli_z(2, 1)) == pauli_string("ZZ"), "CX maps Zt");

    CliffordFrame h(1);
    left_H(h, 0);
    left_S(h, 0);
    require(preimage(h, pauli_x(1, 0)) == pauli_y(1, 0), "left composition order");
}

void test_extended_clifford_frame_images() {
    using namespace symft;
    using SingleGate = void (*)(CliffordFrame&, int);
    struct SingleCase {
        const char* name;
        SingleGate apply;
        const char* x_image;
        const char* z_image;
    };
    const std::vector<SingleCase> single_cases{
        {"H_NXY", static_cast<SingleGate>(&left_H_NXY), "-Y", "-Z"},
        {"H_NXZ", static_cast<SingleGate>(&left_H_NXZ), "-Z", "-X"},
        {"H_NYZ", static_cast<SingleGate>(&left_H_NYZ), "-X", "-Y"},
        {"H_XY", static_cast<SingleGate>(&left_H_XY), "Y", "-Z"},
        {"H_YZ", static_cast<SingleGate>(&left_H_YZ), "-X", "Y"},
        {"C_NXYZ", static_cast<SingleGate>(&left_C_NXYZ), "-Y", "-X"},
        {"C_NZYX", static_cast<SingleGate>(&left_C_NZYX), "-Z", "-Y"},
        {"C_XNYZ", static_cast<SingleGate>(&left_C_XNYZ), "-Y", "X"},
        {"C_XYNZ", static_cast<SingleGate>(&left_C_XYNZ), "Y", "-X"},
        {"C_XYZ", static_cast<SingleGate>(&left_C_XYZ), "Y", "X"},
        {"C_ZNYX", static_cast<SingleGate>(&left_C_ZNYX), "Z", "-Y"},
        {"C_ZYNX", static_cast<SingleGate>(&left_C_ZYNX), "-Z", "Y"},
        {"C_ZYX", static_cast<SingleGate>(&left_C_ZYX), "Z", "Y"},
        {"SQRT_X", static_cast<SingleGate>(&left_SQRT_X), "X", "-Y"},
        {"SQRT_X_DAG", static_cast<SingleGate>(&left_SQRT_X_DAG), "X", "Y"},
        {"SQRT_Y", static_cast<SingleGate>(&left_SQRT_Y), "-Z", "X"},
        {"SQRT_Y_DAG", static_cast<SingleGate>(&left_SQRT_Y_DAG), "Z", "-X"},
        {"Y", static_cast<SingleGate>(&left_Y), "-X", "-Z"},
    };
    for (const auto& c : single_cases) {
        CliffordFrame frame(1);
        c.apply(frame, 0);
        require(
            preimage(frame, pauli_x(1, 0)) == expected_pauli_image(c.x_image),
            std::string(c.name) + " X generator image");
        require(
            preimage(frame, pauli_z(1, 0)) == expected_pauli_image(c.z_image),
            std::string(c.name) + " Z generator image");
    }

    using TwoGate = void (*)(CliffordFrame&, int, int);
    struct TwoCase {
        const char* name;
        TwoGate apply;
        const char* xa_image;
        const char* za_image;
        const char* xb_image;
        const char* zb_image;
    };
    const std::vector<TwoCase> two_cases{
        {"CY", static_cast<TwoGate>(&left_CY), "XY", "Z_", "ZX", "ZZ"},
        {"CXSWAP", static_cast<TwoGate>(&left_CXSWAP), "XX", "_Z", "X_", "ZZ"},
        {"CZSWAP", static_cast<TwoGate>(&left_CZSWAP), "ZX", "_Z", "XZ", "Z_"},
        {"ISWAP", static_cast<TwoGate>(&left_ISWAP), "ZY", "_Z", "YZ", "Z_"},
        {"ISWAP_DAG", static_cast<TwoGate>(&left_ISWAP_DAG), "-ZY", "_Z", "-YZ", "Z_"},
        {"SQRT_XX", static_cast<TwoGate>(&left_SQRT_XX), "X_", "-YX", "_X", "-XY"},
        {"SQRT_XX_DAG", static_cast<TwoGate>(&left_SQRT_XX_DAG), "X_", "YX", "_X", "XY"},
        {"SQRT_YY", static_cast<TwoGate>(&left_SQRT_YY), "-ZY", "XY", "-YZ", "YX"},
        {"SQRT_YY_DAG", static_cast<TwoGate>(&left_SQRT_YY_DAG), "ZY", "-XY", "YZ", "-YX"},
        {"SQRT_ZZ", static_cast<TwoGate>(&left_SQRT_ZZ), "YZ", "Z_", "ZY", "_Z"},
        {"SQRT_ZZ_DAG", static_cast<TwoGate>(&left_SQRT_ZZ_DAG), "-YZ", "Z_", "-ZY", "_Z"},
        {"SWAPCX", static_cast<TwoGate>(&left_SWAPCX), "_X", "ZZ", "XX", "Z_"},
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
            preimage(frame, pauli_x(2, 0)) == expected_pauli_image(c.xa_image),
            std::string(c.name) + " X_ generator image");
        require(
            preimage(frame, pauli_z(2, 0)) == expected_pauli_image(c.za_image),
            std::string(c.name) + " Z_ generator image");
        require(
            preimage(frame, pauli_x(2, 1)) == expected_pauli_image(c.xb_image),
            std::string(c.name) + " _X generator image");
        require(
            preimage(frame, pauli_z(2, 1)) == expected_pauli_image(c.zb_image),
            std::string(c.name) + " _Z generator image");
    }
}

void test_active_rotation() {
    using namespace symft;
    ActiveState st(1);
    rotate_pauli(st, pauli_x(1, 0), M_PI / 4.0);
    require(approx(st.alpha[0], {std::cos(M_PI / 4.0), 0.0}), "X rotation amplitude 0");
    require(approx(st.alpha[1], {0.0, -std::sin(M_PI / 4.0)}), "X rotation amplitude 1");

    ActiveState y(1);
    rotate_pauli(y, pauli_y(1, 0), M_PI / 6.0);
    require(approx(y.alpha[0], {std::cos(M_PI / 6.0), 0.0}), "Y rotation amplitude 0");
    require(approx(y.alpha[1], {std::sin(M_PI / 6.0), 0.0}), "Y rotation amplitude 1");

    ActiveState z(1, {{3.0, 4.0}, {5.0, -2.0}});
    auto before = z.alpha;
    ActivePauliAction action(pauli_z(1, 0));
    PrecomputedActivePauliRotationKernel kernel(action, 0.31);
    rotate_pauli(z, kernel, false);
    require(approx(z.alpha[0], std::exp(symft::Complex(0.0, -0.31)) * before[0]), "precomputed Z rotation 0");
    require(approx(z.alpha[1], std::exp(symft::Complex(0.0, 0.31)) * before[1]), "precomputed Z rotation 1");

    ActiveState x_expected(1, {{0.2, -0.3}, {0.7, 0.4}});
    ActiveState x_fast(1, x_expected.alpha);
    const auto x_pauli = pauli_x(1, 0);
    rotate_pauli(x_expected, x_pauli, 0.23);
    rotate_pauli(x_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(x_pauli), 0.23), false);
    require(approx(x_fast.alpha[0], x_expected.alpha[0]) && approx(x_fast.alpha[1], x_expected.alpha[1]), "precomputed X fast rotation");

    ActiveState y_expected(1, {{0.1, 0.8}, {-0.6, 0.2}});
    ActiveState y_fast(1, y_expected.alpha);
    const auto y_pauli = pauli_y(1, 0);
    rotate_pauli(y_expected, y_pauli, -0.19);
    rotate_pauli(y_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(y_pauli), 0.19), true);
    require(approx(y_fast.alpha[0], y_expected.alpha[0]) && approx(y_fast.alpha[1], y_expected.alpha[1]), "precomputed Y fast signed rotation");

    std::vector<Complex> alpha6(64);
    for (std::size_t i = 0; i < alpha6.size(); ++i) {
        alpha6[i] = {0.01 * static_cast<double>(i + 1), -0.02 * static_cast<double>((i % 7) + 1)};
    }
    ActiveState x6_expected(6, alpha6);
    ActiveState x6_fast(6, alpha6);
    const auto x6_pauli = pauli_x(6, 5);
    rotate_pauli(x6_expected, x6_pauli, 0.17);
    rotate_pauli(x6_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(x6_pauli), 0.17), false);
    for (std::size_t i = 0; i < alpha6.size(); ++i) {
        require(approx(x6_fast.alpha[i], x6_expected.alpha[i]), "precomputed multi-qubit X fast rotation");
    }

    ActiveState y6_expected(6, alpha6);
    ActiveState y6_fast(6, alpha6);
    const auto y6_pauli = pauli_y(6, 4);
    rotate_pauli(y6_expected, y6_pauli, 0.11);
    rotate_pauli(y6_fast, PrecomputedActivePauliRotationKernel(ActivePauliAction(y6_pauli), 0.11), false);
    for (std::size_t i = 0; i < alpha6.size(); ++i) {
        require(approx(y6_fast.alpha[i], y6_expected.alpha[i]), "precomputed multi-qubit Y fast rotation");
    }
}

void test_high_pivot_selection() {
    using namespace symft;
    const auto x1x3 = pauli_x(4, 1) * pauli_x(4, 3);
    const PrecomputedActivePauliRotationKernel rotation_kernel(ActivePauliAction(x1x3), 0.125);
    require(rotation_kernel.pair_bit == 3, "active rotation uses highest X pivot bit");

    const PrecomputedActivePauliMeasurementKernel diagonal_kernel(pauli_z(4, 0) * pauli_z(4, 3));
    require(diagonal_kernel.is_diagonal && diagonal_kernel.pivot == 3, "diagonal active measurement uses highest Z pivot");

    const PrecomputedActivePauliMeasurementKernel nondiagonal_kernel(pauli_x(4, 1) * pauli_z(4, 2) * pauli_x(4, 3));
    require(!nondiagonal_kernel.is_diagonal && nondiagonal_kernel.pivot == 3, "nondiagonal active measurement uses highest X pivot");

    PendingFactoredState pending(4, 0);
    pending.pending_operations.push_back(PendingPauliRotation{0.25, SymbolicPauliString(pauli_x(4, 0) * pauli_x(4, 2))});
    pending.pending_operations.push_back(PendingPauliMeasurement{SymbolicPauliString(pauli_z(4, 2)), std::nullopt, std::nullopt});
    process_next_pending_operation(pending);
    require(pending.k == 1, "dormant rotation promotion creates one active qubit");
    const auto& transformed = std::get<PendingPauliMeasurement>(pending.pending_operations.front()).pauli.pauli;
    require(transformed == pauli_z(4, 0), "dormant promotion uses highest dormant pivot");
}

symft::FactoredInstructionProgram plan_without_pending_optimization(
    symft::PendingFactoredState state) {
    while (symft::has_pending_operations(state)) {
        symft::process_next_pending_operation(state);
    }
    return symft::factored_instruction_program(std::move(state));
}

std::vector<int> pending_record_order(const symft::PendingFactoredState& state) {
    std::vector<int> records;
    for (const auto& operation : state.pending_operations) {
        std::visit(
            [&](const auto& typed) {
                if constexpr (requires { typed.record; }) {
                    if (typed.record) {
                        records.push_back(*typed.record);
                    }
                }
            },
            operation);
    }
    return records;
}

void require_pending_record_conditions_are_causal(
    const symft::PendingFactoredState& state,
    const std::string& context) {
    std::vector<std::pair<int, std::size_t>> producers;
    for (std::size_t index = 0; index < state.pending_operations.size(); ++index) {
        std::visit(
            [&](const auto& typed) {
                if constexpr (requires { typed.record_condition; }) {
                    if (typed.record_condition) {
                        producers.emplace_back(*typed.record_condition, index);
                    }
                }
            },
            state.pending_operations[index]);
    }

    auto require_expression_is_causal = [&](const symft::SymbolicBool& expression, std::size_t use_index) {
        for (int condition : expression.conditions) {
            for (const auto& [producer, producer_index] : producers) {
                if (producer == condition) {
                    require(producer_index < use_index,
                            context + " does not use a measurement condition before assignment");
                }
            }
        }
    };
    for (std::size_t index = 0; index < state.pending_operations.size(); ++index) {
        std::visit(
            [&](const auto& typed) {
                using T = std::decay_t<decltype(typed)>;
                if constexpr (std::is_same_v<T, symft::PendingClassicalRecord>) {
                    require_expression_is_causal(typed.outcome, index);
                } else {
                    require_expression_is_causal(typed.pauli.sign, index);
                }
            },
            state.pending_operations[index]);
    }
}

void test_pending_operation_optimizer() {
    using namespace symft;
    {
        PendingFactoredState pending(2, 0);
        const SymbolicBool sign(false, {1});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.1, SymbolicPauliString(pauli_x(2, 0), sign)});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.2, SymbolicPauliString(pauli_z(2, 1))});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.3, SymbolicPauliString(pauli_x(2, 0), sign)});
        pending.pending_operations.push_back(
            PendingPauliMeasurement{SymbolicPauliString(pauli_x(2, 0)), 1, 2});

        const auto stats = optimize_pending_operations(pending);
        require(stats.input_operations == 4 && stats.output_operations == 3,
                "pending optimizer removes one fused rotation");
        require(stats.fused_rotations == 1 && stats.cancelled_rotations == 0,
                "pending optimizer reports commuting rotation fusion");
        require(stats.measurement_left_swaps == 2,
                "fusion enables earlier commuting measurement scheduling in its segment");
        require(std::holds_alternative<PendingPauliMeasurement>(pending.pending_operations[0]),
                "commuting measurement is scheduled before the fused rotation run");

        bool found_fused = false;
        for (const auto& operation : pending.pending_operations) {
            if (const auto* rotation = std::get_if<PendingPauliRotation>(&operation);
                rotation != nullptr && rotation->pauli.pauli.same_body(pauli_x(2, 0))) {
                require(std::abs(rotation->kernel_angle - 0.4) < 1e-12,
                        "commuting rotations add their angles");
                require(rotation->pauli.sign == sign,
                        "commuting rotation fusion preserves the symbolic sign");
                found_fused = true;
            }
        }
        require(found_fused, "pending optimizer retains the fused rotation");
    }
    {
        PendingFactoredState pending(1, 0);
        pending.pending_operations.push_back(
            PendingPauliRotation{0.1, SymbolicPauliString(pauli_x(1, 0))});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.2, SymbolicPauliString(pauli_z(1, 0))});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.3, SymbolicPauliString(pauli_x(1, 0))});
        const auto stats = optimize_pending_operations(pending);
        require(stats.fused_rotations == 0 && pending.pending_operations.size() == 3,
                "anti-commuting rotation blocks fusion");
    }
    {
        PendingFactoredState pending(1, 0);
        pending.pending_operations.push_back(PendingPauliRotation{
            0.1,
            SymbolicPauliString(pauli_x(1, 0), SymbolicBool(false, {1}))});
        pending.pending_operations.push_back(PendingPauliRotation{
            0.2,
            SymbolicPauliString(pauli_x(1, 0), SymbolicBool(false, {2}))});
        pending.pending_operations.push_back(
            PendingPauliMeasurement{SymbolicPauliString(pauli_z(1, 0)), 1, 3});
        const auto stats = optimize_pending_operations(pending);
        require(stats.fused_rotations == 0,
                "different symbolic rotation signs do not fuse");
        require(stats.measurement_left_swaps == 0 &&
                    std::holds_alternative<PendingPauliRotation>(pending.pending_operations[0]),
                "anti-commuting measurement does not move left");
    }
    {
        PendingFactoredState pending(2, 0);
        pending.pending_operations.push_back(
            PendingPauliRotation{0.1, SymbolicPauliString(pauli_x(2, 0))});
        pending.pending_operations.push_back(
            PendingPauliMeasurement{SymbolicPauliString(pauli_z(2, 1)), 1, 2});
        const auto stats = optimize_pending_operations(pending);
        require(stats.fused_rotations == 0 && stats.measurement_left_swaps == 0 &&
                    std::holds_alternative<PendingPauliRotation>(pending.pending_operations[0]),
                "unrelated commuting measurement does not cause pure schedule churn");
    }
    {
        PendingFactoredState pending(1, 0);
        const SymbolicBool sign(false, {1});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.2, SymbolicPauliString(pauli_x(1, 0), sign)});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.3, SymbolicPauliString(pauli_x(1, 0), !sign)});
        const auto stats = optimize_pending_operations(pending);
        require(stats.fused_rotations == 1 && pending.pending_operations.size() == 1,
                "opposite symbolic signs still fuse");
        const auto& fused = std::get<PendingPauliRotation>(pending.pending_operations[0]);
        require(std::abs(fused.kernel_angle - 0.1) < 1e-12 && fused.pauli.sign == !sign,
                "opposite symbolic signs subtract rotation angles");
    }
    {
        PendingFactoredState pending(1, 0);
        pending.pending_operations.push_back(
            PendingPauliRotation{0.25, SymbolicPauliString(pauli_y(1, 0))});
        pending.pending_operations.push_back(
            PendingPauliRotation{-0.25, SymbolicPauliString(pauli_y(1, 0))});
        const auto stats = optimize_pending_operations(pending);
        require(stats.fused_rotations == 1 && stats.cancelled_rotations == 1 &&
                    pending.pending_operations.empty(),
                "exact inverse rotations cancel completely");
    }
    {
        PendingFactoredState pending(1, 0);
        pending.pending_operations.push_back(
            PendingPauliRotation{0.1, SymbolicPauliString(pauli_x(1, 0))});
        pending.pending_operations.push_back(
            PendingPauliRotation{0.2, SymbolicPauliString(pauli_x(1, 0))});
        const auto stats = optimize_pending_operations(pending, {1});
        require(stats.fused_rotations == 0 && pending.pending_operations.size() == 2,
                "preserved pending prefix blocks cross-detector fusion");
        require(stats.prefix_remap[1] == 1 && stats.prefix_remap[2] == 2,
                "preserved pending prefixes are remapped");
    }
    {
        PendingFactoredState pending(1, 0);
        pending.pending_operations.push_back(
            PendingPauliRotation{0.1, SymbolicPauliString(pauli_x(1, 0))});
        pending.pending_operations.push_back(
            PendingPauliMeasurement{SymbolicPauliString(pauli_x(1, 0)), 1, 2});
        const auto stats = optimize_pending_operations(pending, {1});
        require(stats.measurement_left_swaps == 0 &&
                    std::holds_alternative<PendingPauliRotation>(pending.pending_operations[0]),
                "detector prefix prevents a measurement from moving across its time boundary");
    }
    {
        PendingFactoredState pending(1, 0);
        pending.pending_operations.push_back(
            PendingPauliRotation{0.1, SymbolicPauliString(pauli_x(1, 0))});
        pending.pending_operations.push_back(
            PendingClassicalRecord{SymbolicBool(false), 1, std::nullopt});
        pending.pending_operations.push_back(
            PendingPauliMeasurement{SymbolicPauliString(pauli_x(1, 0)), 2, 3});
        const auto stats = optimize_pending_operations(pending);
        require(stats.measurement_left_swaps == 0 &&
                    std::holds_alternative<PendingClassicalRecord>(pending.pending_operations[1]),
                "measurement movement preserves classical record event ordering");
    }
}

void test_pending_operation_optimizer_end_to_end() {
    using namespace symft;
    const std::string fused_circuit =
        "H 0\n"
        "R_Z(0.1) 0\n"
        "R_Z(0.2) 0\n"
        "H 0\n"
        "M 0\n";
    const auto reference_parsed = parse_stim_text(fused_circuit);
    const auto optimized_parsed = parse_stim_text(fused_circuit);
    const auto reference = plan_without_pending_optimization(PendingFactoredState(reference_parsed.state));
    const auto optimized = plan_factored_updates(PendingFactoredState(optimized_parsed.state));
    require(optimized.instructions.size() < reference.instructions.size(),
            "end-to-end optimizer fuses repeated physical rotations");
    require(sample_measurements(reference, 512, 811) == sample_measurements(optimized, 512, 811),
            "fused pending program preserves sampled measurement records");

    const std::string squeezed_circuit =
        "H 0\n"
        "R_Z(0.1) 0\n"
        "M 0\n";
    const auto squeezed_reference_parsed = parse_stim_text(squeezed_circuit);
    const auto squeezed_optimized_parsed = parse_stim_text(squeezed_circuit);
    const auto squeezed_reference =
        plan_without_pending_optimization(PendingFactoredState(squeezed_reference_parsed.state));
    const auto squeezed_optimized =
        plan_factored_updates(PendingFactoredState(squeezed_optimized_parsed.state));
    require(squeezed_reference.max_k == 1 && squeezed_optimized.max_k == 0,
            "early commuting measurement removes an unnecessary active promotion");
    require(
        sample_measurements(squeezed_reference, 512, 821) ==
            sample_measurements(squeezed_optimized, 512, 821),
        "measurement squeezing preserves sampled measurement records");

    const std::string feedback_circuit =
        "H 0\n"
        "R_Z(0.125) 0\n"
        "M 0\n"
        "CX rec[-1] 1\n"
        "M 1\n";
    const auto feedback_reference_parsed = parse_stim_text(feedback_circuit);
    const auto feedback_optimized_parsed = parse_stim_text(feedback_circuit);
    PendingFactoredState feedback_reference_pending(feedback_reference_parsed.state);
    PendingFactoredState feedback_optimized_pending(feedback_optimized_parsed.state);
    const auto feedback_record_order = pending_record_order(feedback_reference_pending);
    require_pending_record_conditions_are_causal(
        feedback_reference_pending,
        "unoptimized feedback program");
    const auto feedback_stats = optimize_pending_operations(feedback_optimized_pending);
    require(feedback_stats.measurement_left_swaps > 0,
            "feedback test moves its commuting measurement earlier");
    require(pending_record_order(feedback_optimized_pending) == feedback_record_order,
            "measurement movement preserves externally visible record order");
    require_pending_record_conditions_are_causal(
        feedback_optimized_pending,
        "optimized feedback program");
    const auto feedback_reference =
        plan_without_pending_optimization(std::move(feedback_reference_pending));
    const auto feedback_optimized = plan_factored_updates(std::move(feedback_optimized_pending));
    require(
        sample_measurements(feedback_reference, 1024, 827) ==
            sample_measurements(feedback_optimized, 1024, 827),
        "early measurement preserves measurement-conditioned feedback records");

    const std::string multiqubit_circuit =
        "H 0\n"
        "R_XX(0.125) 0 1\n"
        "MXX 0 1\n"
        "CX rec[-1] 2\n"
        "M 2\n";
    const auto multiqubit_reference_parsed = parse_stim_text(multiqubit_circuit);
    const auto multiqubit_optimized_parsed = parse_stim_text(multiqubit_circuit);
    PendingFactoredState multiqubit_reference_pending(multiqubit_reference_parsed.state);
    PendingFactoredState multiqubit_optimized_pending(multiqubit_optimized_parsed.state);
    const auto multiqubit_stats = optimize_pending_operations(multiqubit_optimized_pending);
    require(multiqubit_stats.measurement_left_swaps > 0,
            "two-qubit measurement moves before its matching rotation");
    require_pending_record_conditions_are_causal(
        multiqubit_optimized_pending,
        "optimized multi-qubit feedback program");
    const auto multiqubit_reference =
        plan_without_pending_optimization(std::move(multiqubit_reference_pending));
    const auto multiqubit_optimized =
        plan_factored_updates(std::move(multiqubit_optimized_pending));
    require(
        sample_measurements(multiqubit_reference, 1024, 829) ==
            sample_measurements(multiqubit_optimized, 1024, 829),
        "multi-qubit measurement movement preserves feedback records");

    const std::string reset_circuit =
        "H 0\n"
        "R_Z(0.125) 0\n"
        "R 0\n"
        "H 0\n"
        "M 0\n";
    const auto reset_reference_parsed = parse_stim_text(reset_circuit);
    const auto reset_optimized_parsed = parse_stim_text(reset_circuit);
    PendingFactoredState reset_reference_pending(reset_reference_parsed.state);
    PendingFactoredState reset_optimized_pending(reset_optimized_parsed.state);
    const auto reset_stats = optimize_pending_operations(reset_optimized_pending);
    require(reset_stats.measurement_left_swaps > 0,
            "reset test moves its internal commuting measurement earlier");
    require_pending_record_conditions_are_causal(
        reset_optimized_pending,
        "optimized reset program");
    const auto reset_reference =
        plan_without_pending_optimization(std::move(reset_reference_pending));
    const auto reset_optimized = plan_factored_updates(std::move(reset_optimized_pending));
    require(
        sample_measurements(reset_reference, 1024, 831) ==
            sample_measurements(reset_optimized, 1024, 831),
        "early internal reset measurement preserves later output records");

    const std::string sign_guard_circuit =
        "R_Z(0.1) 0\n"
        "H 1\n"
        "M 1\n"
        "CX rec[-1] 0\n"
        "R_Z(0.2) 0\n"
        "M 0\n";
    const auto sign_guard_parsed = parse_stim_text(sign_guard_circuit);
    PendingFactoredState sign_guard_pending(sign_guard_parsed.state);
    const auto sign_guard_records = pending_record_order(sign_guard_pending);
    const auto sign_guard_stats = optimize_pending_operations(sign_guard_pending);
    require(sign_guard_stats.fused_rotations == 0,
            "measurement-conditioned Pauli feedback prevents unsafe rotation fusion");
    require(pending_record_order(sign_guard_pending) == sign_guard_records,
            "feedback sign guard preserves record ordering");
    require_pending_record_conditions_are_causal(
        sign_guard_pending,
        "optimized feedback sign-guard program");

    const std::string detector_boundary_circuit =
        "M 2\n"
        "H 0\n"
        "R_Z(0.125) 0\n"
        "DETECTOR rec[-1]\n"
        "M 0\n";
    const auto detector_program =
        plan_stim_factored_program(parse_stim_text(detector_boundary_circuit));
    std::size_t rotation_index = detector_program.instructions.size();
    std::size_t detector_index = detector_program.instructions.size();
    std::size_t second_record_index = detector_program.instructions.size();
    for (std::size_t index = 0; index < detector_program.instructions.size(); ++index) {
        std::visit(
            [&](const auto& instruction) {
                using T = std::decay_t<decltype(instruction)>;
                if constexpr (std::is_same_v<T, ApplyPrecomputedActivePauliRotation> ||
                              std::is_same_v<T, PromoteDormantRotation>) {
                    rotation_index = std::min(rotation_index, index);
                } else if constexpr (std::is_same_v<T, RecordDetector>) {
                    detector_index = std::min(detector_index, index);
                }
                if constexpr (requires { instruction.record; }) {
                    if (instruction.record && *instruction.record == 2) {
                        second_record_index = std::min(second_record_index, index);
                    }
                }
            },
            detector_program.instructions[index]);
    }
    require(rotation_index < detector_index && detector_index < second_record_index,
            "detector remains between the original rotation prefix and later measurement");
}

void test_high_pivot_rotation_kernels() {
    using namespace symft;
    const int small_k = 6;
    const int large_k = 16;
    const double theta = 0.071;
    const auto uniform_small = pauli_x(small_k, 1) * pauli_x(small_k, small_k - 1);
    const auto real_small = pauli_x(small_k, 1) * pauli_y(small_k, small_k - 1);
    const auto general_small = pauli_x(small_k, 1) * pauli_z(small_k, 3) * pauli_x(small_k, small_k - 1);
    check_high_pivot_single_rotation_kernel(uniform_small, theta, small_k);
    check_high_pivot_single_rotation_kernel(real_small, theta, small_k);
    check_high_pivot_single_rotation_kernel(general_small, theta, small_k);

    const auto uniform_large = pauli_x(large_k, 2) * pauli_x(large_k, large_k - 1);
    const auto real_large = pauli_x(large_k, 2) * pauli_y(large_k, large_k - 1);
    const auto general_large = pauli_x(large_k, 2) * pauli_z(large_k, 5) * pauli_x(large_k, large_k - 1);
    check_high_pivot_single_rotation_kernel(uniform_large, theta, large_k);
    check_high_pivot_single_rotation_kernel(real_large, theta, large_k);
    check_high_pivot_single_rotation_kernel(general_large, theta, large_k);

    for (const std::uint64_t lower_mask : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3}}) {
        check_high_pivot_single_rotation_kernel(uniform_xmask_pauli(8, 7, lower_mask), theta, 8);
        check_high_pivot_single_rotation_kernel(real_xmask_pauli(8, 7, lower_mask), theta, 8);
        check_high_pivot_single_rotation_kernel(general_xmask_pauli(8, 7, lower_mask), theta, 8);
        check_high_pivot_single_rotation_kernel(uniform_xmask_pauli(large_k, large_k - 1, lower_mask), theta, large_k);
        check_high_pivot_single_rotation_kernel(real_xmask_pauli(large_k, large_k - 1, lower_mask), theta, large_k);
        check_high_pivot_single_rotation_kernel(general_xmask_pauli(large_k, large_k - 1, lower_mask), theta, large_k);
    }

    check_high_pivot_batch_rotation_kernel(uniform_large, theta, large_k, false);
    check_high_pivot_batch_rotation_kernel(uniform_large, theta, large_k, true);
    check_high_pivot_batch_rotation_kernel(real_large, theta, large_k, false);
    check_high_pivot_batch_rotation_kernel(real_large, theta, large_k, true);
    check_high_pivot_batch_rotation_kernel(general_large, theta, large_k, false);
    check_high_pivot_batch_rotation_kernel(general_large, theta, large_k, true);
    check_high_pivot_batch_rotation_kernel(uniform_large, theta, large_k, false, 2);
    check_high_pivot_batch_rotation_kernel(uniform_large, theta, large_k, true, 2);
    check_high_pivot_batch_rotation_kernel(uniform_large, theta, large_k, true, 5, true);
    check_high_pivot_batch_rotation_kernel(real_large, theta, large_k, true, 5, true);
    check_high_pivot_batch_rotation_kernel(general_large, theta, large_k, true, 5, true);
}

void test_high_pivot_measurement_kernels() {
    const int k = 8;
    check_diagonal_active_measurement_projection_kernel(symft::pauli_z(k, 2) * symft::pauli_z(k, k - 1), k, 601);
    check_diagonal_active_measurement_projection_kernel(symft::pauli_z(k, 2), k, 602);
    check_batch_active_measurement_projection_kernel(symft::pauli_z(k, 2) * symft::pauli_z(k, k - 1), k, 603);
    check_batch_active_measurement_projection_kernel(symft::pauli_z(k, 2), k, 604);
    check_batch_active_measurement_projection_kernel(symft::pauli_z(k, 2) * symft::pauli_z(k, k - 1), k, 605, 7, true);
    for (const std::uint64_t lower_mask : {std::uint64_t{1}, std::uint64_t{2}, std::uint64_t{3}}) {
        check_active_measurement_projection_kernel(uniform_xmask_pauli(k, k - 1, lower_mask), k, 701 + lower_mask);
        check_active_measurement_projection_kernel(real_xmask_pauli(k, k - 1, lower_mask), k, 801 + lower_mask);
        check_active_measurement_projection_kernel(general_xmask_pauli(k, k - 1, lower_mask), k, 901 + lower_mask);
        check_batch_active_measurement_projection_kernel(uniform_xmask_pauli(k, k - 1, lower_mask), k, 1001 + lower_mask);
        check_batch_active_measurement_projection_kernel(real_xmask_pauli(k, k - 1, lower_mask), k, 1101 + lower_mask);
        check_batch_active_measurement_projection_kernel(general_xmask_pauli(k, k - 1, lower_mask), k, 1201 + lower_mask);
        check_batch_active_measurement_projection_kernel(uniform_xmask_pauli(k, k - 1, lower_mask), k, 1301 + lower_mask, 2);
        check_batch_active_measurement_projection_kernel(real_xmask_pauli(k, k - 1, lower_mask), k, 1401 + lower_mask, 2);
        check_batch_active_measurement_projection_kernel(general_xmask_pauli(k, k - 1, lower_mask), k, 1501 + lower_mask, 2);
        check_batch_active_measurement_projection_kernel(uniform_xmask_pauli(k, k - 1, lower_mask), k, 1601 + lower_mask, 7, true);
        check_batch_active_measurement_projection_kernel(real_xmask_pauli(k, k - 1, lower_mask), k, 1701 + lower_mask, 7, true);
        check_batch_active_measurement_projection_kernel(general_xmask_pauli(k, k - 1, lower_mask), k, 1801 + lower_mask, 7, true);
    }
}

void test_d5_nondiagonal_measurement_simd_kernel() {
    using namespace symft;
    constexpr int k = 10;
    const PrecomputedActivePauliMeasurementKernel kernel(
        pauli_x(k, 0) * pauli_x(k, 1) * pauli_y(k, 9));
    require(!kernel.is_diagonal, "d5 SIMD fixture is nondiagonal");
    require(kernel.action.xmask == 0x203 && kernel.action.zmask == 0x200,
            "d5 SIMD fixture matches cultivation measurement masks");

    const auto alpha = deterministic_alpha(k, 7);
    std::vector<double> re(alpha.size());
    std::vector<double> im(alpha.size());
    for (std::size_t idx = 0; idx < alpha.size(); ++idx) {
        re[idx] = alpha[idx].real();
        im[idx] = alpha[idx].imag();
    }
    std::vector<double> scalar_re(kernel.out_dim);
    std::vector<double> scalar_im(kernel.out_dim);
    std::vector<double> selected_re(kernel.out_dim);
    std::vector<double> selected_im(kernel.out_dim);
    const auto& scalar = simd::scalar_table();
    const auto& selected = simd::dispatch_table();
    for (const bool branch : {false, true}) {
        const double scalar_probability = scalar.measure_nondiagonal_probability_soa(
            re.data(), im.data(), alpha.size(), kernel.action.xmask, kernel.action.zmask,
            static_cast<unsigned>(kernel.pivot), kernel.nondiagonal_coefficient1_even, branch);
        const double selected_probability = selected.measure_nondiagonal_probability_soa(
            re.data(), im.data(), alpha.size(), kernel.action.xmask, kernel.action.zmask,
            static_cast<unsigned>(kernel.pivot), kernel.nondiagonal_coefficient1_even, branch);
        require(
            std::abs(scalar_probability - selected_probability) < 1e-12,
            "d5 nondiagonal SIMD probability matches scalar");
        scalar.project_nondiagonal_soa(
            re.data(), im.data(), scalar_re.data(), scalar_im.data(), alpha.size(),
            kernel.action.xmask, kernel.action.zmask, static_cast<unsigned>(kernel.pivot),
            kernel.nondiagonal_coefficient1_even, branch, 1.25);
        selected.project_nondiagonal_soa(
            re.data(), im.data(), selected_re.data(), selected_im.data(), alpha.size(),
            kernel.action.xmask, kernel.action.zmask, static_cast<unsigned>(kernel.pivot),
            kernel.nondiagonal_coefficient1_even, branch, 1.25);
        for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
            require(
                std::abs(scalar_re[idx] - selected_re[idx]) < 1e-12 &&
                    std::abs(scalar_im[idx] - selected_im[idx]) < 1e-12,
                "d5 nondiagonal SIMD projection matches scalar");
        }
    }
}

void test_dense_backend_reporting() {
    require(
        symft::active_simd_backend() == symft::active_batch_backend(),
        "single-shot and prepared batch report the shared dense-vector backend");
}

void test_active_h_rewrite_stays_virtual() {
    using namespace symft;
    FrameFactoredState state(2, 1);
    apply_pauli_rotation(state, pauli_z(2, 0) * pauli_x(2, 1), M_PI / 2.0);
    apply_pauli_measurement(state, pauli_z(2, 1));
    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 2, "virtual active H rewrite promotes dormant qubit");

    const auto records = sample_measurements(program, 8, 17);
    for (const auto& shot : records) {
        require(packed_bit(shot, 0), "virtual active H rewrite preserves deterministic dormant rotation");
    }
}

void test_dormant_measurement_tableau_reuse() {
    using namespace symft;
    FrameFactoredState state(2, 1);
    const PauliString measured = pauli_z(2, 0) * pauli_x(2, 1);
    apply_pauli_measurement(state, measured);
    apply_pauli_measurement(state, measured);
    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 1, "dormant measurement tableau does not touch alpha");

    const auto records = sample_measurements(program, 64, 123);
    for (const auto& shot : records) {
        require(
            packed_bit(shot, 0) == packed_bit(shot, 1),
            "dormant measurement tableau reuses fixed stabilizer outcome");
    }
}

void test_dormant_measurement_sign_feeds_promotion() {
    using namespace symft;
    FrameFactoredState state(1, 0);
    apply_pauli_measurement(state, pauli_x(1, 0));
    apply_pauli_rotation(state, pauli_z(1, 0), M_PI / 2.0);
    apply_pauli_measurement(state, pauli_x(1, 0));
    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 1, "dormant sign promotion test promotes one active qubit");

    const auto records = sample_measurements(program, 64, 456);
    for (const auto& shot : records) {
        require(
            packed_bit(shot, 0) != packed_bit(shot, 1),
            "dormant measurement sign feeds later promotion");
    }
}

void test_measurement_record_substitution_cancels_reset() {
    using namespace symft;
    FrameFactoredState state(1, 0);
    const SymbolicBool dense_sign(false, {1, 2, 3, 4, 5, 6, 7, 8});
    state.context->bump_next_condition(dense_sign);
    const int record_condition = state.context->fresh_condition();
    apply_pauli_measurement(state, pauli_x(1, 0), dense_sign, 1, record_condition);
    apply_pauli(state, pauli_z(1, 0), xor_bool(symbolic_bool(record_condition), dense_sign));
    apply_pauli_measurement(state, pauli_x(1, 0), SymbolicBool(false), 2, state.context->fresh_condition());

    PendingFactoredState pending(state);
    const auto program = plan_factored_updates(pending);
    require(program.instructions.size() >= 2, "reset cancellation test emits two measurements");

    const auto* measurement = std::get_if<IntroduceDormantMeasurementBranch>(&program.instructions[0]);
    require(measurement != nullptr, "reset cancellation test starts with dormant branch measurement");
    require(
        measurement->outcome.conditions.size() == dense_sign.conditions.size() + 1,
        "recorded measurement outcome still carries the original symbolic sign");

    bool checked_second_measurement = false;
    for (std::size_t i = 1; i < program.instructions.size(); ++i) {
        std::visit(
            [&](const auto& instruction) {
                using T = std::decay_t<decltype(instruction)>;
                if constexpr (std::is_same_v<T, RecordMeasurement>) {
                    require(
                        !instruction.outcome.constant && instruction.outcome.conditions.empty(),
                        "reset substitution cancels the expanded effective branch expression");
                    checked_second_measurement = true;
                }
            },
            program.instructions[i]);
    }
    require(checked_second_measurement, "reset cancellation test checks the second measurement");
}

void test_measurement_relation_reduces_dense_suffix_sign() {
    using namespace symft;
    const SymbolicBool dense_sign(false, {1, 2, 3, 4, 5, 6, 7, 8});
    const int record_condition = 9;
    const int branch = 10;
    const int later_record_condition = 11;
    const SymbolicBool outcome = xor_bool(dense_sign, symbolic_bool(branch));
    FactoredInstructionProgram program(
        1,
        0,
        std::vector<FactoredInstruction>{
            IntroduceDormantMeasurementBranch{
                branch,
                outcome,
                1,
                record_condition,
                SymbolicBoolEvaluationPlan(outcome),
            },
            RecordMeasurement{
                dense_sign,
                2,
                later_record_condition,
                SymbolicBoolEvaluationPlan(dense_sign),
            },
        },
        0,
        SymbolicContext(12));

    const auto& reduced = std::get<RecordMeasurement>(program.instructions[1]).outcome;
    require(
        reduced == xor_bool(symbolic_bool(record_condition), symbolic_bool(branch)),
        "measurement relation reduces dense lambda to m_t xor y_t");
}

void test_measurement_relation_keeps_sparse_record_sign() {
    using namespace symft;
    const SymbolicBool dense_sign(false, {1, 2, 3, 4, 5, 6, 7, 8});
    const int record_condition = 9;
    const int branch = 10;
    const int later_record_condition = 11;
    const SymbolicBool outcome = xor_bool(dense_sign, symbolic_bool(branch));
    const SymbolicBool sparse_record = symbolic_bool(record_condition);
    FactoredInstructionProgram program(
        1,
        0,
        std::vector<FactoredInstruction>{
            IntroduceDormantMeasurementBranch{
                branch,
                outcome,
                1,
                record_condition,
                SymbolicBoolEvaluationPlan(outcome),
            },
            RecordMeasurement{
                sparse_record,
                2,
                later_record_condition,
                SymbolicBoolEvaluationPlan(sparse_record),
            },
        },
        0,
        SymbolicContext(12));

    const auto& reduced = std::get<RecordMeasurement>(program.instructions[1]).outcome;
    require(reduced == sparse_record, "measurement relation keeps already sparse record sign");
}

void test_parser_feedback() {
    using namespace symft;
    const auto parsed = parse_stim_text("M !0\nCX rec[-1] 1\nM 1\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto records = sample_measurements(program);
    require(program.nrecords == 2, "feedback record count");
    require(packed_bit(records, 0) && packed_bit(records, 1), "feedback deterministic records");
}

void test_stim_frontend_circuit_lowering() {
    using namespace symft;
    const auto circuit = parse_stim_circuit_text("REPEAT 2 {\nM !0\n}\nDETECTOR rec[-1] rec[-2]\n");
    require(circuit.nqubits == 1, "circuit qubit count");
    require(circuit.nrecords == 2, "circuit record count");
    require(circuit.instructions.size() == 2, "circuit flattened repeat instructions");
    require(circuit.instructions[0].kind == CircuitInstructionKind::MZ, "circuit measurement kind");
    require(circuit.instructions[0].measurement_targets[0].inverted, "circuit inverted measurement target");
    require(circuit.detectors.size() == 1, "circuit detector count");
    require(circuit.detectors[0].records[0] == 2 && circuit.detectors[0].records[1] == 1, "circuit detector records resolved");
    require(circuit.detectors[0].after_instruction == 2, "circuit detector source position");

    const auto lowered = lower_circuit_to_factored(circuit);
    require(lowered.measurement_records.size() == 2, "lowered record count");
    require(lowered.instruction_pending_operation_counts.size() == 3, "lowered instruction pending-count table");
    require(lowered.instruction_pending_operation_counts[2] == 2, "lowered detector pending position");
    PendingFactoredState pending(lowered.state);
    const auto program = plan_factored_updates(pending);
    require(program.pending_prefix_instruction_indices.size() == 3, "planned pending-prefix checkpoints");
    const auto records = sample_measurements(program);
    require(packed_bit(records, 0) && packed_bit(records, 1), "lowered circuit deterministic records");

    const auto detector_after_clifford = parse_stim_text("M 0\nH 0\nDETECTOR rec[-1]\n");
    require(detector_after_clifford.detectors[0].after_instruction == 2, "detector keeps source order after Clifford");
    require(detector_after_clifford.detectors[0].after_pending_operation == 1, "Clifford does not shift pending detector position");

    const auto cy_feedback = parse_stim_text("M !0\nCY rec[-1] 1\nM 1\n");
    PendingFactoredState pending_cy(cy_feedback.state);
    const auto cy_program = plan_factored_updates(pending_cy);
    const auto cy_records = sample_measurements(cy_program);
    require(packed_bit(cy_records, 0) && packed_bit(cy_records, 1), "CY feedback deterministic records");

    const auto ordinary_cy = parse_stim_text("X 0\nCY 0 1\nM 1\n");
    PendingFactoredState pending_ordinary_cy(ordinary_cy.state);
    const auto ordinary_cy_program = plan_factored_updates(pending_ordinary_cy);
    const auto ordinary_cy_records = sample_measurements(ordinary_cy_program);
    require(packed_bit(ordinary_cy_records, 0), "ordinary CY deterministic target record");
}

void test_extended_stim_frontend() {
    using namespace symft;
    {
        const auto parsed = parse_stim_text("R_X(0.5) 0\n");
        require(parsed.state.pending_operations.size() == 1, "R_X emits one pending rotation");
        const auto& rotation = std::get<PendingPauliRotation>(parsed.state.pending_operations[0]);
        require(std::abs(rotation.kernel_angle - M_PI / 4.0) < 1e-12, "R_X half-turn angle conversion");
        require(rotation.pauli.pauli.same_body(pauli_x(1, 0)), "R_X rotation body");
    }
    {
        const auto parsed = parse_stim_text("R_Z(pi/pi) 0\n");
        const auto& rotation = std::get<PendingPauliRotation>(parsed.state.pending_operations[0]);
        require(std::abs(rotation.kernel_angle - M_PI / 2.0) < 1e-12, "rotation expression is converted from half-turns");
    }
    {
        const auto circuit = parse_stim_circuit_text(
            "R_XX(0.25) 0 1\n"
            "R_PAULI(-0.5) X0*Z1\n"
            "U3(0.5,0.25,-0.5) 0\n");
        require(circuit.instructions.size() == 5, "Pauli and U3 rotations emit expected instructions");
        require(
            std::abs(circuit.instructions[0].kernel_angle - M_PI / 8.0) < 1e-12,
            "two-qubit rotation uses half-turns");
        require(
            std::abs(circuit.instructions[1].kernel_angle + M_PI / 4.0) < 1e-12,
            "general Pauli rotation uses half-turns");
        require(
            std::abs(circuit.instructions[2].kernel_angle + M_PI / 4.0) < 1e-12 &&
                std::abs(circuit.instructions[3].kernel_angle - M_PI / 4.0) < 1e-12 &&
                std::abs(circuit.instructions[4].kernel_angle - M_PI / 8.0) < 1e-12,
            "U3 rotations use half-turns");
    }
    {
        const auto circuit = parse_stim_circuit_text(
            "MXX !0 1\n"
            "MPAD 1 0\n"
            "OBSERVABLE_INCLUDE(0) rec[-1] X0\n");
        require(circuit.nrecords == 3, "pair measurement and MPAD record count");
        require(circuit.instructions.size() == 2, "observable Pauli target ignored as an operation");
        require(circuit.observables.size() == 1, "observable parsed");
        require(circuit.observables[0].records.size() == 1, "observable ignores Pauli targets for circuit sampling");
        require(circuit.observables[0].records[0] == 3, "observable record index after MPAD");
    }
    {
        const auto circuit = parse_stim_circuit_text("MPAD 1\n");
        require(circuit.nqubits == 0, "MPAD targets do not imply qubits");
        require(circuit.nrecords == 1, "MPAD still appends a measurement record");
    }
    {
        require_throws(
            [] { (void)parse_stim_text("X 0\nCX sweep[x] 0\n"); },
            "invalid sweep target is rejected");
        require_throws_with_message(
            [] { (void)parse_stim_text("X 0\nCX sweep[0] 0\n"); },
            "sweep-controlled operations are not supported",
            "sweep-controlled CX is rejected");
        require_throws_with_message(
            [] { (void)parse_stim_text("X 0\nCZ 0 sweep[0]\n"); },
            "sweep-controlled operations are not supported",
            "reverse sweep-controlled CZ is rejected");
    }
    {
        require_throws(
            [] { (void)parse_stim_circuit_text("REPEAT 0 {\nM 0\n}\n"); },
            "zero-count REPEAT is rejected");
        require_throws(
            [] { (void)parse_stim_circuit_text("REPEAT(2) 1 {\nM 0\n}\n"); },
            "REPEAT rejects parens arguments");
        require_throws(
            [] { (void)parse_stim_circuit_text("M 0\nOBSERVABLE_INCLUDE(0.6) rec[-1]\n"); },
            "OBSERVABLE_INCLUDE rejects non-integer indices");
        require_throws(
            [] { (void)parse_stim_circuit_text("R(0.25) 0\n"); },
            "reset rejects parens arguments");
        require_throws(
            [] { (void)parse_stim_circuit_text("I_ERROR(0.8,0.8) 0\n"); },
            "I_ERROR rejects probability sums above one");
        require_throws(
            [] { (void)parse_stim_circuit_text("II_ERROR(0.8,0.8) 0 1\n"); },
            "II_ERROR rejects probability sums above one");
        require_throws(
            [] { (void)parse_stim_circuit_text("TICK 0\n"); },
            "TICK rejects targets");
        require_throws(
            [] { (void)parse_stim_circuit_text("TICK()\n"); },
            "TICK rejects empty parens");
        require_throws(
            [] { (void)parse_stim_circuit_text("SHIFT_COORDS(1) 0\n"); },
            "SHIFT_COORDS rejects targets");
        require_throws(
            [] { (void)parse_stim_circuit_text("SHIFT_COORDS\n"); },
            "SHIFT_COORDS requires offsets");
        require_throws(
            [] { (void)parse_stim_circuit_text("QUBIT_COORDS foo\n"); },
            "QUBIT_COORDS rejects non-qubit targets");
        require_throws(
            [] { (void)parse_stim_circuit_text("QUBIT_COORDS() 0\n"); },
            "QUBIT_COORDS rejects empty parens");
    }
    {
        const auto circuit = parse_stim_circuit_text(
            "E(0.25) X0\n"
            "ELSE_CORRELATED_ERROR(0.5) Z0\n"
            "M 0\n");
        require(circuit.instructions.size() == 2, "correlated error chain flushes before measurement");
        require(circuit.instructions[0].kind == CircuitInstructionKind::PauliProductChannel, "correlated error channel kind");
        require(circuit.instructions[0].probabilities.size() == 2, "correlated error channel alternatives");
        require(std::abs(circuit.instructions[0].probabilities[0] - 0.25) < 1e-12, "first correlated error probability");
        require(std::abs(circuit.instructions[0].probabilities[1] - 0.375) < 1e-12, "else correlated error probability");
    }
    {
        const auto parsed = parse_stim_text(
            "H_XY 0\n"
            "ZCX 0 1\n"
            "SWAPCZ 0 1\n"
            "SQRT_YY_DAG 0 1\n"
            "R_YY(0.25) 0 1\n"
            "R_PAULI(1) X0*Z1\n"
            "M 0 1\n");
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        const auto records = sample_measurements(program, 4, 11);
        require(records.size() == 4, "extended Stim and rotation gates sample");
    }
    {
        const auto parsed = parse_stim_text(
            "SPP X0*Z1\n"
            "PAULI_CHANNEL_1(0.1, 0.2, 0.3) 0\n"
            "PAULI_CHANNEL_2(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1) 0 1\n"
            "DEPOLARIZE3(0.1) 0 1 2\n"
            "HERALDED_ERASE(0.1) 0\n"
            "HERALDED_PAULI_CHANNEL_1(0, 0.1, 0, 0) 1\n"
            "M 0 1\n");
        PendingFactoredState pending(parsed.state);
        const auto program = plan_factored_updates(pending);
        require(program.nrecords == 4, "heralded channels add measurement records");
        const auto records = sample_measurements(program, 3, 13);
        require(records.size() == 3, "extended noise gates sample");
    }
}

void test_t_gate_exact_rotation() {
    using namespace symft;
    const auto parsed = parse_stim_text("H 0\nT 0\nH 0\nM 0\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    require(program.max_k == 1, "T promotes one active qubit");
    int ones = 0;
    const auto records = sample_measurements(program, 200, 7);
    for (const auto& shot : records) {
        ones += packed_bit(shot, 0) ? 1 : 0;
    }
    require(ones > 10 && ones < 50, "T then MX produces the expected non-deterministic measurement");
}

void test_presampled_exogenous() {
    using namespace symft;
    const auto parsed = parse_stim_text("X_ERROR(1) 0\nM 0\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto samples = presample_exogenous(program, 4, 17);
    require(samples.nshots == 4, "presampled shot count");
    require(samples.nsymbols == program.nsymbols, "presampled symbol count");

    FactoredExecutorState runtime(program, samples.next_rng_state);
    for (int shot = 0; shot < samples.nshots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program, samples, shot);
        require(packed_bit(runtime.measurement_words, 0), "presampled deterministic X error");
    }

    const auto packed_samples = presample_exogenous_packed(program, 4, 17);
    require(packed_samples.nshots == 4, "packed presampled shot count");
    require(packed_samples.nsymbols == program.nsymbols, "packed presampled symbol count");
    PresampledExpressionPlan packed_expression_plan;
    prepare_presampled_expression_plan(packed_expression_plan, program, packed_samples);
    PresampledExpressionBlock packed_expression_block;
    evaluate_presampled_expression_block(
        packed_expression_block,
        packed_expression_plan,
        packed_samples);
    for (int shot = 0; shot < packed_samples.nshots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program, packed_expression_plan, packed_expression_block, shot);
        require(packed_bit(runtime.measurement_words, 0), "packed presampled expression deterministic X error");
    }

    require(default_single_shot_sample_chunk_shots() == 2048, "single-shot default sample chunk");
    require(default_batch_count(10) == 32, "batch default keeps active footprint cap");
    const auto chunked_records = sample_measurements(program, 9, 17, 3);
    require(chunked_records.size() == 9, "chunked single-shot sample count");
    for (const auto& shot : chunked_records) {
        require(packed_bit(shot, 0), "chunked single-shot deterministic X error");
    }

    const auto random_error = parse_stim_text("X_ERROR(0.5) 0\nM 0\n");
    PendingFactoredState pending_random(random_error.state);
    const auto random_program = plan_factored_updates(pending_random);
    const auto packed_random_samples = presample_exogenous_packed(random_program, 11, 31);
    const int random_condition = random_program.sampled_bernoulli_conditions.front();
    const auto random_row =
        packed_random_samples.value_words[(static_cast<std::size_t>(random_condition - 1) * packed_random_samples.shot_words)];
    require(random_row != 0 && random_row != ((std::uint64_t{1} << 11) - 1), "packed fair Bernoulli bits are non-degenerate");
    BatchFactoredExecutorState packed_runtime(random_program, 5, 41);
    execute_batch_in_place(packed_runtime, random_program, packed_random_samples, 3);
    PresampledExpressionPlan random_expression_plan;
    prepare_presampled_expression_plan(random_expression_plan, random_program, packed_random_samples);
    PresampledExpressionBlock random_expression_block;
    evaluate_presampled_expression_block(
        random_expression_block,
        random_expression_plan,
        packed_random_samples);
    BatchFactoredExecutorState expression_runtime(random_program, 5, 41);
    execute_batch_in_place(
        expression_runtime,
        random_program,
        random_expression_plan,
        random_expression_block,
        3);
    require(
        expression_runtime.measurement_words == packed_runtime.measurement_words,
        "batch presampled expression slice matches packed exogenous execution");

    const auto mid_error = parse_stim_text("X_ERROR(0.3) 0\nM 0\n");
    PendingFactoredState pending_mid(mid_error.state);
    const auto mid_program = plan_factored_updates(pending_mid);
    const int mid_shots = 4096;
    const auto mid_samples = presample_exogenous_packed(mid_program, mid_shots, 1234);
    const int mid_condition = mid_program.sampled_bernoulli_conditions.front();
    const std::size_t mid_base = static_cast<std::size_t>(mid_condition - 1) * mid_samples.shot_words;
    int mid_ones = 0;
    for (int shot = 0; shot < mid_shots; ++shot) {
        const std::size_t word = static_cast<std::size_t>(shot >> 6);
        if ((mid_samples.value_words[mid_base + word] & (std::uint64_t{1} << (shot & 63))) != 0) {
            ++mid_ones;
        }
    }
    require(mid_ones > 1000 && mid_ones < 1450, "packed mid-probability Bernoulli has plausible weight");
}

void test_active_component_factorization() {
    using namespace symft;

    auto base = component_test_program();
    require(base.active_component_plan != nullptr, "component plan is available");
    require(
        !base.use_active_components,
        "small component program conservatively keeps the dense default");
    require(
        base.active_component_plan->measurements.size() == 3,
        "component plan records active measurements");
    require(
        base.active_component_plan->instruction_steps.front().kind ==
            ActiveComponentStepKind::IgnoredGlobalPhase,
        "component plan safely discards an unobservable k=0 global phase");
    const auto& reordered_measurement =
        base.active_component_plan->measurements.front().kernel;
    require(
        reordered_measurement.pivot == 0 &&
            detail::highest_set_bit64(
                reordered_measurement.action.xmask) == 2,
        "component measurement preserves a non-highest remapped pivot");

    auto dense = base;
    dense.use_active_components = false;
    auto component = base;
    component.use_active_components = true;

    const auto dense_single = sample_measurements(dense, 257, 0x314159, 31);
    const auto component_single =
        sample_measurements(component, 257, 0x314159, 31);
    require(
        dense_single == component_single,
        "single-shot component sampling matches dense sampling");

    const auto dense_batch =
        sample_measurements_batch(dense, 257, 13, 0x271828);
    const auto component_batch =
        sample_measurements_batch(component, 257, 13, 0x271828);
    require(
        dense_batch == component_batch,
        "basis-major batch component sampling matches dense sampling");

    const int shots = 17;
    const auto samples =
        presample_exogenous_packed(dense, shots, 0x123456);
    PresampledExpressionPlan expression_plan;
    prepare_presampled_expression_plan(
        expression_plan,
        dense,
        samples);
    PresampledExpressionBlock expression_block;
    evaluate_presampled_expression_block(
        expression_block,
        expression_plan,
        samples);

    BatchFactoredExecutorState dense_runtime(dense, shots, 0xabcdef);
    dense_runtime.dense_shot_major_active = true;
    reset_batch_executor(dense_runtime, dense, shots);
    execute_batch_in_place(
        dense_runtime,
        dense,
        expression_plan,
        expression_block);

    BatchFactoredExecutorState component_runtime(
        component,
        shots,
        0xabcdef);
    require(
        component_runtime.active_components_enabled,
        "forced component batch uses component storage");
    component_runtime.dense_shot_major_active = true;
    reset_batch_executor(component_runtime, component, shots);
    execute_batch_in_place(
        component_runtime,
        component,
        expression_plan,
        expression_block);
    require(
        dense_runtime.measurement_words ==
            component_runtime.measurement_words,
        "shot-major presampled component sampling matches dense sampling");

    BatchFactoredExecutorState dense_postselected(
        dense,
        shots,
        0xfeedface);
    dense_postselected.dense_shot_major_active = true;
    dense_postselected.store_detector_records = false;
    reset_batch_executor(dense_postselected, dense, shots);
    BatchDetectorPostselectionScratch dense_postselection_scratch;
    const auto dense_postselection =
        execute_batch_postselected_in_place(
            dense_postselected,
            dense,
            expression_plan,
            expression_block,
            0,
            dense_postselection_scratch);

    BatchFactoredExecutorState component_postselected(
        component,
        shots,
        0xfeedface);
    require(
        component_postselected.active_components_enabled,
        "component postselection starts with component storage");
    component_postselected.dense_shot_major_active = true;
    component_postselected.store_detector_records = false;
    reset_batch_executor(
        component_postselected,
        component,
        shots);
    BatchDetectorPostselectionScratch component_postselection_scratch;
    const auto component_postselection =
        execute_batch_postselected_in_place(
            component_postselected,
            component,
            expression_plan,
            expression_block,
            0,
            component_postselection_scratch);
    require(
        dense_postselection.discarded ==
                component_postselection.discarded &&
            dense_postselection.accepted ==
                component_postselection.accepted &&
            dense_postselected.active_shots ==
                component_postselected.active_shots,
        "component batch postselection keeps the same survivors");
    for (int record = 1; record <= dense.nrecords; ++record) {
        for (int shot = 0;
             shot < dense_postselected.active_shots;
             ++shot) {
            const std::size_t word = batch_shot_word(shot);
            const bool dense_bit =
                (dense_postselected.measurement_words[
                     batch_record_offset(
                         dense_postselected,
                         record,
                         word)] &
                 batch_shot_mask(shot)) != 0;
            const bool component_bit =
                (component_postselected.measurement_words[
                     batch_record_offset(
                         component_postselected,
                         record,
                         word)] &
                 batch_shot_mask(shot)) != 0;
            require(
                dense_bit == component_bit,
                "component batch postselection preserves compacted records");
        }
    }

    std::vector<FactoredInstruction> profitable_instructions;
    for (int q = 0; q < 12; ++q) {
        profitable_instructions.emplace_back(
            component_test_promotion(0.03 * static_cast<double>(q + 1)));
    }
    for (int repeat = 0; repeat < 5; ++repeat) {
        for (int q = 0; q < 12; ++q) {
            PauliString local_z(12);
            local_z.set_zbit(q);
            profitable_instructions.emplace_back(
                component_test_rotation(local_z, 0.07));
        }
    }
    const FactoredInstructionProgram profitable(
        12,
        0,
        std::move(profitable_instructions),
        12);
    require(
        profitable.use_active_components,
        "automatic selector enables a clearly profitable component program");
    require(
        profitable.active_component_plan->component_allocated_dimension <
            profitable.active_component_plan->dense_peak_dimension,
        "selected component program reduces allocated active dimension");
}

void test_batch_sampler() {
    using namespace symft;
    const auto parsed = parse_stim_text("X_ERROR(1) 0\nM 0\n");
    PendingFactoredState pending(parsed.state);
    const auto program = plan_factored_updates(pending);
    const auto records = sample_measurements_batch(program, 9, 4, 17);
    require(records.size() == 9, "batch sampler shot count");
    for (const auto& shot : records) {
        require(packed_bit(shot, 0), "batch presampled deterministic X error");
    }

    const auto feedback = parse_stim_text("M !0\nCX rec[-1] 1\nM 1\n");
    PendingFactoredState pending_feedback(feedback.state);
    const auto feedback_program = plan_factored_updates(pending_feedback);
    const auto feedback_records = sample_measurements_batch(feedback_program, 7, 3, 19);
    for (const auto& shot : feedback_records) {
        require(packed_bit(shot, 0) && packed_bit(shot, 1), "batch feedback deterministic records");
    }

    const auto t_circuit = parse_stim_text("H 0\nT 0\nM 0\n");
    PendingFactoredState pending_t(t_circuit.state);
    const auto t_program = plan_factored_updates(pending_t);
    const auto t_records = sample_measurements_batch(t_program, 200, 32, 23);
    int ones = 0;
    for (const auto& shot : t_records) {
        ones += packed_bit(shot, 0) ? 1 : 0;
    }
    require(ones > 50 && ones < 150, "batch T then MX produces non-deterministic X measurement");
}

void test_batch_postselection() {
    using namespace symft;
    {
        const auto parsed = parse_stim_text("M !0\nDETECTOR rec[-1]\n");
        const auto program = planned_stim_program(parsed);
        const auto samples = presample_exogenous_packed(program, 8, 17);
        BatchFactoredExecutorState runtime(program, 8, 19);
        BatchDetectorPostselectionScratch scratch;
        const auto result = execute_expression_postselected_for_test(
            runtime,
            program,
            samples,
            scratch);
        require(result.discarded == 8, "batch postselection rejects fired detector");
        require(result.accepted == 0, "batch postselection no accepted shots after rejection");
        require(runtime.active_shots == 0, "batch postselection compacts all rejected shots");
    }
    {
        const auto parsed = parse_stim_text("M 0\nDETECTOR rec[-1]\n");
        const auto program = planned_stim_program(parsed);
        const auto samples = presample_exogenous_packed(program, 8, 23);
        BatchFactoredExecutorState runtime(program, 8, 29);
        BatchDetectorPostselectionScratch scratch;
        const auto result = execute_expression_postselected_for_test(
            runtime,
            program,
            samples,
            scratch);
        require(result.discarded == 0, "batch postselection keeps quiet detector");
        require(result.accepted == 8, "batch postselection accepted count");
        require(runtime.active_shots == 8, "batch postselection keeps active shots");
        require(runtime.measurement_words[0] == 0, "batch postselection keeps quiet measurement records");
    }
    {
        const auto parsed = parse_stim_text(
            "X_ERROR(0.125) 0\n"
            "M 0\n"
            "DETECTOR rec[-1]\n"
            "H 1\n"
            "T 1\n"
            "T_DAG 1\n"
            "H 1\n"
            "M 1\n");
        const auto program = planned_stim_program(parsed);
        const auto samples = presample_exogenous_packed(program, 64, 41);
        BatchDetectorPostselectionScratch default_scratch;
        BatchFactoredExecutorState default_runtime(program, 64, 43);
        const auto default_result = execute_expression_postselected_for_test(
            default_runtime,
            program,
            samples,
            default_scratch);
        BatchDetectorPostselectionScratch alternate_scratch;
        BatchFactoredExecutorState alternate_runtime(program, 64, 43);
        const auto alternate_result = execute_expression_postselected_for_test(
            alternate_runtime,
            program,
            samples,
            alternate_scratch,
            BatchDetectorPostselectionOptions{1});
        require(default_result.discarded > 0, "mask-threshold postselection test kills at least one lane");
        require(default_runtime.active_shots == default_result.accepted, "default mask-threshold compacts dead lanes");
        require(default_result.discarded == alternate_result.discarded, "mask-threshold postselection discarded count");
        require(default_result.accepted == alternate_result.accepted, "mask-threshold postselection accepted count");
        require(
            default_runtime.measurement_words == alternate_runtime.measurement_words,
            "mask-threshold postselection records");
    }
}

void test_detectors() {
    using namespace symft;
    const auto accepted = parse_stim_text("M !0\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    const auto summary = estimate_stim_logical_error_rate(accepted, 5);
    require(summary.shots == 5, "summary shot count");
    require(summary.discarded == 0, "summary no discard");
    require(summary.logical_errors == 5, "summary logical count");

    const auto rejected = parse_stim_text("M !0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n");
    const auto summary2 = estimate_stim_logical_error_rate(rejected, 5);
    require(summary2.discarded == 5, "summary discard count");
    require(summary2.accepted == 0, "summary accepted count");
}

void test_prepared_circuit_sampler_reuse() {
    using namespace symft;
    const std::string path =
        (std::filesystem::temp_directory_path() / "symft_prepared_sampler_test.stim").string();
    {
        std::ofstream out(path);
        out << "M 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n";
    }

    CircuitSamplingOptions options;
    options.batch_size = 16;
    options.sample_chunk_shots = 64;
    options.threads = 4;

    auto batch = prepare_batch_sampler_from_stim_file(path, options);
    const auto large = batch.sample(257);
    require(large.counts.shots == 257, "prepared batch large shot count");
    require(large.counts.accepted == 257, "prepared batch large accepted count");

    const auto small = batch.sample(1);
    require(small.counts.shots == 1, "prepared batch inactive workers do not leak shots");
    require(small.counts.accepted == 1, "prepared batch inactive workers do not leak accepted count");
    require(small.counts.discarded == 0, "prepared batch inactive workers do not leak discarded count");

    auto single = prepare_single_shot_sampler_from_stim_file(path, options);
    require(single.info().threads == 4, "prepared single configured thread count");
    const auto single_large = single.sample(257);
    require(single_large.active_threads == 4, "prepared single activates chunk workers");
    require(single_large.counts.shots == 257, "prepared single large shot count");
    require(single_large.counts.accepted == 257, "prepared single large accepted count");
    const auto single_a = single.sample(3);
    const auto single_b = single.sample(4);
    require(single_a.active_threads == 1, "prepared single limits workers to available chunks");
    require(single_a.counts.shots == 3, "prepared single first shot count");
    require(single_b.counts.shots == 4, "prepared single second shot count");

    std::remove(path.c_str());
}

void test_prepared_sampler_multithreading() {
    using namespace symft;

    const std::string random_path =
        (std::filesystem::temp_directory_path() / "symft_threaded_single_sampler_test.stim").string();
    {
        std::ofstream out(random_path);
        out << "X_ERROR(0.25) 0\n"
            << "M 0\n"
            << "DETECTOR rec[-1]\n"
            << "OBSERVABLE_INCLUDE(0) rec[-1]\n";
    }

    CircuitSamplingOptions serial_options;
    serial_options.sample_chunk_shots = 64;
    serial_options.batch_size = 16;
    CircuitSamplingOptions parallel_options = serial_options;
    parallel_options.threads = 4;

    auto serial = prepare_single_shot_sampler_from_stim_file(random_path, serial_options);
    auto parallel = prepare_single_shot_sampler_from_stim_file(random_path, parallel_options);
    const auto serial_result = serial.sample(1025, 73);
    const auto parallel_result = parallel.sample(1025, 73);
    require(parallel_result.active_threads == 4, "threaded single random test activates workers");
    require_same_counts(serial_result.counts, parallel_result.counts, "threaded single preserves");

    auto serial_batch = prepare_batch_sampler_from_stim_file(random_path, serial_options);
    auto parallel_batch = prepare_batch_sampler_from_stim_file(random_path, parallel_options);
    const auto serial_batch_result = serial_batch.sample(1025, 117);
    const auto parallel_batch_result = parallel_batch.sample(1025, 117);
    require(parallel_batch_result.active_threads == 4, "threaded batch random test activates workers");
    require_same_counts(
        serial_batch_result.counts,
        parallel_batch_result.counts,
        "threaded batch preserves");

    serial_options.postselect_detectors = true;
    parallel_options.postselect_detectors = true;
    auto serial_postselected = prepare_single_shot_sampler_from_stim_file(random_path, serial_options);
    auto parallel_postselected = prepare_single_shot_sampler_from_stim_file(random_path, parallel_options);
    const auto serial_postselected_result = serial_postselected.sample(1025, 91);
    const auto parallel_postselected_result = parallel_postselected.sample(1025, 91);
    require_same_counts(
        serial_postselected_result.counts,
        parallel_postselected_result.counts,
        "threaded postselected single preserves");

    auto serial_postselected_batch = prepare_batch_sampler_from_stim_file(random_path, serial_options);
    auto parallel_postselected_batch = prepare_batch_sampler_from_stim_file(random_path, parallel_options);
    const auto serial_postselected_batch_result = serial_postselected_batch.sample(1025, 101);
    const auto parallel_postselected_batch_result = parallel_postselected_batch.sample(1025, 101);
    require_same_counts(
        serial_postselected_batch_result.counts,
        parallel_postselected_batch_result.counts,
        "threaded postselected batch preserves");

    auto reference_single = prepare_single_shot_sampler_from_stim_file(random_path, parallel_options);
    auto move_constructed_single = [&] {
        auto source = prepare_single_shot_sampler_from_stim_file(random_path, parallel_options);
        return PreparedCircuitSingleShotSampler(std::move(source));
    }();
    const auto reference_single_result = reference_single.sample(1025, 109);
    const auto move_constructed_single_result = move_constructed_single.sample(1025, 109);
    require_same_counts(
        reference_single_result.counts,
        move_constructed_single_result.counts,
        "move-constructed postselected single preserves");

    auto reference_batch = prepare_batch_sampler_from_stim_file(random_path, parallel_options);
    auto move_constructed_batch = [&] {
        auto source = prepare_batch_sampler_from_stim_file(random_path, parallel_options);
        return PreparedCircuitBatchSampler(std::move(source));
    }();
    auto move_assigned_batch = prepare_batch_sampler_from_stim_file(random_path, parallel_options);
    {
        auto source = prepare_batch_sampler_from_stim_file(random_path, parallel_options);
        move_assigned_batch = std::move(source);
    }
    const auto reference_batch_result = reference_batch.sample(1025, 131);
    const auto move_constructed_batch_result = move_constructed_batch.sample(1025, 131);
    const auto move_assigned_batch_result = move_assigned_batch.sample(1025, 131);
    require_same_counts(
        reference_batch_result.counts,
        move_constructed_batch_result.counts,
        "move-constructed postselected batch preserves");
    require_same_counts(
        reference_batch_result.counts,
        move_assigned_batch_result.counts,
        "move-assigned postselected batch preserves");

    std::remove(random_path.c_str());
}

} // namespace

int main() {
    test_pauli_algebra();
    test_clifford_frame();
    test_extended_clifford_frame_images();
    test_active_rotation();
    test_high_pivot_selection();
    test_pending_operation_optimizer();
    test_pending_operation_optimizer_end_to_end();
    test_high_pivot_rotation_kernels();
    test_high_pivot_measurement_kernels();
    test_d5_nondiagonal_measurement_simd_kernel();
    test_dense_backend_reporting();
    test_active_h_rewrite_stays_virtual();
    test_dormant_measurement_tableau_reuse();
    test_dormant_measurement_sign_feeds_promotion();
    test_measurement_record_substitution_cancels_reset();
    test_measurement_relation_reduces_dense_suffix_sign();
    test_measurement_relation_keeps_sparse_record_sign();
    test_parser_feedback();
    test_stim_frontend_circuit_lowering();
    test_extended_stim_frontend();
    test_t_gate_exact_rotation();
    test_presampled_exogenous();
    test_active_component_factorization();
    test_batch_sampler();
    test_batch_postselection();
    test_detectors();
    test_prepared_circuit_sampler_reuse();
    test_prepared_sampler_multithreading();
    std::cout << "symft_cpp_tests passed (SIMD backend: " << symft::active_simd_backend() << ")\n";
    return 0;
}
