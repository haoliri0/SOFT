#include "batch_internal.hpp"
#include "sampler/active_kernels.hpp"

namespace symft {

const std::vector<double>& fill_shot_coefficient_scalars(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& sign_bits,
    double minus_coeff,
    double plus_coeff) {
    const int lanes = runtime.active_pitch;
    if (runtime.shot_coefficient_scalars.size() < static_cast<std::size_t>(lanes)) {
        runtime.shot_coefficient_scalars.resize(static_cast<std::size_t>(lanes), 0.0);
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t bits = word < sign_bits.size() ? sign_bits[word] : 0;
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        SYMFT_BATCH_SIMD_LOOP
        for (int bit = 0; bit < live; ++bit) {
            runtime.shot_coefficient_scalars[static_cast<std::size_t>(base_shot + bit)] =
                ((bits >> bit) & 1ULL) != 0 ? plus_coeff : minus_coeff;
        }
    }
    return runtime.shot_coefficient_scalars;
}

BatchSignMode batch_sign_mode(const BatchFactoredExecutorState& runtime, const std::vector<std::uint64_t>& sign_bits) {
    bool saw_zero = false;
    bool saw_one = false;
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = batch_live_word_mask(runtime, word);
        const std::uint64_t bits = (word < sign_bits.size() ? sign_bits[word] : 0) & live;
        saw_one = saw_one || bits != 0;
        saw_zero = saw_zero || bits != live;
        if (saw_one && saw_zero) {
            return BatchSignMode::Mixed;
        }
    }
    return saw_one ? BatchSignMode::AllPlus : BatchSignMode::AllMinus;
}

bool batch_bit_at(const std::vector<std::uint64_t>& bits, int shot) {
    const std::size_t word = batch_shot_word(shot);
    return word < bits.size() && ((bits[word] & batch_shot_mask(shot)) != 0);
}

void finish_active_measurement_branch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const std::vector<std::uint64_t>& branch_bits) {
    --runtime.k;
    ++runtime.ndormant;
    assign_batch_symbol(runtime, branch_condition, branch_bits);
}

void copy_projected_active_prefix_from_scratch(BatchFactoredExecutorState& runtime, std::size_t out_dim) {
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    const std::size_t live = static_cast<std::size_t>(runtime.active_shots);
    for (std::size_t basis = 0; basis < out_dim; ++basis) {
        const std::size_t base = basis * pitch;
        std::copy_n(runtime.scratch_re.data() + base, live, runtime.active_re.data() + base);
        std::copy_n(runtime.scratch_im.data() + base, live, runtime.active_im.data() + base);
    }
}

void measure_nondiagonal_true_prob_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    std::fill_n(runtime.branch_prob_true.data(), runtime.active_shots, 0.0);
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        const std::size_t source0 = detail::compact_nondiagonal_measurement_source0(kernel, idx);
        const std::size_t source1 = detail::compact_nondiagonal_measurement_source1(kernel, idx);
        const Complex coefficient1 =
            detail::compact_nondiagonal_measurement_coefficient1(kernel, idx, true);
        const std::size_t base0 = source0 * pitch;
        const std::size_t base1 = source1 * pitch;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            const double ar = inv_sqrt2 * runtime.active_re[base0 + lane] +
                              coefficient1.real() * runtime.active_re[base1 + lane] -
                              coefficient1.imag() * runtime.active_im[base1 + lane];
            const double ai = inv_sqrt2 * runtime.active_im[base0 + lane] +
                              coefficient1.real() * runtime.active_im[base1 + lane] +
                              coefficient1.imag() * runtime.active_re[base1 + lane];
            runtime.branch_prob_true[lane] += ar * ar + ai * ai;
        }
    }
}

void project_nondiagonal_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    const std::vector<std::uint64_t>& branch_bits) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    const auto& branch_directions = fill_shot_coefficient_scalars(runtime, branch_bits, 1.0, -1.0);
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        const std::size_t source0 = detail::compact_nondiagonal_measurement_source0(kernel, idx);
        const std::size_t source1 = detail::compact_nondiagonal_measurement_source1(kernel, idx);
        const Complex false_coefficient1 =
            detail::compact_nondiagonal_measurement_coefficient1(kernel, idx, false);
        const Complex true_coefficient1 = -false_coefficient1;
        const std::size_t base0 = source0 * pitch;
        const std::size_t base1 = source1 * pitch;
        const std::size_t out_base = idx * pitch;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            const Complex coefficient1 =
                branch_directions[lane] < 0.0 ? true_coefficient1 : false_coefficient1;
            const double ar = inv_sqrt2 * runtime.active_re[base0 + lane] +
                              coefficient1.real() * runtime.active_re[base1 + lane] -
                              coefficient1.imag() * runtime.active_im[base1 + lane];
            const double ai = inv_sqrt2 * runtime.active_im[base0 + lane] +
                              coefficient1.real() * runtime.active_im[base1 + lane] +
                              coefficient1.imag() * runtime.active_re[base1 + lane];
            runtime.scratch_re[out_base + lane] = ar * runtime.branch_invnorms[lane];
            runtime.scratch_im[out_base + lane] = ai * runtime.branch_invnorms[lane];
        }
    }
    copy_projected_active_prefix_from_scratch(runtime, kernel.out_dim);
}

void rotate_uniform_imag_pairs_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const BatchSignMode mode = batch_sign_mode(runtime, sign_bits);
    const double minus_q = detail::compact_rotation_coefficient(kernel, 0, false).imag();
    const double plus_q = detail::compact_rotation_coefficient(kernel, 0, true).imag();
    if (mode != BatchSignMode::Mixed) {
        const double q = mode == BatchSignMode::AllPlus ? plus_q : minus_q;
        batch_interleaved::table().rotate_uniform_imag_xmask_const(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.cos_kernel_angle,
            q);
        return;
    }
    const auto& coeffs = fill_shot_coefficient_scalars(
        runtime,
        sign_bits,
        minus_q,
        plus_q);
    if (kernel.pair_count < kXmaskRotationPairThreshold && runtime.active_pitch != 2) {
            batch_interleaved::table().rotate_uniform_imag_pairs(
            runtime.active_re.data(),
            runtime.active_im.data(),
            static_cast<std::size_t>(runtime.active_pitch),
            runtime.active_shots,
            kernel.action.xmask,
            kernel.pair_bit,
            kernel.pair_count,
            kernel.cos_kernel_angle,
            coeffs.data());
        return;
    }
    batch_interleaved::table().rotate_uniform_imag_xmask(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        kernel.action.xmask,
        kernel.pair_bit,
        kernel.pair_count,
        kernel.cos_kernel_angle,
        coeffs.data());
}

void rotate_compact_pairs_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    const double c = kernel.cos_kernel_angle;
    const auto& directions = fill_shot_coefficient_scalars(runtime, sign_bits, 1.0, -1.0);
    for (std::size_t idx = 0; idx < kernel.pair_count; ++idx) {
        const std::size_t left = detail::insert_zero_bit(idx, static_cast<int>(kernel.pair_bit));
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.action.xmask);
        const bool left_odd = detail::active_action_phase_odd(kernel.action, left);
        const double left_parity = left_odd ? -1.0 : 1.0;
        const double right_parity = kernel.action.xz_overlap_odd ? -left_parity : left_parity;
        const double left_minus_re = left_parity * kernel.minus_even_coefficient.real();
        const double left_minus_im = left_parity * kernel.minus_even_coefficient.imag();
        const double right_minus_re = right_parity * kernel.minus_even_coefficient.real();
        const double right_minus_im = right_parity * kernel.minus_even_coefficient.imag();
        const std::size_t left_base = left * pitch;
        const std::size_t right_base = right * pitch;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            const double direction = directions[lane];
            const double left_re = direction * left_minus_re;
            const double left_im = direction * left_minus_im;
            const double right_re = direction * right_minus_re;
            const double right_im = direction * right_minus_im;
            const double r0 = runtime.active_re[left_base + lane];
            const double i0 = runtime.active_im[left_base + lane];
            const double r1 = runtime.active_re[right_base + lane];
            const double i1 = runtime.active_im[right_base + lane];
            runtime.active_re[left_base + lane] = c * r0 + right_re * r1 - right_im * i1;
            runtime.active_im[left_base + lane] = c * i0 + right_re * i1 + right_im * r1;
            runtime.active_re[right_base + lane] = c * r1 + left_re * r0 - left_im * i0;
            runtime.active_im[right_base + lane] = c * i1 + left_re * i0 + left_im * r0;
        }
    }
}

void rotate_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match batch active state");
    }
    const std::size_t dim = active_length(runtime.k);
    if (runtime.dense_shot_major_active) {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            detail::rotate_contiguous_active(
                batch_active_re_for_shot(runtime, shot),
                batch_active_im_for_shot(runtime, shot),
                dim,
                kernel,
                batch_bit_at(sign_bits, shot));
        }
        return;
    }
    if (runtime.active_pitch == 1) {
        detail::rotate_contiguous_active(
            runtime.active_re.data(),
            runtime.active_im.data(),
            dim,
            kernel,
            batch_bit_at(sign_bits, 0));
        return;
    }
    const double c = kernel.cos_kernel_angle;
    if (kernel.is_diagonal) {
        const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
        const auto& directions = fill_shot_coefficient_scalars(runtime, sign_bits, 1.0, -1.0);
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex minus_coefficient = detail::compact_rotation_coefficient(kernel, basis, false);
            const std::size_t base = basis * pitch;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const std::size_t lane = static_cast<std::size_t>(shot);
                const Complex coefficient = directions[lane] * minus_coefficient;
                const double fr = c + coefficient.real();
                const double fi = coefficient.imag();
                const double r = runtime.active_re[base + lane];
                const double i = runtime.active_im[base + lane];
                runtime.active_re[base + lane] = fr * r - fi * i;
                runtime.active_im[base + lane] = fr * i + fi * r;
            }
        }
        return;
    }
    if (kernel.uniform_imag_pairs) {
        rotate_uniform_imag_pairs_batch(runtime, kernel, sign_bits);
        return;
    }
    rotate_compact_pairs_batch(runtime, kernel, sign_bits);
}

void promote_first_dormant_rotation_batch(
    BatchFactoredExecutorState& runtime,
    double kernel_angle,
    const std::vector<std::uint64_t>& sign_bits) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t promoted_dim = 2 * dim;
    if (runtime.dense_shot_major_active && runtime.active_stride < promoted_dim) {
        fail("batch active shot-major stride is too short for dormant promotion");
    }
    if (!runtime.dense_shot_major_active &&
        runtime.active_re.size() < promoted_dim * static_cast<std::size_t>(runtime.active_pitch)) {
        fail("batch active storage has too few columns for dormant promotion");
    }
    const double c = std::cos(kernel_angle);
    const double s = std::sin(kernel_angle);
    if (runtime.dense_shot_major_active) {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const double q = batch_bit_at(sign_bits, shot) ? s : -s;
            detail::promote_contiguous_active(
                batch_active_re_for_shot(runtime, shot),
                batch_active_im_for_shot(runtime, shot),
                dim,
                c,
                q);
        }
        ++runtime.k;
        --runtime.ndormant;
        return;
    }
    if (runtime.active_pitch == 1) {
        const double q = batch_bit_at(sign_bits, 0) ? s : -s;
        detail::promote_contiguous_active(
            runtime.active_re.data(),
            runtime.active_im.data(),
            dim,
            c,
            q);
        ++runtime.k;
        --runtime.ndormant;
        return;
    }
    const auto& coeffs = fill_shot_coefficient_scalars(runtime, sign_bits, -s, s);
    batch_interleaved::table().promote_first_dormant_rotation(
        runtime.active_re.data(),
        runtime.active_im.data(),
        static_cast<std::size_t>(runtime.active_pitch),
        runtime.active_shots,
        dim,
        c,
        coeffs.data());
    ++runtime.k;
    --runtime.ndormant;
}

void sample_batch_measurement_branches_from_true(
    BatchFactoredExecutorState& runtime,
    std::vector<std::uint64_t>& branch_bits,
    std::vector<double>& prob_true,
    std::vector<double>& invnorms) {
    if (branch_bits.size() < runtime.batch_words) {
        branch_bits.resize(runtime.batch_words, 0);
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        std::uint64_t packed = 0;
        for (int bit = 0; bit < live; ++bit) {
            const int shot = base_shot + bit;
            const double pt = std::clamp(prob_true[static_cast<std::size_t>(shot)], 0.0, 1.0);
            const bool branch = sample_bernoulli(runtime.rng_state, pt);
            if (branch) {
                packed |= std::uint64_t{1} << bit;
            }
            const double probability = branch ? pt : 1.0 - pt;
            if (probability <= 0.0) {
                fail("sampled an impossible active measurement branch");
            }
            invnorms[static_cast<std::size_t>(shot)] = 1.0 / std::sqrt(probability);
        }
        branch_bits[word] = packed;
    }
    for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
        branch_bits[word] = 0;
    }
    std::fill(
        invnorms.begin() + static_cast<std::ptrdiff_t>(runtime.active_shots),
        invnorms.begin() + static_cast<std::ptrdiff_t>(runtime.active_pitch),
        1.0);
}

template <typename ProbabilityFn, typename ProjectFn>
void measure_shot_major_active_branch_batch(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    ProbabilityFn probability,
    ProjectFn project) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        const int base_shot = static_cast<int>(word << 6);
        const int live = std::min(64, runtime.active_shots - base_shot);
        std::uint64_t packed = 0;
        for (int bit = 0; bit < live; ++bit) {
            const int shot = base_shot + bit;
            double* re = batch_active_re_for_shot(runtime, shot);
            double* im = batch_active_im_for_shot(runtime, shot);
            const double pt = probability(re, im);
            const bool branch = sample_bernoulli(runtime.rng_state, pt);
            if (branch) {
                packed |= std::uint64_t{1} << bit;
            }
            const double probability_sampled = branch ? pt : 1.0 - pt;
            if (probability_sampled <= 0.0) {
                fail("sampled an impossible active measurement branch");
            }
            project(shot, re, im, branch, 1.0 / std::sqrt(probability_sampled));
        }
        runtime.eval_scratch[word] = packed;
    }
    for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
        runtime.eval_scratch[word] = 0;
    }
    finish_active_measurement_branch(runtime, branch_condition, runtime.eval_scratch);
}

void measure_diagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    const std::size_t out_dim = kernel.out_dim;
    if (runtime.dense_shot_major_active) {
        measure_shot_major_active_branch_batch(
            runtime,
            branch_condition,
            [&](double* re, double* im) {
                return detail::diagonal_probability_contiguous(re, im, kernel, true);
            },
            [&](int, double* re, double* im, bool branch, double invnorm) {
                detail::project_diagonal_contiguous(re, im, kernel, branch, invnorm);
            });
        return;
    }
    if (runtime.active_pitch == 1) {
        runtime.branch_prob_true[0] = detail::diagonal_probability_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            kernel,
            true);
        sample_batch_measurement_branches_from_true(
            runtime,
            runtime.eval_scratch,
            runtime.branch_prob_true,
            runtime.branch_invnorms);
        const auto& branch_bits = runtime.eval_scratch;
        detail::project_diagonal_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            kernel,
            batch_bit_at(branch_bits, 0),
            runtime.branch_invnorms[0]);
        finish_active_measurement_branch(runtime, branch_condition, branch_bits);
        return;
    }
    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    std::fill_n(runtime.branch_prob_true.data(), runtime.active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const std::size_t source =
            detail::compact_diagonal_measurement_source(kernel, idx, true);
        const std::size_t base = source * pitch;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            const double r = runtime.active_re[base + lane];
            const double i = runtime.active_im[base + lane];
            runtime.branch_prob_true[lane] += r * r + i * i;
        }
    }
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    const auto& branch_directions = fill_shot_coefficient_scalars(runtime, branch_bits, 1.0, -1.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const std::size_t out_base = idx * pitch;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            const std::size_t source = detail::compact_diagonal_measurement_source(
                kernel,
                idx,
                branch_directions[lane] < 0.0);
            const std::size_t source_lane = source * pitch + lane;
            runtime.active_re[out_base + lane] = runtime.active_re[source_lane] * runtime.branch_invnorms[lane];
            runtime.active_im[out_base + lane] = runtime.active_im[source_lane] * runtime.branch_invnorms[lane];
        }
    }
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_nondiagonal_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    if (runtime.dense_shot_major_active) {
        measure_shot_major_active_branch_batch(
            runtime,
            branch_condition,
            [&](double* re, double* im) {
                return detail::nondiagonal_probability_contiguous(re, im, kernel, true);
            },
            [&](int shot, double* re, double* im, bool branch, double invnorm) {
                detail::project_nondiagonal_contiguous(
                    re,
                    im,
                    batch_scratch_re_for_shot(runtime, shot),
                    batch_scratch_im_for_shot(runtime, shot),
                    kernel,
                    branch,
                    invnorm);
            });
        return;
    }
    if (runtime.active_pitch == 1) {
        runtime.branch_prob_true[0] = detail::nondiagonal_probability_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            kernel,
            true);
        sample_batch_measurement_branches_from_true(
            runtime,
            runtime.eval_scratch,
            runtime.branch_prob_true,
            runtime.branch_invnorms);
        const auto& branch_bits = runtime.eval_scratch;
        detail::project_nondiagonal_contiguous(
            runtime.active_re.data(),
            runtime.active_im.data(),
            runtime.scratch_re.data(),
            runtime.scratch_im.data(),
            kernel,
            batch_bit_at(branch_bits, 0),
            runtime.branch_invnorms[0]);
        finish_active_measurement_branch(runtime, branch_condition, branch_bits);
        return;
    }
    measure_nondiagonal_true_prob_batch(runtime, kernel);
    sample_batch_measurement_branches_from_true(
        runtime,
        runtime.eval_scratch,
        runtime.branch_prob_true,
        runtime.branch_invnorms);
    const auto& branch_bits = runtime.eval_scratch;
    project_nondiagonal_batch(runtime, kernel, branch_bits);
    finish_active_measurement_branch(runtime, branch_condition, branch_bits);
}

void measure_precomputed_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    if (kernel.action.nqubits != runtime.k) {
        fail("measurement kernel dimension does not match batch active state");
    }
    if (kernel.is_diagonal) {
        measure_diagonal_active_pauli_branch_batch(runtime, kernel, branch_condition);
    } else {
        measure_nondiagonal_active_pauli_branch_batch(runtime, kernel, branch_condition);
    }
}

void measure_precomputed_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    measure_precomputed_active_pauli_branch_batch(runtime, kernel, branch_condition);
    if (write_direct_branch_measurement_record(
            runtime,
            branch_condition,
            outcome_plan,
            record,
            record_condition)) {
        return;
    }
    eval_symbolic_bool_batch(runtime.eval_scratch, outcome_plan, runtime);
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
}

} // namespace symft
