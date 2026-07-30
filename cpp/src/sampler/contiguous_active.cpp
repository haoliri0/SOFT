#include "sampler/contiguous_active.hpp"

#include "sampler/active_kernels.hpp"
#include "simd/simd.hpp"

#include <algorithm>

namespace symft::detail {

void rotate_contiguous_active(
    double* re,
    double* im,
    std::size_t dim,
    const PrecomputedActivePauliRotationKernel& kernel,
    bool sign) {
    const double c = kernel.cos_kernel_angle;
    if (kernel.is_diagonal) {
        SYMFT_SINGLE_SIMD_LOOP
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex coefficient = compact_rotation_coefficient(kernel, basis, sign);
            const double fr = c + coefficient.real();
            const double fi = coefficient.imag();
            const double r = re[basis];
            const double i = im[basis];
            re[basis] = fr * r - fi * i;
            im[basis] = fr * i + fi * r;
        }
        return;
    }

    const std::size_t npairs = kernel.pair_count;
    if (kernel.uniform_imag_pairs) {
        const Complex coefficient = compact_rotation_coefficient(kernel, 0, sign);
        if (npairs < kSimdPairRotationThreshold) {
            rotate_uniform_imag_pairs_soa_inline(
                re,
                im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                c,
                coefficient.imag());
        } else {
            simd::dispatch_table().rotate_uniform_imag_pairs_soa(
                re,
                im,
                dim,
                kernel.action.xmask,
                kernel.pair_bit,
                c,
                coefficient.imag());
        }
        return;
    }

    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t left = insert_zero_bit(idx, static_cast<int>(kernel.pair_bit));
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.action.xmask);
        const bool left_odd = active_action_phase_odd(kernel.action, left);
        const double left_direction = sign != left_odd ? -1.0 : 1.0;
        const double right_direction =
            kernel.action.xz_overlap_odd ? -left_direction : left_direction;
        const double left_re = left_direction * kernel.minus_even_coefficient.real();
        const double left_im = left_direction * kernel.minus_even_coefficient.imag();
        const double right_re = right_direction * kernel.minus_even_coefficient.real();
        const double right_im = right_direction * kernel.minus_even_coefficient.imag();
        const double r0 = re[left];
        const double i0 = im[left];
        const double r1 = re[right];
        const double i1 = im[right];
        re[left] = c * r0 + right_re * r1 - right_im * i1;
        im[left] = c * i0 + right_re * i1 + right_im * r1;
        re[right] = c * r1 + left_re * r0 - left_im * i0;
        im[right] = c * i1 + left_re * i0 + left_im * r0;
    }
}

void promote_contiguous_active(
    double* re,
    double* im,
    std::size_t dim,
    double c,
    double q) {
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double r = re[basis];
        const double i = im[basis];
        re[basis] = c * r;
        im[basis] = c * i;
        re[dim + basis] = -q * i;
        im[dim + basis] = q * r;
    }
}

double diagonal_probability_contiguous(
    const double* re,
    const double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        const std::size_t source =
            compact_diagonal_measurement_source(kernel, idx, branch);
        probability += re[source] * re[source] + im[source] * im[source];
    }
    return std::clamp(probability, 0.0, 1.0);
}

double nondiagonal_probability_contiguous(
    const double* re,
    const double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    const double probability =
        simd::dispatch_table().measure_nondiagonal_probability_soa(
            re,
            im,
            kernel.out_dim << 1,
            kernel.action.xmask,
            kernel.action.zmask,
            static_cast<unsigned>(kernel.pivot),
            kernel.nondiagonal_coefficient1_even,
            branch);
    return std::clamp(probability, 0.0, 1.0);
}

void project_diagonal_contiguous(
    double* re,
    double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double invnorm) {
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < kernel.out_dim; ++idx) {
        const std::size_t source =
            compact_diagonal_measurement_source(kernel, idx, branch);
        re[idx] = re[source] * invnorm;
        im[idx] = im[source] * invnorm;
    }
}

void project_nondiagonal_contiguous(
    double* re,
    double* im,
    double* scratch_re,
    double* scratch_im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double invnorm) {
    const std::size_t out_dim = kernel.out_dim;
    simd::dispatch_table().project_nondiagonal_soa(
        re,
        im,
        scratch_re,
        scratch_im,
        out_dim << 1,
        kernel.action.xmask,
        kernel.action.zmask,
        static_cast<unsigned>(kernel.pivot),
        kernel.nondiagonal_coefficient1_even,
        branch,
        invnorm);
    std::copy_n(scratch_re, out_dim, re);
    std::copy_n(scratch_im, out_dim, im);
}

} // namespace symft::detail
