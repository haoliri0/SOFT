#include "simd/simd.hpp"

#include <bit>
#include <cmath>

namespace symft::simd {
namespace {

#if defined(__clang__)
#define SYMFT_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SIMD_LOOP
#endif

void scalar_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double scalar_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
    const auto* raw = reinterpret_cast<const double*>(alpha);
    double out = 0.0;
    SYMFT_SIMD_LOOP
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = 2 * indices[i];
        const double r = raw[src];
        const double im = raw[src + 1];
        out += r * r + im * im;
    }
    return out;
}

void scalar_mul_assign_soa(double* re, double* im, const Complex* coeff, double c, std::size_t n) {
    const auto* cr = reinterpret_cast<const double*>(coeff);
    SYMFT_SIMD_LOOP
    for (std::size_t i = 0; i < n; ++i) {
        const double fr = c + cr[2 * i];
        const double fi = cr[2 * i + 1];
        const double r = re[i];
        const double v = im[i];
        re[i] = fr * r - fi * v;
        im[i] = fr * v + fi * r;
    }
}

double scalar_norm_sum_soa(const double* re, const double* im, const std::size_t* indices, std::size_t n) {
    double out = 0.0;
    SYMFT_SIMD_LOOP
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t src = indices[i];
        const double r = re[src];
        const double v = im[src];
        out += r * r + v * v;
    }
    return out;
}

std::size_t insert_zero_bit(std::size_t packed, unsigned bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - 1;
    return (packed & low_mask) | ((packed & ~low_mask) << 1);
}

double scalar_measure_nondiagonal_probability_soa(
    const double* re,
    const double* im,
    std::size_t dim,
    std::uint64_t xmask,
    std::uint64_t zmask,
    unsigned pivot,
    Complex coefficient1_even,
    bool branch) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const std::size_t out_dim = dim >> 1;
    double probability = 0.0;
    SYMFT_SIMD_LOOP
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const std::size_t source0 = insert_zero_bit(idx, pivot);
        const std::size_t source1 = source0 ^ static_cast<std::size_t>(xmask);
        const bool odd = (std::popcount(static_cast<std::uint64_t>(source0) & zmask) & 1) != 0;
        const double direction = branch != odd ? -1.0 : 1.0;
        const double c1r = direction * coefficient1_even.real();
        const double c1i = direction * coefficient1_even.imag();
        const double ar = inv_sqrt2 * re[source0] + c1r * re[source1] - c1i * im[source1];
        const double ai = inv_sqrt2 * im[source0] + c1r * im[source1] + c1i * re[source1];
        probability += ar * ar + ai * ai;
    }
    return probability;
}

void scalar_project_nondiagonal_soa(
    const double* re,
    const double* im,
    double* out_re,
    double* out_im,
    std::size_t dim,
    std::uint64_t xmask,
    std::uint64_t zmask,
    unsigned pivot,
    Complex coefficient1_even,
    bool branch,
    double invnorm) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const std::size_t out_dim = dim >> 1;
    SYMFT_SIMD_LOOP
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const std::size_t source0 = insert_zero_bit(idx, pivot);
        const std::size_t source1 = source0 ^ static_cast<std::size_t>(xmask);
        const bool odd = (std::popcount(static_cast<std::uint64_t>(source0) & zmask) & 1) != 0;
        const double direction = branch != odd ? -1.0 : 1.0;
        const double c1r = direction * coefficient1_even.real();
        const double c1i = direction * coefficient1_even.imag();
        const double ar = inv_sqrt2 * re[source0] + c1r * re[source1] - c1i * im[source1];
        const double ai = inv_sqrt2 * im[source0] + c1r * im[source1] + c1i * re[source1];
        out_re[idx] = ar * invnorm;
        out_im[idx] = ai * invnorm;
    }
}

void scalar_rotate_uniform_imag_pairs_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    for (std::size_t block = 0; block < dim; block += step) {
        SYMFT_SIMD_LOOP
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            re[i0] = c * r0 - q * im1;
            im[i0] = c * im0 + q * r1;
            re[i1] = c * r1 - q * im0;
            im[i1] = c * im1 + q * r0;
        }
    }
}

void scalar_rotate_real_pair_flip_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* phase_signs,
    double c,
    double base_coeff) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        SYMFT_SIMD_LOOP
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            const double q = phase_signs[pair_idx++] * base_coeff;
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            re[i0] = c * r0 - q * r1;
            im[i0] = c * im0 - q * im1;
            re[i1] = c * r1 + q * r0;
            im[i1] = c * im1 + q * im0;
        }
    }
}

void scalar_rotate_general_pairs_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    const Complex* left_coeff,
    const Complex* right_coeff,
    double c) {
    const auto* left = reinterpret_cast<const double*>(left_coeff);
    const auto* right = reinterpret_cast<const double*>(right_coeff);
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        SYMFT_SIMD_LOOP
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            const double r0 = re[i0];
            const double im0 = im[i0];
            const double r1 = re[i1];
            const double im1 = im[i1];
            const double lr = left[2 * pair_idx];
            const double li = left[2 * pair_idx + 1];
            const double rr = right[2 * pair_idx];
            const double ri = right[2 * pair_idx + 1];
            re[i0] = c * r0 + rr * r1 - ri * im1;
            im[i0] = c * im0 + rr * im1 + ri * r1;
            re[i1] = c * r1 + lr * r0 - li * im0;
            im[i1] = c * im1 + lr * im0 + li * r0;
            ++pair_idx;
        }
    }
}

const KernelTable table = {
    "scalar",
    scalar_mul_assign,
    scalar_norm_sum,
    scalar_mul_assign_soa,
    scalar_norm_sum_soa,
    scalar_measure_nondiagonal_probability_soa,
    scalar_project_nondiagonal_soa,
    scalar_rotate_uniform_imag_pairs_soa,
    scalar_rotate_real_pair_flip_soa,
    scalar_rotate_general_pairs_soa,
};

} // namespace

const KernelTable& scalar_table() {
    return table;
}

} // namespace symft::simd
