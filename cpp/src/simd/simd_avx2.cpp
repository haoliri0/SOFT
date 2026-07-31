#include "simd/simd.hpp"

#include <immintrin.h>

namespace symft::simd {
namespace {

#if defined(__clang__)
#define SYMFT_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SIMD_LOOP
#endif

void avx2_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    const auto* cr = reinterpret_cast<const double*>(coeff);
    auto* ar = reinterpret_cast<double*>(alpha);
    const __m256d vc = _mm256_set_pd(0.0, c, 0.0, c);
    const __m256d signs = _mm256_set_pd(1.0, -1.0, 1.0, -1.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        const __m256d a = _mm256_loadu_pd(ar + 2 * i);
        const __m256d k = _mm256_loadu_pd(cr + 2 * i);
        const __m256d factor = _mm256_add_pd(k, vc);
        const __m256d real = _mm256_movedup_pd(a);
        const __m256d imag = _mm256_permute_pd(a, 0b1111);
        const __m256d factor_swapped = _mm256_permute_pd(factor, 0b0101);
        const __m256d out = _mm256_add_pd(_mm256_mul_pd(real, factor), _mm256_mul_pd(_mm256_mul_pd(imag, factor_swapped), signs));
        _mm256_storeu_pd(ar + 2 * i, out);
    }
    for (; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double avx2_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
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

__m256d load_complex_re4(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpacklo_pd(c0, c1), 0xd8);
}

__m256d load_complex_im4(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpackhi_pd(c0, c1), 0xd8);
}

__m256i pair_indices4(std::size_t i0, std::uint64_t xmask) {
    const auto mask = static_cast<std::size_t>(xmask);
    return _mm256_set_epi64x(
        static_cast<long long>((i0 + 3) ^ mask),
        static_cast<long long>((i0 + 2) ^ mask),
        static_cast<long long>((i0 + 1) ^ mask),
        static_cast<long long>(i0 ^ mask));
}

__m256d gather_pair4(const double* values, std::size_t i0, std::uint64_t xmask) {
    return _mm256_i64gather_pd(values, pair_indices4(i0, xmask), 8);
}

void scatter_pair4(double* values, std::size_t i0, std::uint64_t xmask, __m256d lanes) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, lanes);
    const auto mask = static_cast<std::size_t>(xmask);
    values[i0 ^ mask] = tmp[0];
    values[(i0 + 1) ^ mask] = tmp[1];
    values[(i0 + 2) ^ mask] = tmp[2];
    values[(i0 + 3) ^ mask] = tmp[3];
}

__m256d permute_pair_bit1(__m256d lanes, std::uint64_t xmask) {
    return xmask == 2 ? _mm256_permute4x64_pd(lanes, 0x4e) : _mm256_permute4x64_pd(lanes, 0x1b);
}

template <std::size_t LaneMask>
__m256d permute_lanes_xor2_const(__m256d lanes) {
    if constexpr (LaneMask == 0) {
        return lanes;
    } else if constexpr (LaneMask == 1) {
        return _mm256_permute_pd(lanes, 0b0101);
    } else if constexpr (LaneMask == 2) {
        return _mm256_permute4x64_pd(lanes, 0x4e);
    } else {
        return _mm256_permute4x64_pd(lanes, 0x1b);
    }
}

template <std::size_t LaneMask>
void avx2_rotate_uniform_imag_pairs_partner_permute_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t lower_mask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t chunk_mask = static_cast<std::size_t>(lower_mask) & ~std::size_t{3};
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vq = _mm256_set1_pd(q);
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 4) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = block + selector + (offset ^ chunk_mask);
            const __m256d r0 = _mm256_loadu_pd(re + i0);
            const __m256d im0 = _mm256_loadu_pd(im + i0);
            const __m256d r1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(re + i1));
            const __m256d im1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(im + i1));
            _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
            _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
            const __m256d out_r1 = _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1));
            const __m256d out_im1 = _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1));
            _mm256_storeu_pd(re + i1, permute_lanes_xor2_const<LaneMask>(out_r1));
            _mm256_storeu_pd(im + i1, permute_lanes_xor2_const<LaneMask>(out_im1));
        }
    }
}

template <std::size_t LaneMask>
void avx2_rotate_real_pair_flip_partner_permute_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t lower_mask,
    unsigned pair_bit,
    const double* phase_signs,
    double c,
    double base_coeff) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t chunk_mask = static_cast<std::size_t>(lower_mask) & ~std::size_t{3};
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vbase = _mm256_set1_pd(base_coeff);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 4) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = block + selector + (offset ^ chunk_mask);
            const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
            const __m256d r0 = _mm256_loadu_pd(re + i0);
            const __m256d im0 = _mm256_loadu_pd(im + i0);
            const __m256d r1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(re + i1));
            const __m256d im1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(im + i1));
            _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
            _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
            const __m256d out_r1 = _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1));
            const __m256d out_im1 = _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1));
            _mm256_storeu_pd(re + i1, permute_lanes_xor2_const<LaneMask>(out_r1));
            _mm256_storeu_pd(im + i1, permute_lanes_xor2_const<LaneMask>(out_im1));
            pair_idx += 4;
        }
    }
}

template <std::size_t LaneMask>
void avx2_rotate_general_pairs_partner_permute_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t lower_mask,
    unsigned pair_bit,
    const Complex* left_coeff,
    const Complex* right_coeff,
    double c) {
    const auto* left = reinterpret_cast<const double*>(left_coeff);
    const auto* right = reinterpret_cast<const double*>(right_coeff);
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t chunk_mask = static_cast<std::size_t>(lower_mask) & ~std::size_t{3};
    const __m256d vc = _mm256_set1_pd(c);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 4) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = block + selector + (offset ^ chunk_mask);
            const __m256d r0 = _mm256_loadu_pd(re + i0);
            const __m256d im0 = _mm256_loadu_pd(im + i0);
            const __m256d r1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(re + i1));
            const __m256d im1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(im + i1));
            const __m256d lr = load_complex_re4(left + 2 * pair_idx);
            const __m256d li = load_complex_im4(left + 2 * pair_idx);
            const __m256d rr = load_complex_re4(right + 2 * pair_idx);
            const __m256d ri = load_complex_im4(right + 2 * pair_idx);
            __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
            out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
            __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
            out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
            __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
            out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
            __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
            out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
            _mm256_storeu_pd(re + i0, out_r0);
            _mm256_storeu_pd(im + i0, out_i0);
            _mm256_storeu_pd(re + i1, permute_lanes_xor2_const<LaneMask>(out_r1));
            _mm256_storeu_pd(im + i1, permute_lanes_xor2_const<LaneMask>(out_i1));
            pair_idx += 4;
        }
    }
}

std::size_t xor_contiguous_run(std::size_t lower_mask, std::size_t selector) {
    return lower_mask == 0 ? selector : (lower_mask & (~lower_mask + std::size_t{1}));
}

void avx2_mul_assign_soa(double* re, double* im, const Complex* coeff, double c, std::size_t n) {
    const auto* cr = reinterpret_cast<const double*>(coeff);
    const __m256d vc = _mm256_set1_pd(c);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256d r = _mm256_loadu_pd(re + i);
        const __m256d v = _mm256_loadu_pd(im + i);
        const __m256d cre = _mm256_add_pd(vc, load_complex_re4(cr + 2 * i));
        const __m256d cim = load_complex_im4(cr + 2 * i);
        _mm256_storeu_pd(re + i, _mm256_fnmadd_pd(cim, v, _mm256_mul_pd(cre, r)));
        _mm256_storeu_pd(im + i, _mm256_fmadd_pd(cim, r, _mm256_mul_pd(cre, v)));
    }
    for (; i < n; ++i) {
        const double fr = c + cr[2 * i];
        const double fi = cr[2 * i + 1];
        const double r = re[i];
        const double v = im[i];
        re[i] = fr * r - fi * v;
        im[i] = fr * v + fi * r;
    }
}

double avx2_norm_sum_soa(const double* re, const double* im, const std::size_t* indices, std::size_t n) {
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

__m256d nondiagonal_coefficient_sign_mask4(
    std::size_t source0,
    std::uint64_t zmask,
    bool branch) {
    __m256i parity = _mm256_and_si256(
        _mm256_set_epi64x(
            static_cast<long long>(source0 + 3),
            static_cast<long long>(source0 + 2),
            static_cast<long long>(source0 + 1),
            static_cast<long long>(source0)),
        _mm256_set1_epi64x(static_cast<long long>(zmask)));
    parity = _mm256_xor_si256(parity, _mm256_srli_epi64(parity, 32));
    parity = _mm256_xor_si256(parity, _mm256_srli_epi64(parity, 16));
    parity = _mm256_xor_si256(parity, _mm256_srli_epi64(parity, 8));
    parity = _mm256_xor_si256(parity, _mm256_srli_epi64(parity, 4));
    parity = _mm256_xor_si256(parity, _mm256_srli_epi64(parity, 2));
    parity = _mm256_xor_si256(parity, _mm256_srli_epi64(parity, 1));
    parity = _mm256_and_si256(parity, _mm256_set1_epi64x(1));
    if (branch) {
        parity = _mm256_xor_si256(parity, _mm256_set1_epi64x(1));
    }
    return _mm256_castsi256_pd(_mm256_slli_epi64(parity, 63));
}

struct NondiagonalAmplitudes4 {
    __m256d re;
    __m256d im;
};

NondiagonalAmplitudes4 nondiagonal_amplitudes4(
    __m256d r0,
    __m256d im0,
    __m256d r1,
    __m256d im1,
    __m256d coefficient_sign_mask,
    __m256d coefficient1_even_re,
    __m256d coefficient1_even_im) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const __m256d c0 = _mm256_set1_pd(inv_sqrt2);
    const __m256d c1r = _mm256_xor_pd(coefficient1_even_re, coefficient_sign_mask);
    const __m256d c1i = _mm256_xor_pd(coefficient1_even_im, coefficient_sign_mask);
    __m256d ar = _mm256_fmadd_pd(c1r, r1, _mm256_mul_pd(c0, r0));
    ar = _mm256_fnmadd_pd(c1i, im1, ar);
    __m256d ai = _mm256_fmadd_pd(c1r, im1, _mm256_mul_pd(c0, im0));
    ai = _mm256_fmadd_pd(c1i, r1, ai);
    return {ar, ai};
}

double horizontal_sum4(__m256d value) {
    const __m128d lo = _mm256_castpd256_pd128(value);
    const __m128d hi = _mm256_extractf128_pd(value, 1);
    const __m128d pair = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(pair, pair));
}

template <std::size_t LaneMask>
double avx2_measure_nondiagonal_probability_partner_permute_soa(
    const double* re,
    const double* im,
    std::size_t dim,
    std::uint64_t xmask,
    std::uint64_t zmask,
    unsigned pivot,
    Complex coefficient1_even,
    bool branch) {
    const std::size_t selector = std::size_t{1} << pivot;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
    const __m256d coefficient1_even_re = _mm256_set1_pd(coefficient1_even.real());
    const __m256d coefficient1_even_im = _mm256_set1_pd(coefficient1_even.imag());
    __m256d probability = _mm256_setzero_pd();
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 4) {
            const std::size_t source0 = block + offset;
            const std::size_t source1 = block + selector + (offset ^ chunk_mask);
            const __m256d r0 = _mm256_loadu_pd(re + source0);
            const __m256d im0 = _mm256_loadu_pd(im + source0);
            const __m256d r1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(re + source1));
            const __m256d im1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(im + source1));
            const auto amplitudes = nondiagonal_amplitudes4(
                r0,
                im0,
                r1,
                im1,
                nondiagonal_coefficient_sign_mask4(source0, zmask, branch),
                coefficient1_even_re,
                coefficient1_even_im);
            probability = _mm256_fmadd_pd(amplitudes.re, amplitudes.re, probability);
            probability = _mm256_fmadd_pd(amplitudes.im, amplitudes.im, probability);
        }
    }
    return horizontal_sum4(probability);
}

double avx2_measure_nondiagonal_probability_soa(
    const double* re,
    const double* im,
    std::size_t dim,
    std::uint64_t xmask,
    std::uint64_t zmask,
    unsigned pivot,
    Complex coefficient1_even,
    bool branch) {
    if (pivot < 2) {
        return scalar_table().measure_nondiagonal_probability_soa(
            re, im, dim, xmask, zmask, pivot, coefficient1_even, branch);
    }
    const std::size_t lane_mask = static_cast<std::size_t>(xmask) & std::size_t{3};
    switch (lane_mask) {
    case 0:
        return avx2_measure_nondiagonal_probability_partner_permute_soa<0>(
            re, im, dim, xmask, zmask, pivot, coefficient1_even, branch);
    case 1:
        return avx2_measure_nondiagonal_probability_partner_permute_soa<1>(
            re, im, dim, xmask, zmask, pivot, coefficient1_even, branch);
    case 2:
        return avx2_measure_nondiagonal_probability_partner_permute_soa<2>(
            re, im, dim, xmask, zmask, pivot, coefficient1_even, branch);
    default:
        return avx2_measure_nondiagonal_probability_partner_permute_soa<3>(
            re, im, dim, xmask, zmask, pivot, coefficient1_even, branch);
    }
}

template <std::size_t LaneMask>
void avx2_project_nondiagonal_partner_permute_soa(
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
    const std::size_t selector = std::size_t{1} << pivot;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
    const __m256d coefficient1_even_re = _mm256_set1_pd(coefficient1_even.real());
    const __m256d coefficient1_even_im = _mm256_set1_pd(coefficient1_even.imag());
    const __m256d vinvnorm = _mm256_set1_pd(invnorm);
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t output_base = block >> 1;
        for (std::size_t offset = 0; offset < selector; offset += 4) {
            const std::size_t source0 = block + offset;
            const std::size_t source1 = block + selector + (offset ^ chunk_mask);
            const __m256d r0 = _mm256_loadu_pd(re + source0);
            const __m256d im0 = _mm256_loadu_pd(im + source0);
            const __m256d r1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(re + source1));
            const __m256d im1 = permute_lanes_xor2_const<LaneMask>(_mm256_loadu_pd(im + source1));
            const auto amplitudes = nondiagonal_amplitudes4(
                r0,
                im0,
                r1,
                im1,
                nondiagonal_coefficient_sign_mask4(source0, zmask, branch),
                coefficient1_even_re,
                coefficient1_even_im);
            _mm256_storeu_pd(out_re + output_base + offset, _mm256_mul_pd(amplitudes.re, vinvnorm));
            _mm256_storeu_pd(out_im + output_base + offset, _mm256_mul_pd(amplitudes.im, vinvnorm));
        }
    }
}

void avx2_project_nondiagonal_soa(
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
    if (pivot < 2) {
        scalar_table().project_nondiagonal_soa(
            re, im, out_re, out_im, dim, xmask, zmask, pivot, coefficient1_even, branch, invnorm);
        return;
    }
    const std::size_t lane_mask = static_cast<std::size_t>(xmask) & std::size_t{3};
    switch (lane_mask) {
    case 0:
        avx2_project_nondiagonal_partner_permute_soa<0>(
            re, im, out_re, out_im, dim, xmask, zmask, pivot, coefficient1_even, branch, invnorm);
        break;
    case 1:
        avx2_project_nondiagonal_partner_permute_soa<1>(
            re, im, out_re, out_im, dim, xmask, zmask, pivot, coefficient1_even, branch, invnorm);
        break;
    case 2:
        avx2_project_nondiagonal_partner_permute_soa<2>(
            re, im, out_re, out_im, dim, xmask, zmask, pivot, coefficient1_even, branch, invnorm);
        break;
    default:
        avx2_project_nondiagonal_partner_permute_soa<3>(
            re, im, out_re, out_im, dim, xmask, zmask, pivot, coefficient1_even, branch, invnorm);
        break;
    }
}

void avx2_rotate_uniform_imag_pairs_soa(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const bool contiguous_partner = lower_mask == 0;
    const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vq = _mm256_set1_pd(q);
    if (selector == 1 && xmask == 1 && dim >= 4) {
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v)));
        }
        for (; basis < dim; basis += 2) {
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            re[basis] = c * r0 - q * im1;
            im[basis] = c * im0 + q * r1;
            re[basis + 1] = c * r1 - q * im0;
            im[basis + 1] = c * im1 + q * r0;
        }
        return;
    }
    if (pair_bit == 1 && (xmask == 2 || xmask == 3) && dim >= 4) {
        for (std::size_t basis = 0; basis + 4 <= dim; basis += 4) {
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = permute_pair_bit1(r, xmask);
            const __m256d swapped_v = permute_pair_bit1(v, xmask);
            _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v)));
        }
        return;
    }
    if (pair_bit >= 2 && dim >= 4) {
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        switch (lane_mask) {
        case 0:
            avx2_rotate_uniform_imag_pairs_partner_permute_soa<0>(re, im, dim, lower_mask, pair_bit, c, q);
            break;
        case 1:
            avx2_rotate_uniform_imag_pairs_partner_permute_soa<1>(re, im, dim, lower_mask, pair_bit, c, q);
            break;
        case 2:
            avx2_rotate_uniform_imag_pairs_partner_permute_soa<2>(re, im, dim, lower_mask, pair_bit, c, q);
            break;
        default:
            avx2_rotate_uniform_imag_pairs_partner_permute_soa<3>(re, im, dim, lower_mask, pair_bit, c, q);
            break;
        }
        return;
    }
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = _mm256_loadu_pd(re + i1);
                const __m256d im1 = _mm256_loadu_pd(im + i1);
                _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                _mm256_storeu_pd(re + i1, _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1)));
                _mm256_storeu_pd(im + i1, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1)));
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 4) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 4 <= xor_run; local += 4) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m256d r0 = _mm256_loadu_pd(re + i0);
                        const __m256d im0 = _mm256_loadu_pd(im + i0);
                        const __m256d r1 = _mm256_loadu_pd(re + i1);
                        const __m256d im1 = _mm256_loadu_pd(im + i1);
                        _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                        _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                        _mm256_storeu_pd(re + i1, _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1)));
                        _mm256_storeu_pd(im + i1, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1)));
                    }
                }
                offset = selector;
            } else {
                for (; offset + 4 <= selector; offset += 4) {
                    const std::size_t i0 = block + offset;
                    const __m256d r0 = _mm256_loadu_pd(re + i0);
                    const __m256d im0 = _mm256_loadu_pd(im + i0);
                    const __m256d r1 = gather_pair4(re, i0, xmask);
                    const __m256d im1 = gather_pair4(im, i0, xmask);
                    _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                    _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                    scatter_pair4(re, i0, xmask, _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1)));
                    scatter_pair4(im, i0, xmask, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1)));
                }
            }
        }
        for (; offset < selector; ++offset) {
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

void avx2_rotate_real_pair_flip_soa(
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
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const bool contiguous_partner = lower_mask == 0;
    const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vbase = _mm256_set1_pd(base_coeff);
    if (selector == 1 && xmask == 1 && dim >= 4) {
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const double q0 = phase_signs[basis >> 1] * base_coeff;
            const double q1 = phase_signs[(basis >> 1) + 1] * base_coeff;
            const __m256d signed_q = _mm256_set_pd(q1, -q1, q0, -q0);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            _mm256_storeu_pd(re + basis, _mm256_fmadd_pd(signed_q, swapped_r, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(signed_q, swapped_v, _mm256_mul_pd(vc, v)));
        }
        for (; basis < dim; basis += 2) {
            const double q = phase_signs[basis >> 1] * base_coeff;
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            re[basis] = c * r0 - q * r1;
            im[basis] = c * im0 - q * im1;
            re[basis + 1] = c * r1 + q * r0;
            im[basis + 1] = c * im1 + q * im0;
        }
        return;
    }
    if (pair_bit == 1 && (xmask == 2 || xmask == 3) && dim >= 4) {
        std::size_t pair_idx = 0;
        for (std::size_t basis = 0; basis + 4 <= dim; basis += 4) {
            const double q0 = phase_signs[pair_idx] * base_coeff;
            const double q1 = phase_signs[pair_idx + 1] * base_coeff;
            const __m256d signed_q =
                xmask == 2 ? _mm256_set_pd(q1, q0, -q1, -q0) : _mm256_set_pd(q0, q1, -q1, -q0);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = permute_pair_bit1(r, xmask);
            const __m256d swapped_v = permute_pair_bit1(v, xmask);
            _mm256_storeu_pd(re + basis, _mm256_fmadd_pd(signed_q, swapped_r, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(signed_q, swapped_v, _mm256_mul_pd(vc, v)));
            pair_idx += 2;
        }
        return;
    }
    if (pair_bit >= 2 && dim >= 4) {
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        switch (lane_mask) {
        case 0:
            avx2_rotate_real_pair_flip_partner_permute_soa<0>(
                re, im, dim, lower_mask, pair_bit, phase_signs, c, base_coeff);
            break;
        case 1:
            avx2_rotate_real_pair_flip_partner_permute_soa<1>(
                re, im, dim, lower_mask, pair_bit, phase_signs, c, base_coeff);
            break;
        case 2:
            avx2_rotate_real_pair_flip_partner_permute_soa<2>(
                re, im, dim, lower_mask, pair_bit, phase_signs, c, base_coeff);
            break;
        default:
            avx2_rotate_real_pair_flip_partner_permute_soa<3>(
                re, im, dim, lower_mask, pair_bit, phase_signs, c, base_coeff);
            break;
        }
        return;
    }
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = _mm256_loadu_pd(re + i1);
                const __m256d im1 = _mm256_loadu_pd(im + i1);
                _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                _mm256_storeu_pd(re + i1, _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1)));
                _mm256_storeu_pd(im + i1, _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1)));
                pair_idx += 4;
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 4) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 4 <= xor_run; local += 4) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
                        const __m256d r0 = _mm256_loadu_pd(re + i0);
                        const __m256d im0 = _mm256_loadu_pd(im + i0);
                        const __m256d r1 = _mm256_loadu_pd(re + i1);
                        const __m256d im1 = _mm256_loadu_pd(im + i1);
                        _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                        _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                        _mm256_storeu_pd(re + i1, _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1)));
                        _mm256_storeu_pd(im + i1, _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1)));
                        pair_idx += 4;
                    }
                }
                offset = selector;
            } else {
                for (; offset + 4 <= selector; offset += 4) {
                    const std::size_t i0 = block + offset;
                    const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
                    const __m256d r0 = _mm256_loadu_pd(re + i0);
                    const __m256d im0 = _mm256_loadu_pd(im + i0);
                    const __m256d r1 = gather_pair4(re, i0, xmask);
                    const __m256d im1 = gather_pair4(im, i0, xmask);
                    _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                    _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                    scatter_pair4(re, i0, xmask, _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1)));
                    scatter_pair4(im, i0, xmask, _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1)));
                    pair_idx += 4;
                }
            }
        }
        for (; offset < selector; ++offset) {
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

void avx2_rotate_general_pairs_soa(
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
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const bool contiguous_partner = lower_mask == 0;
    const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
    const __m256d vc = _mm256_set1_pd(c);
    if (selector == 1 && xmask == 1 && dim >= 4) {
        std::size_t basis = 0;
        for (; basis + 4 <= dim; basis += 4) {
            const std::size_t pair_idx = basis >> 1;
            const __m256d coeff_r =
                _mm256_set_pd(left[2 * (pair_idx + 1)], right[2 * (pair_idx + 1)], left[2 * pair_idx], right[2 * pair_idx]);
            const __m256d coeff_i = _mm256_set_pd(
                left[2 * (pair_idx + 1) + 1],
                right[2 * (pair_idx + 1) + 1],
                left[2 * pair_idx + 1],
                right[2 * pair_idx + 1]);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = _mm256_permute_pd(r, 0b0101);
            const __m256d swapped_v = _mm256_permute_pd(v, 0b0101);
            __m256d out_r = _mm256_fmadd_pd(vc, r, _mm256_mul_pd(coeff_r, swapped_r));
            out_r = _mm256_fnmadd_pd(coeff_i, swapped_v, out_r);
            __m256d out_i = _mm256_fmadd_pd(vc, v, _mm256_mul_pd(coeff_r, swapped_v));
            out_i = _mm256_fmadd_pd(coeff_i, swapped_r, out_i);
            _mm256_storeu_pd(re + basis, out_r);
            _mm256_storeu_pd(im + basis, out_i);
        }
        for (; basis < dim; basis += 2) {
            const std::size_t pair_idx = basis >> 1;
            const double r0 = re[basis];
            const double im0 = im[basis];
            const double r1 = re[basis + 1];
            const double im1 = im[basis + 1];
            const double lr = left[2 * pair_idx];
            const double li = left[2 * pair_idx + 1];
            const double rr = right[2 * pair_idx];
            const double ri = right[2 * pair_idx + 1];
            re[basis] = c * r0 + rr * r1 - ri * im1;
            im[basis] = c * im0 + rr * im1 + ri * r1;
            re[basis + 1] = c * r1 + lr * r0 - li * im0;
            im[basis + 1] = c * im1 + lr * im0 + li * r0;
        }
        return;
    }
    if (pair_bit == 1 && (xmask == 2 || xmask == 3) && dim >= 4) {
        std::size_t pair_idx = 0;
        for (std::size_t basis = 0; basis + 4 <= dim; basis += 4) {
            const __m256d coeff_r = xmask == 2
                                        ? _mm256_set_pd(
                                              left[2 * (pair_idx + 1)],
                                              left[2 * pair_idx],
                                              right[2 * (pair_idx + 1)],
                                              right[2 * pair_idx])
                                        : _mm256_set_pd(
                                              left[2 * pair_idx],
                                              left[2 * (pair_idx + 1)],
                                              right[2 * (pair_idx + 1)],
                                              right[2 * pair_idx]);
            const __m256d coeff_i = xmask == 2
                                        ? _mm256_set_pd(
                                              left[2 * (pair_idx + 1) + 1],
                                              left[2 * pair_idx + 1],
                                              right[2 * (pair_idx + 1) + 1],
                                              right[2 * pair_idx + 1])
                                        : _mm256_set_pd(
                                              left[2 * pair_idx + 1],
                                              left[2 * (pair_idx + 1) + 1],
                                              right[2 * (pair_idx + 1) + 1],
                                              right[2 * pair_idx + 1]);
            const __m256d r = _mm256_loadu_pd(re + basis);
            const __m256d v = _mm256_loadu_pd(im + basis);
            const __m256d swapped_r = permute_pair_bit1(r, xmask);
            const __m256d swapped_v = permute_pair_bit1(v, xmask);
            __m256d out_r = _mm256_fmadd_pd(vc, r, _mm256_mul_pd(coeff_r, swapped_r));
            out_r = _mm256_fnmadd_pd(coeff_i, swapped_v, out_r);
            __m256d out_i = _mm256_fmadd_pd(vc, v, _mm256_mul_pd(coeff_r, swapped_v));
            out_i = _mm256_fmadd_pd(coeff_i, swapped_r, out_i);
            _mm256_storeu_pd(re + basis, out_r);
            _mm256_storeu_pd(im + basis, out_i);
            pair_idx += 2;
        }
        return;
    }
    if (pair_bit >= 2 && dim >= 4) {
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        switch (lane_mask) {
        case 0:
            avx2_rotate_general_pairs_partner_permute_soa<0>(
                re, im, dim, lower_mask, pair_bit, left_coeff, right_coeff, c);
            break;
        case 1:
            avx2_rotate_general_pairs_partner_permute_soa<1>(
                re, im, dim, lower_mask, pair_bit, left_coeff, right_coeff, c);
            break;
        case 2:
            avx2_rotate_general_pairs_partner_permute_soa<2>(
                re, im, dim, lower_mask, pair_bit, left_coeff, right_coeff, c);
            break;
        default:
            avx2_rotate_general_pairs_partner_permute_soa<3>(
                re, im, dim, lower_mask, pair_bit, left_coeff, right_coeff, c);
            break;
        }
        return;
    }
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = _mm256_loadu_pd(re + i1);
                const __m256d im1 = _mm256_loadu_pd(im + i1);
                const __m256d lr = load_complex_re4(left + 2 * pair_idx);
                const __m256d li = load_complex_im4(left + 2 * pair_idx);
                const __m256d rr = load_complex_re4(right + 2 * pair_idx);
                const __m256d ri = load_complex_im4(right + 2 * pair_idx);
                __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
                out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
                __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
                out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
                __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
                out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
                __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
                out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
                _mm256_storeu_pd(re + i0, out_r0);
                _mm256_storeu_pd(im + i0, out_i0);
                _mm256_storeu_pd(re + i1, out_r1);
                _mm256_storeu_pd(im + i1, out_i1);
                pair_idx += 4;
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 4) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 4 <= xor_run; local += 4) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m256d r0 = _mm256_loadu_pd(re + i0);
                        const __m256d im0 = _mm256_loadu_pd(im + i0);
                        const __m256d r1 = _mm256_loadu_pd(re + i1);
                        const __m256d im1 = _mm256_loadu_pd(im + i1);
                        const __m256d lr = load_complex_re4(left + 2 * pair_idx);
                        const __m256d li = load_complex_im4(left + 2 * pair_idx);
                        const __m256d rr = load_complex_re4(right + 2 * pair_idx);
                        const __m256d ri = load_complex_im4(right + 2 * pair_idx);
                        __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
                        out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
                        __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
                        out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
                        __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
                        out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
                        __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
                        out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
                        _mm256_storeu_pd(re + i0, out_r0);
                        _mm256_storeu_pd(im + i0, out_i0);
                        _mm256_storeu_pd(re + i1, out_r1);
                        _mm256_storeu_pd(im + i1, out_i1);
                        pair_idx += 4;
                    }
                }
                offset = selector;
            } else {
                for (; offset + 4 <= selector; offset += 4) {
                    const std::size_t i0 = block + offset;
                    const __m256d r0 = _mm256_loadu_pd(re + i0);
                    const __m256d im0 = _mm256_loadu_pd(im + i0);
                    const __m256d r1 = gather_pair4(re, i0, xmask);
                    const __m256d im1 = gather_pair4(im, i0, xmask);
                    const __m256d lr = load_complex_re4(left + 2 * pair_idx);
                    const __m256d li = load_complex_im4(left + 2 * pair_idx);
                    const __m256d rr = load_complex_re4(right + 2 * pair_idx);
                    const __m256d ri = load_complex_im4(right + 2 * pair_idx);
                    __m256d out_r0 = _mm256_fmadd_pd(vc, r0, _mm256_mul_pd(rr, r1));
                    out_r0 = _mm256_fnmadd_pd(ri, im1, out_r0);
                    __m256d out_i0 = _mm256_fmadd_pd(vc, im0, _mm256_mul_pd(rr, im1));
                    out_i0 = _mm256_fmadd_pd(ri, r1, out_i0);
                    __m256d out_r1 = _mm256_fmadd_pd(vc, r1, _mm256_mul_pd(lr, r0));
                    out_r1 = _mm256_fnmadd_pd(li, im0, out_r1);
                    __m256d out_i1 = _mm256_fmadd_pd(vc, im1, _mm256_mul_pd(lr, im0));
                    out_i1 = _mm256_fmadd_pd(li, r0, out_i1);
                    _mm256_storeu_pd(re + i0, out_r0);
                    _mm256_storeu_pd(im + i0, out_i0);
                    scatter_pair4(re, i0, xmask, out_r1);
                    scatter_pair4(im, i0, xmask, out_i1);
                    pair_idx += 4;
                }
            }
        }
        for (; offset < selector; ++offset) {
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
    "avx2",
    avx2_mul_assign,
    avx2_norm_sum,
    avx2_mul_assign_soa,
    avx2_norm_sum_soa,
    avx2_measure_nondiagonal_probability_soa,
    avx2_project_nondiagonal_soa,
    avx2_rotate_uniform_imag_pairs_soa,
    avx2_rotate_real_pair_flip_soa,
    avx2_rotate_general_pairs_soa,
};

} // namespace

const KernelTable& avx2_table() {
    return table;
}

} // namespace symft::simd
