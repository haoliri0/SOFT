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

void avx512_mul_assign(Complex* alpha, const Complex* coeff, double c, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        alpha[i] *= Complex(c, 0.0) + coeff[i];
    }
}

double avx512_norm_sum(const Complex* alpha, const std::size_t* indices, std::size_t n) {
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

void avx512_mul_assign_soa(double* re, double* im, const Complex* coeff, double c, std::size_t n) {
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

double avx512_norm_sum_soa(const double* re, const double* im, const std::size_t* indices, std::size_t n) {
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

__m512d permute_lanes_xor3(__m512d lanes, std::size_t mask);

__m512d nondiagonal_coefficient_sign_mask8(
    std::size_t source0,
    std::uint64_t zmask,
    bool branch) {
    __m512i parity = _mm512_and_si512(
        _mm512_set_epi64(
            static_cast<long long>(source0 + 7),
            static_cast<long long>(source0 + 6),
            static_cast<long long>(source0 + 5),
            static_cast<long long>(source0 + 4),
            static_cast<long long>(source0 + 3),
            static_cast<long long>(source0 + 2),
            static_cast<long long>(source0 + 1),
            static_cast<long long>(source0)),
        _mm512_set1_epi64(static_cast<long long>(zmask)));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi64(parity, 32));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi64(parity, 16));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi64(parity, 8));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi64(parity, 4));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi64(parity, 2));
    parity = _mm512_xor_si512(parity, _mm512_srli_epi64(parity, 1));
    parity = _mm512_and_si512(parity, _mm512_set1_epi64(1));
    if (branch) {
        parity = _mm512_xor_si512(parity, _mm512_set1_epi64(1));
    }
    return _mm512_castsi512_pd(_mm512_slli_epi64(parity, 63));
}

struct NondiagonalAmplitudes8 {
    __m512d re;
    __m512d im;
};

NondiagonalAmplitudes8 nondiagonal_amplitudes8(
    __m512d r0,
    __m512d im0,
    __m512d r1,
    __m512d im1,
    __m512d coefficient_sign_mask,
    __m512d coefficient1_even_re,
    __m512d coefficient1_even_im) {
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    const __m512d c0 = _mm512_set1_pd(inv_sqrt2);
    const __m512d c1r = _mm512_xor_pd(coefficient1_even_re, coefficient_sign_mask);
    const __m512d c1i = _mm512_xor_pd(coefficient1_even_im, coefficient_sign_mask);
    __m512d ar = _mm512_fmadd_pd(c1r, r1, _mm512_mul_pd(c0, r0));
    ar = _mm512_fnmadd_pd(c1i, im1, ar);
    __m512d ai = _mm512_fmadd_pd(c1r, im1, _mm512_mul_pd(c0, im0));
    ai = _mm512_fmadd_pd(c1i, r1, ai);
    return {ar, ai};
}

double avx512_measure_nondiagonal_probability_soa(
    const double* re,
    const double* im,
    std::size_t dim,
    std::uint64_t xmask,
    std::uint64_t zmask,
    unsigned pivot,
    Complex coefficient1_even,
    bool branch) {
    if (pivot < 3) {
        return scalar_table().measure_nondiagonal_probability_soa(
            re, im, dim, xmask, zmask, pivot, coefficient1_even, branch);
    }
    const std::size_t selector = std::size_t{1} << pivot;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const std::size_t lane_mask = lower_mask & std::size_t{7};
    const std::size_t chunk_mask = lower_mask & ~std::size_t{7};
    const __m512d coefficient1_even_re = _mm512_set1_pd(coefficient1_even.real());
    const __m512d coefficient1_even_im = _mm512_set1_pd(coefficient1_even.imag());
    __m512d probability = _mm512_setzero_pd();
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 8) {
            const std::size_t source0 = block + offset;
            const std::size_t source1 = block + selector + (offset ^ chunk_mask);
            const __m512d r0 = _mm512_loadu_pd(re + source0);
            const __m512d im0 = _mm512_loadu_pd(im + source0);
            const __m512d r1 = permute_lanes_xor3(_mm512_loadu_pd(re + source1), lane_mask);
            const __m512d im1 = permute_lanes_xor3(_mm512_loadu_pd(im + source1), lane_mask);
            const auto amplitudes = nondiagonal_amplitudes8(
                r0,
                im0,
                r1,
                im1,
                nondiagonal_coefficient_sign_mask8(source0, zmask, branch),
                coefficient1_even_re,
                coefficient1_even_im);
            probability = _mm512_fmadd_pd(amplitudes.re, amplitudes.re, probability);
            probability = _mm512_fmadd_pd(amplitudes.im, amplitudes.im, probability);
        }
    }
    return _mm512_reduce_add_pd(probability);
}

void avx512_project_nondiagonal_soa(
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
    if (pivot < 3) {
        scalar_table().project_nondiagonal_soa(
            re, im, out_re, out_im, dim, xmask, zmask, pivot, coefficient1_even, branch, invnorm);
        return;
    }
    const std::size_t selector = std::size_t{1} << pivot;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const std::size_t lane_mask = lower_mask & std::size_t{7};
    const std::size_t chunk_mask = lower_mask & ~std::size_t{7};
    const __m512d coefficient1_even_re = _mm512_set1_pd(coefficient1_even.real());
    const __m512d coefficient1_even_im = _mm512_set1_pd(coefficient1_even.imag());
    const __m512d vinvnorm = _mm512_set1_pd(invnorm);
    for (std::size_t block = 0; block < dim; block += step) {
        const std::size_t output_base = block >> 1;
        for (std::size_t offset = 0; offset < selector; offset += 8) {
            const std::size_t source0 = block + offset;
            const std::size_t source1 = block + selector + (offset ^ chunk_mask);
            const __m512d r0 = _mm512_loadu_pd(re + source0);
            const __m512d im0 = _mm512_loadu_pd(im + source0);
            const __m512d r1 = permute_lanes_xor3(_mm512_loadu_pd(re + source1), lane_mask);
            const __m512d im1 = permute_lanes_xor3(_mm512_loadu_pd(im + source1), lane_mask);
            const auto amplitudes = nondiagonal_amplitudes8(
                r0,
                im0,
                r1,
                im1,
                nondiagonal_coefficient_sign_mask8(source0, zmask, branch),
                coefficient1_even_re,
                coefficient1_even_im);
            _mm512_storeu_pd(out_re + output_base + offset, _mm512_mul_pd(amplitudes.re, vinvnorm));
            _mm512_storeu_pd(out_im + output_base + offset, _mm512_mul_pd(amplitudes.im, vinvnorm));
        }
    }
}

__m512i pair_indices8(std::size_t i0, std::uint64_t xmask) {
    const auto mask = static_cast<std::size_t>(xmask);
    return _mm512_set_epi64(
        static_cast<long long>((i0 + 7) ^ mask),
        static_cast<long long>((i0 + 6) ^ mask),
        static_cast<long long>((i0 + 5) ^ mask),
        static_cast<long long>((i0 + 4) ^ mask),
        static_cast<long long>((i0 + 3) ^ mask),
        static_cast<long long>((i0 + 2) ^ mask),
        static_cast<long long>((i0 + 1) ^ mask),
        static_cast<long long>(i0 ^ mask));
}

__m512d gather_pair8(const double* values, std::size_t i0, std::uint64_t xmask) {
    return _mm512_i64gather_pd(pair_indices8(i0, xmask), values, 8);
}

void scatter_pair8(double* values, std::size_t i0, std::uint64_t xmask, __m512d lanes) {
    _mm512_i64scatter_pd(values, pair_indices8(i0, xmask), lanes, 8);
}

__m512d permute_pair_bit0(__m512d lanes) {
    const __m512i indices = _mm512_set_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    return _mm512_permutexvar_pd(indices, lanes);
}

__m512d permute_pair_bit1(__m512d lanes, std::uint64_t xmask) {
    if (xmask == 2) {
        const __m512i indices = _mm512_set_epi64(5, 4, 7, 6, 1, 0, 3, 2);
        return _mm512_permutexvar_pd(indices, lanes);
    }
    const __m512i indices = _mm512_set_epi64(4, 5, 6, 7, 0, 1, 2, 3);
    return _mm512_permutexvar_pd(indices, lanes);
}

__m512d permute_lanes_xor3(__m512d lanes, std::size_t mask) {
    switch (mask & std::size_t{7}) {
    case 0:
        return lanes;
    case 1:
        return permute_pair_bit0(lanes);
    case 2:
        return permute_pair_bit1(lanes, 2);
    case 3:
        return permute_pair_bit1(lanes, 3);
    case 4: {
        const __m512i indices = _mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4);
        return _mm512_permutexvar_pd(indices, lanes);
    }
    case 5: {
        const __m512i indices = _mm512_set_epi64(2, 3, 0, 1, 6, 7, 4, 5);
        return _mm512_permutexvar_pd(indices, lanes);
    }
    case 6: {
        const __m512i indices = _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6);
        return _mm512_permutexvar_pd(indices, lanes);
    }
    default: {
        const __m512i indices = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
        return _mm512_permutexvar_pd(indices, lanes);
    }
    }
}

__m512i complex_indices8(std::size_t pair_idx, int component) {
    const std::size_t base = pair_idx << 1;
    return _mm512_set_epi64(
        static_cast<long long>(base + 14 + component),
        static_cast<long long>(base + 12 + component),
        static_cast<long long>(base + 10 + component),
        static_cast<long long>(base + 8 + component),
        static_cast<long long>(base + 6 + component),
        static_cast<long long>(base + 4 + component),
        static_cast<long long>(base + 2 + component),
        static_cast<long long>(base + component));
}

__m512d load_complex_re8(const double* coeff, std::size_t pair_idx) {
    return _mm512_i64gather_pd(complex_indices8(pair_idx, 0), coeff, 8);
}

__m512d load_complex_im8(const double* coeff, std::size_t pair_idx) {
    return _mm512_i64gather_pd(complex_indices8(pair_idx, 1), coeff, 8);
}

std::size_t xor_contiguous_run(std::size_t lower_mask, std::size_t selector) {
    return lower_mask == 0 ? selector : (lower_mask & (~lower_mask + std::size_t{1}));
}

void avx512_rotate_uniform_imag_pairs_soa(
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
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d vq = _mm512_set1_pd(q);
    if (selector == 1 && xmask == 1 && dim >= 8) {
        for (std::size_t basis = 0; basis + 8 <= dim; basis += 8) {
            const __m512d r = _mm512_loadu_pd(re + basis);
            const __m512d v = _mm512_loadu_pd(im + basis);
            const __m512d swapped_r = permute_pair_bit0(r);
            const __m512d swapped_v = permute_pair_bit0(v);
            _mm512_storeu_pd(re + basis, _mm512_fnmadd_pd(vq, swapped_v, _mm512_mul_pd(vc, r)));
            _mm512_storeu_pd(im + basis, _mm512_fmadd_pd(vq, swapped_r, _mm512_mul_pd(vc, v)));
        }
        return;
    }
    if (pair_bit == 1 && (xmask == 2 || xmask == 3) && dim >= 8) {
        for (std::size_t basis = 0; basis + 8 <= dim; basis += 8) {
            const __m512d r = _mm512_loadu_pd(re + basis);
            const __m512d v = _mm512_loadu_pd(im + basis);
            const __m512d swapped_r = permute_pair_bit1(r, xmask);
            const __m512d swapped_v = permute_pair_bit1(v, xmask);
            _mm512_storeu_pd(re + basis, _mm512_fnmadd_pd(vq, swapped_v, _mm512_mul_pd(vc, r)));
            _mm512_storeu_pd(im + basis, _mm512_fmadd_pd(vq, swapped_r, _mm512_mul_pd(vc, v)));
        }
        return;
    }
    if (pair_bit >= 3 && dim >= 8) {
        const std::size_t lane_mask = lower_mask & std::size_t{7};
        const std::size_t chunk_mask = lower_mask & ~std::size_t{7};
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = permute_lanes_xor3(_mm512_loadu_pd(re + i1), lane_mask);
                const __m512d im1 = permute_lanes_xor3(_mm512_loadu_pd(im + i1), lane_mask);
                _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
                _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
                const __m512d out_r1 = _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1));
                const __m512d out_im1 = _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1));
                _mm512_storeu_pd(re + i1, permute_lanes_xor3(out_r1, lane_mask));
                _mm512_storeu_pd(im + i1, permute_lanes_xor3(out_im1, lane_mask));
            }
        }
        return;
    }
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 8 <= selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = _mm512_loadu_pd(re + i1);
                const __m512d im1 = _mm512_loadu_pd(im + i1);
                _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
                _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
                _mm512_storeu_pd(re + i1, _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1)));
                _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1)));
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 8) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 8 <= xor_run; local += 8) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m512d r0 = _mm512_loadu_pd(re + i0);
                        const __m512d im0 = _mm512_loadu_pd(im + i0);
                        const __m512d r1 = _mm512_loadu_pd(re + i1);
                        const __m512d im1 = _mm512_loadu_pd(im + i1);
                        _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
                        _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
                        _mm512_storeu_pd(re + i1, _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1)));
                        _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1)));
                    }
                }
                offset = selector;
            } else {
                for (; offset + 8 <= selector; offset += 8) {
                    const std::size_t i0 = block + offset;
                    const __m512d r0 = _mm512_loadu_pd(re + i0);
                    const __m512d im0 = _mm512_loadu_pd(im + i0);
                    const __m512d r1 = gather_pair8(re, i0, xmask);
                    const __m512d im1 = gather_pair8(im, i0, xmask);
                    _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(vq, im1, _mm512_mul_pd(vc, r0)));
                    _mm512_storeu_pd(im + i0, _mm512_fmadd_pd(vq, r1, _mm512_mul_pd(vc, im0)));
                    scatter_pair8(re, i0, xmask, _mm512_fnmadd_pd(vq, im0, _mm512_mul_pd(vc, r1)));
                    scatter_pair8(im, i0, xmask, _mm512_fmadd_pd(vq, r0, _mm512_mul_pd(vc, im1)));
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

void avx512_rotate_real_pair_flip_soa(
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
    const __m512d vc = _mm512_set1_pd(c);
    const __m512d vbase = _mm512_set1_pd(base_coeff);
    if (selector == 1 && xmask == 1 && dim >= 8) {
        for (std::size_t basis = 0; basis + 8 <= dim; basis += 8) {
            const std::size_t idx = basis >> 1;
            const double q0 = phase_signs[idx] * base_coeff;
            const double q1 = phase_signs[idx + 1] * base_coeff;
            const double q2 = phase_signs[idx + 2] * base_coeff;
            const double q3 = phase_signs[idx + 3] * base_coeff;
            const __m512d signed_q = _mm512_set_pd(q3, -q3, q2, -q2, q1, -q1, q0, -q0);
            const __m512d r = _mm512_loadu_pd(re + basis);
            const __m512d v = _mm512_loadu_pd(im + basis);
            const __m512d swapped_r = permute_pair_bit0(r);
            const __m512d swapped_v = permute_pair_bit0(v);
            _mm512_storeu_pd(re + basis, _mm512_fmadd_pd(signed_q, swapped_r, _mm512_mul_pd(vc, r)));
            _mm512_storeu_pd(im + basis, _mm512_fmadd_pd(signed_q, swapped_v, _mm512_mul_pd(vc, v)));
        }
        return;
    }
    if (pair_bit == 1 && (xmask == 2 || xmask == 3) && dim >= 8) {
        std::size_t pair_idx = 0;
        for (std::size_t basis = 0; basis + 8 <= dim; basis += 8) {
            const double q0 = phase_signs[pair_idx] * base_coeff;
            const double q1 = phase_signs[pair_idx + 1] * base_coeff;
            const double q2 = phase_signs[pair_idx + 2] * base_coeff;
            const double q3 = phase_signs[pair_idx + 3] * base_coeff;
            const __m512d signed_q = xmask == 2
                                          ? _mm512_set_pd(q3, q2, -q3, -q2, q1, q0, -q1, -q0)
                                          : _mm512_set_pd(q2, q3, -q3, -q2, q0, q1, -q1, -q0);
            const __m512d r = _mm512_loadu_pd(re + basis);
            const __m512d v = _mm512_loadu_pd(im + basis);
            const __m512d swapped_r = permute_pair_bit1(r, xmask);
            const __m512d swapped_v = permute_pair_bit1(v, xmask);
            _mm512_storeu_pd(re + basis, _mm512_fmadd_pd(signed_q, swapped_r, _mm512_mul_pd(vc, r)));
            _mm512_storeu_pd(im + basis, _mm512_fmadd_pd(signed_q, swapped_v, _mm512_mul_pd(vc, v)));
            pair_idx += 4;
        }
        return;
    }
    if (pair_bit >= 3 && dim >= 8) {
        const std::size_t lane_mask = lower_mask & std::size_t{7};
        const std::size_t chunk_mask = lower_mask & ~std::size_t{7};
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m512d q = _mm512_mul_pd(_mm512_loadu_pd(phase_signs + pair_idx), vbase);
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = permute_lanes_xor3(_mm512_loadu_pd(re + i1), lane_mask);
                const __m512d im1 = permute_lanes_xor3(_mm512_loadu_pd(im + i1), lane_mask);
                _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(q, r1, _mm512_mul_pd(vc, r0)));
                _mm512_storeu_pd(im + i0, _mm512_fnmadd_pd(q, im1, _mm512_mul_pd(vc, im0)));
                const __m512d out_r1 = _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1));
                const __m512d out_im1 = _mm512_fmadd_pd(q, im0, _mm512_mul_pd(vc, im1));
                _mm512_storeu_pd(re + i1, permute_lanes_xor3(out_r1, lane_mask));
                _mm512_storeu_pd(im + i1, permute_lanes_xor3(out_im1, lane_mask));
                pair_idx += 8;
            }
        }
        return;
    }
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 8 <= selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m512d q = _mm512_mul_pd(_mm512_loadu_pd(phase_signs + pair_idx), vbase);
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = _mm512_loadu_pd(re + i1);
                const __m512d im1 = _mm512_loadu_pd(im + i1);
                _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(q, r1, _mm512_mul_pd(vc, r0)));
                _mm512_storeu_pd(im + i0, _mm512_fnmadd_pd(q, im1, _mm512_mul_pd(vc, im0)));
                _mm512_storeu_pd(re + i1, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
                _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(q, im0, _mm512_mul_pd(vc, im1)));
                pair_idx += 8;
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 8) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 8 <= xor_run; local += 8) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m512d q = _mm512_mul_pd(_mm512_loadu_pd(phase_signs + pair_idx), vbase);
                        const __m512d r0 = _mm512_loadu_pd(re + i0);
                        const __m512d im0 = _mm512_loadu_pd(im + i0);
                        const __m512d r1 = _mm512_loadu_pd(re + i1);
                        const __m512d im1 = _mm512_loadu_pd(im + i1);
                        _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(q, r1, _mm512_mul_pd(vc, r0)));
                        _mm512_storeu_pd(im + i0, _mm512_fnmadd_pd(q, im1, _mm512_mul_pd(vc, im0)));
                        _mm512_storeu_pd(re + i1, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
                        _mm512_storeu_pd(im + i1, _mm512_fmadd_pd(q, im0, _mm512_mul_pd(vc, im1)));
                        pair_idx += 8;
                    }
                }
                offset = selector;
            } else {
                for (; offset + 8 <= selector; offset += 8) {
                    const std::size_t i0 = block + offset;
                    const __m512d q = _mm512_mul_pd(_mm512_loadu_pd(phase_signs + pair_idx), vbase);
                    const __m512d r0 = _mm512_loadu_pd(re + i0);
                    const __m512d im0 = _mm512_loadu_pd(im + i0);
                    const __m512d r1 = gather_pair8(re, i0, xmask);
                    const __m512d im1 = gather_pair8(im, i0, xmask);
                    _mm512_storeu_pd(re + i0, _mm512_fnmadd_pd(q, r1, _mm512_mul_pd(vc, r0)));
                    _mm512_storeu_pd(im + i0, _mm512_fnmadd_pd(q, im1, _mm512_mul_pd(vc, im0)));
                    scatter_pair8(re, i0, xmask, _mm512_fmadd_pd(q, r0, _mm512_mul_pd(vc, r1)));
                    scatter_pair8(im, i0, xmask, _mm512_fmadd_pd(q, im0, _mm512_mul_pd(vc, im1)));
                    pair_idx += 8;
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

void avx512_rotate_general_pairs_soa(
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
    const __m512d vc = _mm512_set1_pd(c);
    if (selector == 1 && xmask == 1 && dim >= 8) {
        for (std::size_t basis = 0; basis + 8 <= dim; basis += 8) {
            const std::size_t pair_idx = basis >> 1;
            const __m512d coeff_r = _mm512_set_pd(
                left[2 * (pair_idx + 3)],
                right[2 * (pair_idx + 3)],
                left[2 * (pair_idx + 2)],
                right[2 * (pair_idx + 2)],
                left[2 * (pair_idx + 1)],
                right[2 * (pair_idx + 1)],
                left[2 * pair_idx],
                right[2 * pair_idx]);
            const __m512d coeff_i = _mm512_set_pd(
                left[2 * (pair_idx + 3) + 1],
                right[2 * (pair_idx + 3) + 1],
                left[2 * (pair_idx + 2) + 1],
                right[2 * (pair_idx + 2) + 1],
                left[2 * (pair_idx + 1) + 1],
                right[2 * (pair_idx + 1) + 1],
                left[2 * pair_idx + 1],
                right[2 * pair_idx + 1]);
            const __m512d r = _mm512_loadu_pd(re + basis);
            const __m512d v = _mm512_loadu_pd(im + basis);
            const __m512d swapped_r = permute_pair_bit0(r);
            const __m512d swapped_v = permute_pair_bit0(v);
            __m512d out_r = _mm512_fmadd_pd(vc, r, _mm512_mul_pd(coeff_r, swapped_r));
            out_r = _mm512_fnmadd_pd(coeff_i, swapped_v, out_r);
            __m512d out_i = _mm512_fmadd_pd(vc, v, _mm512_mul_pd(coeff_r, swapped_v));
            out_i = _mm512_fmadd_pd(coeff_i, swapped_r, out_i);
            _mm512_storeu_pd(re + basis, out_r);
            _mm512_storeu_pd(im + basis, out_i);
        }
        return;
    }
    if (pair_bit == 1 && (xmask == 2 || xmask == 3) && dim >= 8) {
        std::size_t pair_idx = 0;
        for (std::size_t basis = 0; basis + 8 <= dim; basis += 8) {
            const __m512d coeff_r = xmask == 2
                                        ? _mm512_set_pd(
                                              left[2 * (pair_idx + 3)],
                                              left[2 * (pair_idx + 2)],
                                              right[2 * (pair_idx + 3)],
                                              right[2 * (pair_idx + 2)],
                                              left[2 * (pair_idx + 1)],
                                              left[2 * pair_idx],
                                              right[2 * (pair_idx + 1)],
                                              right[2 * pair_idx])
                                        : _mm512_set_pd(
                                              left[2 * (pair_idx + 2)],
                                              left[2 * (pair_idx + 3)],
                                              right[2 * (pair_idx + 3)],
                                              right[2 * (pair_idx + 2)],
                                              left[2 * pair_idx],
                                              left[2 * (pair_idx + 1)],
                                              right[2 * (pair_idx + 1)],
                                              right[2 * pair_idx]);
            const __m512d coeff_i = xmask == 2
                                        ? _mm512_set_pd(
                                              left[2 * (pair_idx + 3) + 1],
                                              left[2 * (pair_idx + 2) + 1],
                                              right[2 * (pair_idx + 3) + 1],
                                              right[2 * (pair_idx + 2) + 1],
                                              left[2 * (pair_idx + 1) + 1],
                                              left[2 * pair_idx + 1],
                                              right[2 * (pair_idx + 1) + 1],
                                              right[2 * pair_idx + 1])
                                        : _mm512_set_pd(
                                              left[2 * (pair_idx + 2) + 1],
                                              left[2 * (pair_idx + 3) + 1],
                                              right[2 * (pair_idx + 3) + 1],
                                              right[2 * (pair_idx + 2) + 1],
                                              left[2 * pair_idx + 1],
                                              left[2 * (pair_idx + 1) + 1],
                                              right[2 * (pair_idx + 1) + 1],
                                              right[2 * pair_idx + 1]);
            const __m512d r = _mm512_loadu_pd(re + basis);
            const __m512d v = _mm512_loadu_pd(im + basis);
            const __m512d swapped_r = permute_pair_bit1(r, xmask);
            const __m512d swapped_v = permute_pair_bit1(v, xmask);
            __m512d out_r = _mm512_fmadd_pd(vc, r, _mm512_mul_pd(coeff_r, swapped_r));
            out_r = _mm512_fnmadd_pd(coeff_i, swapped_v, out_r);
            __m512d out_i = _mm512_fmadd_pd(vc, v, _mm512_mul_pd(coeff_r, swapped_v));
            out_i = _mm512_fmadd_pd(coeff_i, swapped_r, out_i);
            _mm512_storeu_pd(re + basis, out_r);
            _mm512_storeu_pd(im + basis, out_i);
            pair_idx += 4;
        }
        return;
    }
    if (pair_bit >= 3 && dim >= 8) {
        const std::size_t lane_mask = lower_mask & std::size_t{7};
        const std::size_t chunk_mask = lower_mask & ~std::size_t{7};
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = permute_lanes_xor3(_mm512_loadu_pd(re + i1), lane_mask);
                const __m512d im1 = permute_lanes_xor3(_mm512_loadu_pd(im + i1), lane_mask);
                const __m512d lr = load_complex_re8(left, pair_idx);
                const __m512d li = load_complex_im8(left, pair_idx);
                const __m512d rr = load_complex_re8(right, pair_idx);
                const __m512d ri = load_complex_im8(right, pair_idx);
                __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                _mm512_storeu_pd(re + i0, out_r0);
                _mm512_storeu_pd(im + i0, out_i0);
                _mm512_storeu_pd(re + i1, permute_lanes_xor3(out_r1, lane_mask));
                _mm512_storeu_pd(im + i1, permute_lanes_xor3(out_i1, lane_mask));
                pair_idx += 8;
            }
        }
        return;
    }
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 8 <= selector; offset += 8) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m512d r0 = _mm512_loadu_pd(re + i0);
                const __m512d im0 = _mm512_loadu_pd(im + i0);
                const __m512d r1 = _mm512_loadu_pd(re + i1);
                const __m512d im1 = _mm512_loadu_pd(im + i1);
                const __m512d lr = load_complex_re8(left, pair_idx);
                const __m512d li = load_complex_im8(left, pair_idx);
                const __m512d rr = load_complex_re8(right, pair_idx);
                const __m512d ri = load_complex_im8(right, pair_idx);
                __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                _mm512_storeu_pd(re + i0, out_r0);
                _mm512_storeu_pd(im + i0, out_i0);
                _mm512_storeu_pd(re + i1, out_r1);
                _mm512_storeu_pd(im + i1, out_i1);
                pair_idx += 8;
            }
        }
        if (!contiguous_partner) {
            if (xor_run >= 8) {
                for (std::size_t segment = 0; segment < selector; segment += xor_run) {
                    const std::size_t right_base = block + selector + (segment ^ lower_mask);
                    for (std::size_t local = 0; local + 8 <= xor_run; local += 8) {
                        const std::size_t i0 = block + segment + local;
                        const std::size_t i1 = right_base + local;
                        const __m512d r0 = _mm512_loadu_pd(re + i0);
                        const __m512d im0 = _mm512_loadu_pd(im + i0);
                        const __m512d r1 = _mm512_loadu_pd(re + i1);
                        const __m512d im1 = _mm512_loadu_pd(im + i1);
                        const __m512d lr = load_complex_re8(left, pair_idx);
                        const __m512d li = load_complex_im8(left, pair_idx);
                        const __m512d rr = load_complex_re8(right, pair_idx);
                        const __m512d ri = load_complex_im8(right, pair_idx);
                        __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                        out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                        __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                        out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                        __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                        out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                        __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                        out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                        _mm512_storeu_pd(re + i0, out_r0);
                        _mm512_storeu_pd(im + i0, out_i0);
                        _mm512_storeu_pd(re + i1, out_r1);
                        _mm512_storeu_pd(im + i1, out_i1);
                        pair_idx += 8;
                    }
                }
                offset = selector;
            } else {
                for (; offset + 8 <= selector; offset += 8) {
                    const std::size_t i0 = block + offset;
                    const __m512d r0 = _mm512_loadu_pd(re + i0);
                    const __m512d im0 = _mm512_loadu_pd(im + i0);
                    const __m512d r1 = gather_pair8(re, i0, xmask);
                    const __m512d im1 = gather_pair8(im, i0, xmask);
                    const __m512d lr = load_complex_re8(left, pair_idx);
                    const __m512d li = load_complex_im8(left, pair_idx);
                    const __m512d rr = load_complex_re8(right, pair_idx);
                    const __m512d ri = load_complex_im8(right, pair_idx);
                    __m512d out_r0 = _mm512_fmadd_pd(vc, r0, _mm512_mul_pd(rr, r1));
                    out_r0 = _mm512_fnmadd_pd(ri, im1, out_r0);
                    __m512d out_i0 = _mm512_fmadd_pd(vc, im0, _mm512_mul_pd(rr, im1));
                    out_i0 = _mm512_fmadd_pd(ri, r1, out_i0);
                    __m512d out_r1 = _mm512_fmadd_pd(vc, r1, _mm512_mul_pd(lr, r0));
                    out_r1 = _mm512_fnmadd_pd(li, im0, out_r1);
                    __m512d out_i1 = _mm512_fmadd_pd(vc, im1, _mm512_mul_pd(lr, im0));
                    out_i1 = _mm512_fmadd_pd(li, r0, out_i1);
                    _mm512_storeu_pd(re + i0, out_r0);
                    _mm512_storeu_pd(im + i0, out_i0);
                    scatter_pair8(re, i0, xmask, out_r1);
                    scatter_pair8(im, i0, xmask, out_i1);
                    pair_idx += 8;
                }
            }
        }
        SYMFT_SIMD_LOOP
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
    "avx512",
    avx512_mul_assign,
    avx512_norm_sum,
    avx512_mul_assign_soa,
    avx512_norm_sum_soa,
    avx512_measure_nondiagonal_probability_soa,
    avx512_project_nondiagonal_soa,
    avx512_rotate_uniform_imag_pairs_soa,
    avx512_rotate_real_pair_flip_soa,
    avx512_rotate_general_pairs_soa,
};

} // namespace

const KernelTable& avx512_table() {
    return table;
}

} // namespace symft::simd
