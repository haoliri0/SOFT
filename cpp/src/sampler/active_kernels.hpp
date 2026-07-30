#pragma once

#include "sampler/active_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

namespace symft::detail {

#if defined(__AVX2__) && defined(__FMA__)
#define SYMFT_INLINE_AVX2 1
#else
#define SYMFT_INLINE_AVX2 0
#endif


inline void rotate_uniform_imag_pairs_scalar(
    Complex* alpha,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    double q) {
    auto* raw = reinterpret_cast<double*>(alpha);
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = insert_zero_bit(idx, static_cast<int>(pair_bit));
        const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
        double* a0 = raw + 2 * i0;
        double* a1 = raw + 2 * i1;
        const double r0 = a0[0];
        const double i0v = a0[1];
        const double r1 = a1[0];
        const double i1v = a1[1];
        a0[0] = c * r0 - q * i1v;
        a0[1] = c * i0v + q * r1;
        a1[0] = c * r1 - q * i0v;
        a1[1] = c * i1v + q * r0;
    }
}

inline void rotate_real_pair_flip_scalar(
    Complex* alpha,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* phase_signs,
    std::size_t npairs,
    double c,
    double base_coeff) {
    auto* raw = reinterpret_cast<double*>(alpha);
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = insert_zero_bit(idx, static_cast<int>(pair_bit));
        const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
        const double q = phase_signs[idx] * base_coeff;
        double* a0 = raw + 2 * i0;
        double* a1 = raw + 2 * i1;
        const double r0 = a0[0];
        const double i0v = a0[1];
        const double r1 = a1[0];
        const double i1v = a1[1];
        a0[0] = c * r0 - q * r1;
        a0[1] = c * i0v - q * i1v;
        a1[0] = c * r1 + q * r0;
        a1[1] = c * i1v + q * i0v;
    }
}

#if SYMFT_INLINE_AVX2
inline __m256d load_complex_re4_inline(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpacklo_pd(c0, c1), 0xd8);
}

inline __m256d load_complex_im4_inline(const double* coeff) {
    const __m256d c0 = _mm256_loadu_pd(coeff);
    const __m256d c1 = _mm256_loadu_pd(coeff + 4);
    return _mm256_permute4x64_pd(_mm256_unpackhi_pd(c0, c1), 0xd8);
}

inline __m256i pair_indices4_inline(std::size_t i0, std::uint64_t xmask) {
    const auto mask = static_cast<std::size_t>(xmask);
    return _mm256_set_epi64x(
        static_cast<long long>((i0 + 3) ^ mask),
        static_cast<long long>((i0 + 2) ^ mask),
        static_cast<long long>((i0 + 1) ^ mask),
        static_cast<long long>(i0 ^ mask));
}

inline __m256d gather_pair4_inline(const double* values, std::size_t i0, std::uint64_t xmask) {
    return _mm256_i64gather_pd(values, pair_indices4_inline(i0, xmask), 8);
}

inline void scatter_pair4_inline(double* values, std::size_t i0, std::uint64_t xmask, __m256d lanes) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, lanes);
    const auto mask = static_cast<std::size_t>(xmask);
    values[i0 ^ mask] = tmp[0];
    values[(i0 + 1) ^ mask] = tmp[1];
    values[(i0 + 2) ^ mask] = tmp[2];
    values[(i0 + 3) ^ mask] = tmp[3];
}

inline __m256d permute_pair_bit1_inline(__m256d lanes, std::uint64_t xmask) {
    return xmask == 2 ? _mm256_permute4x64_pd(lanes, 0x4e) : _mm256_permute4x64_pd(lanes, 0x1b);
}

inline __m256d permute_lanes_xor2_inline(__m256d lanes, std::size_t mask) {
    switch (mask & std::size_t{3}) {
    case 0:
        return lanes;
    case 1:
        return _mm256_permute_pd(lanes, 0b0101);
    case 2:
        return _mm256_permute4x64_pd(lanes, 0x4e);
    default:
        return _mm256_permute4x64_pd(lanes, 0x1b);
    }
}
#endif

inline std::size_t xor_contiguous_run(std::size_t lower_mask, std::size_t selector) {
    return lower_mask == 0 ? selector : (lower_mask & (~lower_mask + std::size_t{1}));
}

inline void mul_assign_soa_inline(double* re, double* im, const Complex* coefficients, double c, std::size_t dim) {
    const auto* coeff = reinterpret_cast<const double*>(coefficients);
    std::size_t basis = 0;
#if SYMFT_INLINE_AVX2
    const __m256d vc = _mm256_set1_pd(c);
    for (; basis + 4 <= dim; basis += 4) {
        const __m256d r = _mm256_loadu_pd(re + basis);
        const __m256d i = _mm256_loadu_pd(im + basis);
        const __m256d fr = _mm256_add_pd(vc, load_complex_re4_inline(coeff + 2 * basis));
        const __m256d fi = load_complex_im4_inline(coeff + 2 * basis);
        _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(fi, i, _mm256_mul_pd(fr, r)));
        _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(fi, r, _mm256_mul_pd(fr, i)));
    }
#endif
    SYMFT_SINGLE_SIMD_LOOP
    for (; basis < dim; ++basis) {
        const double fr = c + coeff[2 * basis];
        const double fi = coeff[2 * basis + 1];
        const double r = re[basis];
        const double i = im[basis];
        re[basis] = fr * r - fi * i;
        im[basis] = fr * i + fi * r;
    }
}

inline double norm_sum_soa_inline(const double* re, const double* im, const std::size_t* sources, std::size_t n) {
    double probability = 0.0;
    SYMFT_SINGLE_SIMD_LOOP
    for (std::size_t idx = 0; idx < n; ++idx) {
        const std::size_t source = sources[idx];
        const double r = re[source];
        const double i = im[source];
        probability += r * r + i * i;
    }
    return probability;
}

inline void rotate_uniform_imag_pairs_soa_inline(
    double* re,
    double* im,
    std::size_t dim,
    std::uint64_t xmask,
    unsigned pair_bit,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t step = selector << 1;
    [[maybe_unused]] const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    [[maybe_unused]] const bool contiguous_partner = lower_mask == 0;
    [[maybe_unused]] const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
#if SYMFT_INLINE_AVX2
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
            const __m256d swapped_r = permute_pair_bit1_inline(r, xmask);
            const __m256d swapped_v = permute_pair_bit1_inline(v, xmask);
            _mm256_storeu_pd(re + basis, _mm256_fnmadd_pd(vq, swapped_v, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(vq, swapped_r, _mm256_mul_pd(vc, v)));
        }
        return;
    }
    if (pair_bit >= 2 && dim >= 4) {
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = permute_lanes_xor2_inline(_mm256_loadu_pd(re + i1), lane_mask);
                const __m256d im1 = permute_lanes_xor2_inline(_mm256_loadu_pd(im + i1), lane_mask);
                _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                const __m256d out_r1 = _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1));
                const __m256d out_im1 = _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1));
                _mm256_storeu_pd(re + i1, permute_lanes_xor2_inline(out_r1, lane_mask));
                _mm256_storeu_pd(im + i1, permute_lanes_xor2_inline(out_im1, lane_mask));
            }
        }
        return;
    }
#endif
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
#if SYMFT_INLINE_AVX2
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
                    const __m256d r1 = gather_pair4_inline(re, i0, xmask);
                    const __m256d im1 = gather_pair4_inline(im, i0, xmask);
                    _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(vq, im1, _mm256_mul_pd(vc, r0)));
                    _mm256_storeu_pd(im + i0, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, im0)));
                    scatter_pair4_inline(re, i0, xmask, _mm256_fnmadd_pd(vq, im0, _mm256_mul_pd(vc, r1)));
                    scatter_pair4_inline(im, i0, xmask, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, im1)));
                }
            }
        }
#endif
        SYMFT_SINGLE_SIMD_LOOP
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

inline void rotate_real_pair_flip_soa_inline(
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
    [[maybe_unused]] const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    [[maybe_unused]] const bool contiguous_partner = lower_mask == 0;
    [[maybe_unused]] const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
#if SYMFT_INLINE_AVX2
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
            const __m256d swapped_r = permute_pair_bit1_inline(r, xmask);
            const __m256d swapped_v = permute_pair_bit1_inline(v, xmask);
            _mm256_storeu_pd(re + basis, _mm256_fmadd_pd(signed_q, swapped_r, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(im + basis, _mm256_fmadd_pd(signed_q, swapped_v, _mm256_mul_pd(vc, v)));
            pair_idx += 2;
        }
        return;
    }
    if (pair_bit >= 2 && dim >= 4) {
        const std::size_t lane_mask = lower_mask & std::size_t{3};
        const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m256d q = _mm256_mul_pd(_mm256_loadu_pd(phase_signs + pair_idx), vbase);
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = permute_lanes_xor2_inline(_mm256_loadu_pd(re + i1), lane_mask);
                const __m256d im1 = permute_lanes_xor2_inline(_mm256_loadu_pd(im + i1), lane_mask);
                _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                const __m256d out_r1 = _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1));
                const __m256d out_im1 = _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1));
                _mm256_storeu_pd(re + i1, permute_lanes_xor2_inline(out_r1, lane_mask));
                _mm256_storeu_pd(im + i1, permute_lanes_xor2_inline(out_im1, lane_mask));
                pair_idx += 4;
            }
        }
        return;
    }
#endif
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
#if SYMFT_INLINE_AVX2
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
                    const __m256d r1 = gather_pair4_inline(re, i0, xmask);
                    const __m256d im1 = gather_pair4_inline(im, i0, xmask);
                    _mm256_storeu_pd(re + i0, _mm256_fnmadd_pd(q, r1, _mm256_mul_pd(vc, r0)));
                    _mm256_storeu_pd(im + i0, _mm256_fnmadd_pd(q, im1, _mm256_mul_pd(vc, im0)));
                    scatter_pair4_inline(re, i0, xmask, _mm256_fmadd_pd(q, r0, _mm256_mul_pd(vc, r1)));
                    scatter_pair4_inline(im, i0, xmask, _mm256_fmadd_pd(q, im0, _mm256_mul_pd(vc, im1)));
                    pair_idx += 4;
                }
            }
        }
#endif
        SYMFT_SINGLE_SIMD_LOOP
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

inline void rotate_general_pairs_soa_inline(
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
    [[maybe_unused]] const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    [[maybe_unused]] const bool contiguous_partner = lower_mask == 0;
    [[maybe_unused]] const std::size_t xor_run = xor_contiguous_run(lower_mask, selector);
#if SYMFT_INLINE_AVX2
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
            const __m256d swapped_r = permute_pair_bit1_inline(r, xmask);
            const __m256d swapped_v = permute_pair_bit1_inline(v, xmask);
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
        const std::size_t chunk_mask = lower_mask & ~std::size_t{3};
        std::size_t pair_idx = 0;
        for (std::size_t block = 0; block < dim; block += step) {
            for (std::size_t offset = 0; offset < selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = block + selector + (offset ^ chunk_mask);
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = permute_lanes_xor2_inline(_mm256_loadu_pd(re + i1), lane_mask);
                const __m256d im1 = permute_lanes_xor2_inline(_mm256_loadu_pd(im + i1), lane_mask);
                const __m256d lr = load_complex_re4_inline(left + 2 * pair_idx);
                const __m256d li = load_complex_im4_inline(left + 2 * pair_idx);
                const __m256d rr = load_complex_re4_inline(right + 2 * pair_idx);
                const __m256d ri = load_complex_im4_inline(right + 2 * pair_idx);
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
                _mm256_storeu_pd(re + i1, permute_lanes_xor2_inline(out_r1, lane_mask));
                _mm256_storeu_pd(im + i1, permute_lanes_xor2_inline(out_i1, lane_mask));
                pair_idx += 4;
            }
        }
        return;
    }
#endif
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        std::size_t offset = 0;
#if SYMFT_INLINE_AVX2
        if (contiguous_partner) {
            const std::size_t right_base = block ^ static_cast<std::size_t>(xmask);
            for (; offset + 4 <= selector; offset += 4) {
                const std::size_t i0 = block + offset;
                const std::size_t i1 = right_base + offset;
                const __m256d r0 = _mm256_loadu_pd(re + i0);
                const __m256d im0 = _mm256_loadu_pd(im + i0);
                const __m256d r1 = _mm256_loadu_pd(re + i1);
                const __m256d im1 = _mm256_loadu_pd(im + i1);
                const __m256d lr = load_complex_re4_inline(left + 2 * pair_idx);
                const __m256d li = load_complex_im4_inline(left + 2 * pair_idx);
                const __m256d rr = load_complex_re4_inline(right + 2 * pair_idx);
                const __m256d ri = load_complex_im4_inline(right + 2 * pair_idx);
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
                        const __m256d lr = load_complex_re4_inline(left + 2 * pair_idx);
                        const __m256d li = load_complex_im4_inline(left + 2 * pair_idx);
                        const __m256d rr = load_complex_re4_inline(right + 2 * pair_idx);
                        const __m256d ri = load_complex_im4_inline(right + 2 * pair_idx);
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
                    const __m256d r1 = gather_pair4_inline(re, i0, xmask);
                    const __m256d im1 = gather_pair4_inline(im, i0, xmask);
                    const __m256d lr = load_complex_re4_inline(left + 2 * pair_idx);
                    const __m256d li = load_complex_im4_inline(left + 2 * pair_idx);
                    const __m256d rr = load_complex_re4_inline(right + 2 * pair_idx);
                    const __m256d ri = load_complex_im4_inline(right + 2 * pair_idx);
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
                    scatter_pair4_inline(re, i0, xmask, out_r1);
                    scatter_pair4_inline(im, i0, xmask, out_i1);
                    pair_idx += 4;
                }
            }
        }
#endif
        SYMFT_SINGLE_SIMD_LOOP
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

} // namespace symft::detail

#undef SYMFT_INLINE_AVX2
