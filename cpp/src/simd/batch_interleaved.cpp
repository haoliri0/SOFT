#include "simd/batch_interleaved.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace symft::batch_interleaved {
namespace {

constexpr double kInvSqrt2 = 0.70710678118654752440;

#if defined(__clang__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BATCH_SIMD_LOOP
#endif

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return ~std::uint64_t{0};
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int active_shots, std::size_t word) {
    return low_bits_mask(active_shots - static_cast<int>(word << 6));
}

double select_double_bits(double false_value, double true_value, std::uint64_t branch_mask) {
    const std::uint64_t false_bits = std::bit_cast<std::uint64_t>(false_value);
    const std::uint64_t true_bits = std::bit_cast<std::uint64_t>(true_value);
    return std::bit_cast<double>((~branch_mask & false_bits) ^ (branch_mask & true_bits));
}

std::size_t insert_zero_bit(std::size_t packed, unsigned bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - std::size_t{1};
    const std::size_t low = packed & low_mask;
    const std::size_t high = packed & ~low_mask;
    return low | (high << 1);
}

void rotate_uniform_imag_pair_const(
    double* r0p,
    double* i0p,
    double* r1p,
    double* i1p,
    int active_shots,
    double c,
    double q) {
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = 0; shot < active_shots; ++shot) {
        const double r0 = r0p[shot];
        const double i0 = i0p[shot];
        const double r1 = r1p[shot];
        const double i1 = i1p[shot];
        r0p[shot] = c * r0 - q * i1;
        i0p[shot] = c * i0 + q * r1;
        r1p[shot] = c * r1 - q * i0;
        i1p[shot] = c * i1 + q * r0;
    }
}

void rotate_real_pair_flip_const(
    double* r0p,
    double* i0p,
    double* r1p,
    double* i1p,
    int active_shots,
    double c,
    double q) {
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = 0; shot < active_shots; ++shot) {
        const double r0 = r0p[shot];
        const double i0 = i0p[shot];
        const double r1 = r1p[shot];
        const double i1 = i1p[shot];
        r0p[shot] = c * r0 - q * r1;
        i0p[shot] = c * i0 - q * i1;
        r1p[shot] = c * r1 + q * r0;
        i1p[shot] = c * i1 + q * i0;
    }
}

void rotate_general_pair_const(
    double* r0p,
    double* i0p,
    double* r1p,
    double* i1p,
    int first_shot,
    int last_shot,
    double c,
    Complex left_coeff,
    Complex right_coeff) {
    const double lr = left_coeff.real();
    const double li = left_coeff.imag();
    const double rr = right_coeff.real();
    const double ri = right_coeff.imag();
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = first_shot; shot < last_shot; ++shot) {
        const double r0 = r0p[shot];
        const double im0 = i0p[shot];
        const double r1 = r1p[shot];
        const double im1 = i1p[shot];
        r0p[shot] = c * r0 + rr * r1 - ri * im1;
        i0p[shot] = c * im0 + rr * im1 + ri * r1;
        r1p[shot] = c * r1 + lr * r0 - li * im0;
        i1p[shot] = c * im1 + lr * im0 + li * r0;
    }
}

#if defined(__AVX2__) && defined(__FMA__)
__m256d pitch2_coeff_lanes(double shot0, double shot1) {
    return _mm256_set_pd(shot1, shot0, shot1, shot0);
}

__m256d pitch2_pair_lanes(double row0, double row1) {
    return _mm256_set_pd(row1, row1, row0, row0);
}

__m256d maybe_swap_pitch2_basis_rows(__m256d lanes, bool swap) {
    return swap ? _mm256_permute4x64_pd(lanes, 0x4e) : lanes;
}

void store_pitch2_partner(double* ptr, __m256d lanes, bool swapped) {
    _mm256_storeu_pd(ptr, maybe_swap_pitch2_basis_rows(lanes, swapped));
}

bool rotate_uniform_imag_xmask_pitch2_avx2(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    if (leading_shots != 2 || active_shots != 2) {
        return false;
    }
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const __m256d vc = _mm256_set1_pd(c);
    const __m256d vq = pitch2_coeff_lanes(q_by_shot[0], q_by_shot[1]);
    if (selector == 1) {
        for (std::size_t block = 0; block < dim; block += 2) {
            double* rp = re + block * 2;
            double* ip = im + block * 2;
            const __m256d r = _mm256_loadu_pd(rp);
            const __m256d v = _mm256_loadu_pd(ip);
            const __m256d partner_r = _mm256_permute4x64_pd(r, 0x4e);
            const __m256d partner_i = _mm256_permute4x64_pd(v, 0x4e);
            _mm256_storeu_pd(rp, _mm256_fnmadd_pd(vq, partner_i, _mm256_mul_pd(vc, r)));
            _mm256_storeu_pd(ip, _mm256_fmadd_pd(vq, partner_r, _mm256_mul_pd(vc, v)));
        }
        return true;
    }
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 2) {
            const std::size_t low = block + offset;
            const std::size_t high0 = block + selector + (offset ^ lower_mask);
            const std::size_t high1 = block + selector + ((offset + 1) ^ lower_mask);
            const bool swapped = high0 > high1;
            const std::size_t high = swapped ? high1 : high0;
            double* r0p = re + low * 2;
            double* i0p = im + low * 2;
            double* r1p = re + high * 2;
            double* i1p = im + high * 2;
            const __m256d r0 = _mm256_loadu_pd(r0p);
            const __m256d i0 = _mm256_loadu_pd(i0p);
            const __m256d r1 = maybe_swap_pitch2_basis_rows(_mm256_loadu_pd(r1p), swapped);
            const __m256d i1 = maybe_swap_pitch2_basis_rows(_mm256_loadu_pd(i1p), swapped);
            _mm256_storeu_pd(r0p, _mm256_fnmadd_pd(vq, i1, _mm256_mul_pd(vc, r0)));
            _mm256_storeu_pd(i0p, _mm256_fmadd_pd(vq, r1, _mm256_mul_pd(vc, i0)));
            store_pitch2_partner(r1p, _mm256_fnmadd_pd(vq, i0, _mm256_mul_pd(vc, r1)), swapped);
            store_pitch2_partner(i1p, _mm256_fmadd_pd(vq, r0, _mm256_mul_pd(vc, i1)), swapped);
        }
    }
    return true;
}

bool rotate_uniform_imag_xmask_const_pitch2_avx2(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    double q) {
    const double q_by_shot[2] = {q, q};
    return rotate_uniform_imag_xmask_pitch2_avx2(
        re,
        im,
        leading_shots,
        active_shots,
        xmask,
        pair_bit,
        npairs,
        c,
        q_by_shot);
}

bool nondiagonal_xmask_measure_true_prob_pitch2_avx2(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    double* prob_true) {
    if (leading_shots != 2 || active_shots != 2 || pair_bit == 0) {
        return false;
    }
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const __m256d vinv_sqrt2 = _mm256_set1_pd(kInvSqrt2);
    __m256d sum = _mm256_setzero_pd();
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 2) {
            const std::size_t source0 = block + offset;
            const std::size_t source1_0 = block + selector + (offset ^ lower_mask);
            const std::size_t source1_1 = block + selector + ((offset + 1) ^ lower_mask);
            const bool swapped = source1_0 > source1_1;
            const std::size_t source1 = swapped ? source1_1 : source1_0;
            const __m256d r0 = _mm256_loadu_pd(re + source0 * 2);
            const __m256d i0 = _mm256_loadu_pd(im + source0 * 2);
            const __m256d r1 = maybe_swap_pitch2_basis_rows(_mm256_loadu_pd(re + source1 * 2), swapped);
            const __m256d i1 = maybe_swap_pitch2_basis_rows(_mm256_loadu_pd(im + source1 * 2), swapped);
            const __m256d c1r = pitch2_pair_lanes(coeff1_false_real[pair_idx], coeff1_false_real[pair_idx + 1]);
            const __m256d c1i = pitch2_pair_lanes(coeff1_false_imag[pair_idx], coeff1_false_imag[pair_idx + 1]);
            const __m256d prod_r = _mm256_fnmadd_pd(c1i, i1, _mm256_mul_pd(c1r, r1));
            const __m256d prod_i = _mm256_fmadd_pd(c1i, r1, _mm256_mul_pd(c1r, i1));
            const __m256d amp_r = _mm256_sub_pd(_mm256_mul_pd(vinv_sqrt2, r0), prod_r);
            const __m256d amp_i = _mm256_sub_pd(_mm256_mul_pd(vinv_sqrt2, i0), prod_i);
            sum = _mm256_fmadd_pd(amp_r, amp_r, _mm256_fmadd_pd(amp_i, amp_i, sum));
            pair_idx += 2;
        }
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, sum);
    prob_true[0] = lanes[0] + lanes[2];
    prob_true[1] = lanes[1] + lanes[3];
    return true;
}

bool nondiagonal_xmask_project_pitch2_avx2(
    const double* re,
    const double* im,
    double* scratch_re,
    double* scratch_im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    if (leading_shots != 2 || active_shots != 2 || pair_bit == 0) {
        return false;
    }
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    const std::size_t lower_mask = static_cast<std::size_t>(xmask) & (selector - 1);
    const std::uint64_t bits = branch_bits[0] & 0x3ULL;
    const double sign0 = (bits & 0x1ULL) != 0 ? -1.0 : 1.0;
    const double sign1 = (bits & 0x2ULL) != 0 ? -1.0 : 1.0;
    const __m256d vinv_sqrt2 = _mm256_set1_pd(kInvSqrt2);
    const __m256d branch_sign = pitch2_coeff_lanes(sign0, sign1);
    const __m256d invnorm = pitch2_coeff_lanes(invnorms[0], invnorms[1]);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; offset += 2) {
            const std::size_t source0 = block + offset;
            const std::size_t source1_0 = block + selector + (offset ^ lower_mask);
            const std::size_t source1_1 = block + selector + ((offset + 1) ^ lower_mask);
            const bool swapped = source1_0 > source1_1;
            const std::size_t source1 = swapped ? source1_1 : source1_0;
            const __m256d r0 = _mm256_loadu_pd(re + source0 * 2);
            const __m256d i0 = _mm256_loadu_pd(im + source0 * 2);
            const __m256d r1 = maybe_swap_pitch2_basis_rows(_mm256_loadu_pd(re + source1 * 2), swapped);
            const __m256d i1 = maybe_swap_pitch2_basis_rows(_mm256_loadu_pd(im + source1 * 2), swapped);
            const __m256d c1r = pitch2_pair_lanes(coeff1_false_real[pair_idx], coeff1_false_real[pair_idx + 1]);
            const __m256d c1i = pitch2_pair_lanes(coeff1_false_imag[pair_idx], coeff1_false_imag[pair_idx + 1]);
            const __m256d prod_r = _mm256_fnmadd_pd(c1i, i1, _mm256_mul_pd(c1r, r1));
            const __m256d prod_i = _mm256_fmadd_pd(c1i, r1, _mm256_mul_pd(c1r, i1));
            const __m256d amp_r = _mm256_fmadd_pd(branch_sign, prod_r, _mm256_mul_pd(vinv_sqrt2, r0));
            const __m256d amp_i = _mm256_fmadd_pd(branch_sign, prod_i, _mm256_mul_pd(vinv_sqrt2, i0));
            _mm256_storeu_pd(scratch_re + pair_idx * 2, _mm256_mul_pd(amp_r, invnorm));
            _mm256_storeu_pd(scratch_im + pair_idx * 2, _mm256_mul_pd(amp_i, invnorm));
            pair_idx += 2;
        }
    }
    return true;
}
#endif

void scalar_rotate_uniform_imag_pairs(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = insert_zero_bit(idx, pair_bit);
        const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
        double* r0p = re + i0 * leading_shots;
        double* i0p = im + i0 * leading_shots;
        double* r1p = re + i1 * leading_shots;
        double* i1p = im + i1 * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = q_by_shot[static_cast<std::size_t>(shot)];
            const double r0 = r0p[shot];
            const double i0v = i0p[shot];
            const double r1 = r1p[shot];
            const double i1v = i1p[shot];
            r0p[shot] = c * r0 - q * i1v;
            i0p[shot] = c * i0v + q * r1;
            r1p[shot] = c * r1 - q * i0v;
            i1p[shot] = c * i1v + q * r0;
        }
    }
}

void scalar_rotate_uniform_imag_xmask(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
#if defined(__AVX2__) && defined(__FMA__)
    if (rotate_uniform_imag_xmask_pitch2_avx2(
            re, im, leading_shots, active_shots, xmask, pair_bit, npairs, c, q_by_shot)) {
        return;
    }
#endif
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            double* r0p = re + i0 * leading_shots;
            double* i0p = im + i0 * leading_shots;
            double* r1p = re + i1 * leading_shots;
            double* i1p = im + i1 * leading_shots;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < active_shots; ++shot) {
                const double q = q_by_shot[static_cast<std::size_t>(shot)];
                const double r0 = r0p[shot];
                const double i0v = i0p[shot];
                const double r1 = r1p[shot];
                const double i1v = i1p[shot];
                r0p[shot] = c * r0 - q * i1v;
                i0p[shot] = c * i0v + q * r1;
                r1p[shot] = c * r1 - q * i0v;
                i1p[shot] = c * i1v + q * r0;
            }
        }
    }
}

void scalar_rotate_uniform_imag_xmask_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    double c,
    double q) {
#if defined(__AVX2__) && defined(__FMA__)
    if (rotate_uniform_imag_xmask_const_pitch2_avx2(
            re, im, leading_shots, active_shots, xmask, pair_bit, npairs, c, q)) {
        return;
    }
#endif
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            rotate_uniform_imag_pair_const(
                re + i0 * leading_shots,
                im + i0 * leading_shots,
                re + i1 * leading_shots,
                im + i1 * leading_shots,
                active_shots,
                c,
                q);
        }
    }
}

void scalar_rotate_real_pair_flip(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    for (std::size_t idx = 0; idx < npairs; ++idx) {
        const std::size_t i0 = insert_zero_bit(idx, pair_bit);
        const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
        const double basis_phase_sign = basis_phase_signs[idx];
        double* r0p = re + i0 * leading_shots;
        double* i0p = im + i0 * leading_shots;
        double* r1p = re + i1 * leading_shots;
        double* i1p = im + i1 * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = basis_phase_sign * q_by_shot[static_cast<std::size_t>(shot)];
            const double r0 = r0p[shot];
            const double i0v = i0p[shot];
            const double r1 = r1p[shot];
            const double i1v = i1p[shot];
            r0p[shot] = c * r0 - q * r1;
            i0p[shot] = c * i0v - q * i1v;
            r1p[shot] = c * r1 + q * r0;
            i1p[shot] = c * i1v + q * i0v;
        }
    }
}

void scalar_rotate_real_pair_flip_xmask(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    const double* q_by_shot) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            const double basis_phase_sign = basis_phase_signs[pair_idx++];
            double* r0p = re + i0 * leading_shots;
            double* i0p = im + i0 * leading_shots;
            double* r1p = re + i1 * leading_shots;
            double* i1p = im + i1 * leading_shots;
            SYMFT_BATCH_SIMD_LOOP
            for (int shot = 0; shot < active_shots; ++shot) {
                const double q = basis_phase_sign * q_by_shot[static_cast<std::size_t>(shot)];
                const double r0 = r0p[shot];
                const double i0v = i0p[shot];
                const double r1 = r1p[shot];
                const double i1v = i1p[shot];
                r0p[shot] = c * r0 - q * r1;
                i0p[shot] = c * i0v - q * i1v;
                r1p[shot] = c * r1 + q * r0;
                i1p[shot] = c * i1v + q * i0v;
            }
        }
    }
}

void scalar_rotate_real_pair_flip_xmask_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    const double* basis_phase_signs,
    std::size_t npairs,
    double c,
    double q) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            rotate_real_pair_flip_const(
                re + i0 * leading_shots,
                im + i0 * leading_shots,
                re + i1 * leading_shots,
                im + i1 * leading_shots,
                active_shots,
                c,
                basis_phase_signs[pair_idx++] * q);
        }
    }
}

void scalar_rotate_diagonal_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    const Complex* coeff,
    double c) {
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double fr = c + coeff[basis].real();
        const double fi = coeff[basis].imag();
        double* rp = re + basis * leading_shots;
        double* ip = im + basis * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double v = ip[shot];
            rp[shot] = fr * r - fi * v;
            ip[shot] = fr * v + fi * r;
        }
    }
}

void rotate_diagonal_range_const(
    double* rp,
    double* ip,
    int first_shot,
    int last_shot,
    double fr,
    double fi) {
    SYMFT_BATCH_SIMD_LOOP
    for (int shot = first_shot; shot < last_shot; ++shot) {
        const double r = rp[shot];
        const double v = ip[shot];
        rp[shot] = fr * r - fi * v;
        ip[shot] = fr * v + fi * r;
    }
}

void scalar_rotate_diagonal_mixed(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    const Complex* minus_coeff,
    const Complex* plus_coeff,
    const std::uint64_t* sign_bits,
    double c) {
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const double minus_fr = c + minus_coeff[basis].real();
        const double minus_fi = minus_coeff[basis].imag();
        const double plus_fr = c + plus_coeff[basis].real();
        const double plus_fi = plus_coeff[basis].imag();
        double* rp = re + basis * leading_shots;
        double* ip = im + basis * leading_shots;
        for (std::size_t word = 0; word < nwords; ++word) {
            const int first = static_cast<int>(word << 6);
            const int last = std::min(active_shots, first + 64);
            const std::uint64_t live = live_word_mask(active_shots, word);
            const std::uint64_t bits = sign_bits[word] & live;
            if (bits == 0) {
                rotate_diagonal_range_const(rp, ip, first, last, minus_fr, minus_fi);
            } else if (bits == live) {
                rotate_diagonal_range_const(rp, ip, first, last, plus_fr, plus_fi);
            } else {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const bool sign = ((bits >> (shot - first)) & 1ULL) != 0;
                    const double fr = sign ? plus_fr : minus_fr;
                    const double fi = sign ? plus_fi : minus_fi;
                    const double r = rp[shot];
                    const double v = ip[shot];
                    rp[shot] = fr * r - fi * v;
                    ip[shot] = fr * v + fi * r;
                }
            }
        }
    }
}

void scalar_rotate_general_xmask_const(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const Complex* left_coeff,
    const Complex* right_coeff,
    double c) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            rotate_general_pair_const(
                re + i0 * leading_shots,
                im + i0 * leading_shots,
                re + i1 * leading_shots,
                im + i1 * leading_shots,
                0,
                active_shots,
                c,
                left_coeff[pair_idx],
                right_coeff[pair_idx]);
            ++pair_idx;
        }
    }
}

void scalar_rotate_general_xmask_mixed(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const Complex* left_minus_coeff,
    const Complex* right_minus_coeff,
    const Complex* left_plus_coeff,
    const Complex* right_plus_coeff,
    const std::uint64_t* sign_bits,
    double c) {
    const std::size_t selector = std::size_t{1} << pair_bit;
    const std::size_t dim = npairs << 1;
    const std::size_t step = selector << 1;
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    std::size_t pair_idx = 0;
    for (std::size_t block = 0; block < dim; block += step) {
        for (std::size_t offset = 0; offset < selector; ++offset) {
            const std::size_t i0 = block + offset;
            const std::size_t i1 = i0 ^ static_cast<std::size_t>(xmask);
            double* r0p = re + i0 * leading_shots;
            double* i0p = im + i0 * leading_shots;
            double* r1p = re + i1 * leading_shots;
            double* i1p = im + i1 * leading_shots;
            for (std::size_t word = 0; word < nwords; ++word) {
                const int first = static_cast<int>(word << 6);
                const int last = std::min(active_shots, first + 64);
                const std::uint64_t live = live_word_mask(active_shots, word);
                const std::uint64_t bits = sign_bits[word] & live;
                if (bits == 0) {
                    rotate_general_pair_const(
                        r0p, i0p, r1p, i1p, first, last, c, left_minus_coeff[pair_idx], right_minus_coeff[pair_idx]);
                } else if (bits == live) {
                    rotate_general_pair_const(
                        r0p, i0p, r1p, i1p, first, last, c, left_plus_coeff[pair_idx], right_plus_coeff[pair_idx]);
                } else {
                    const double lmr = left_minus_coeff[pair_idx].real();
                    const double lmi = left_minus_coeff[pair_idx].imag();
                    const double rmr = right_minus_coeff[pair_idx].real();
                    const double rmi = right_minus_coeff[pair_idx].imag();
                    const double lpr = left_plus_coeff[pair_idx].real();
                    const double lpi = left_plus_coeff[pair_idx].imag();
                    const double rpr = right_plus_coeff[pair_idx].real();
                    const double rpi = right_plus_coeff[pair_idx].imag();
                    SYMFT_BATCH_SIMD_LOOP
                    for (int shot = first; shot < last; ++shot) {
                        const bool sign = ((bits >> (shot - first)) & 1ULL) != 0;
                        const double lr = sign ? lpr : lmr;
                        const double li = sign ? lpi : lmi;
                        const double rr = sign ? rpr : rmr;
                        const double ri = sign ? rpi : rmi;
                        const double r0 = r0p[shot];
                        const double im0 = i0p[shot];
                        const double r1 = r1p[shot];
                        const double im1 = i1p[shot];
                        r0p[shot] = c * r0 + rr * r1 - ri * im1;
                        i0p[shot] = c * im0 + rr * im1 + ri * r1;
                        r1p[shot] = c * r1 + lr * r0 - li * im0;
                        i1p[shot] = c * im1 + lr * im0 + li * r0;
                    }
                }
            }
            ++pair_idx;
        }
    }
}

void scalar_promote_first_dormant_rotation(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    std::size_t dim,
    double c,
    const double* q_by_shot) {
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* r0p = re + basis * leading_shots;
        double* i0p = im + basis * leading_shots;
        double* r1p = re + (dim + basis) * leading_shots;
        double* i1p = im + (dim + basis) * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double q = q_by_shot[static_cast<std::size_t>(shot)];
            const double r = r0p[shot];
            const double i = i0p[shot];
            r0p[shot] = c * r;
            i0p[shot] = c * i;
            r1p[shot] = -q * i;
            i1p[shot] = q * r;
        }
    }
}

void scalar_diagonal_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_true,
    std::size_t out_dim,
    double* prob_true) {
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* rp = re + source_true[idx] * leading_shots;
        const double* ip = im + source_true[idx] * leading_shots;
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double r = rp[shot];
            const double i = ip[shot];
            prob_true[shot] += r * r + i * i;
        }
    }
}

void scalar_diagonal_project(
    double* re,
    double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source_false,
    const std::size_t* source_true,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        double* dst_r = re + idx * leading_shots;
        double* dst_i = im + idx * leading_shots;
        const double* false_r = re + source_false[idx] * leading_shots;
        const double* false_i = im + source_false[idx] * leading_shots;
        const double* true_r = re + source_true[idx] * leading_shots;
        const double* true_i = im + source_true[idx] * leading_shots;
        for (std::size_t word = 0; word < nwords; ++word) {
            const int first = static_cast<int>(word << 6);
            const int last = std::min(active_shots, first + 64);
            const std::uint64_t live = live_word_mask(active_shots, word);
            const std::uint64_t bits = branch_bits[word] & live;
            if (bits == 0) {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const double n = invnorms[shot];
                    dst_r[shot] = false_r[shot] * n;
                    dst_i[shot] = false_i[shot] * n;
                }
            } else if (bits == live) {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const double n = invnorms[shot];
                    dst_r[shot] = true_r[shot] * n;
                    dst_i[shot] = true_i[shot] * n;
                }
            } else {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const std::uint64_t branch_mask = 0 - ((bits >> (shot - first)) & 1ULL);
                    const double n = invnorms[shot];
                    dst_r[shot] = select_double_bits(false_r[shot], true_r[shot], branch_mask) * n;
                    dst_i[shot] = select_double_bits(false_i[shot], true_i[shot], branch_mask) * n;
                }
            }
        }
    }
}

void scalar_nondiagonal_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    std::size_t out_dim,
    double* prob_true) {
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = re + source0_false[idx] * leading_shots;
        const double* i0p = im + source0_false[idx] * leading_shots;
        const double* r1p = re + source1_false[idx] * leading_shots;
        const double* i1p = im + source1_false[idx] * leading_shots;
        const double c1r = coeff1_false_real[idx];
        const double c1i = coeff1_false_imag[idx];
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
            const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
            const double amp_r = kInvSqrt2 * r0p[shot] - prod_r;
            const double amp_i = kInvSqrt2 * i0p[shot] - prod_i;
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i;
        }
    }
}

void scalar_nondiagonal_xmask_measure_true_prob(
    const double* re,
    const double* im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    double* prob_true) {
#if defined(__AVX2__) && defined(__FMA__)
    if (nondiagonal_xmask_measure_true_prob_pitch2_avx2(
            re,
            im,
            leading_shots,
            active_shots,
            xmask,
            pair_bit,
            npairs,
            coeff1_false_real,
            coeff1_false_imag,
            prob_true)) {
        return;
    }
#endif
    std::fill(prob_true, prob_true + active_shots, 0.0);
    for (std::size_t pair_idx = 0; pair_idx < npairs; ++pair_idx) {
        const std::size_t source0 = insert_zero_bit(pair_idx, pair_bit);
        const std::size_t source1 = source0 ^ static_cast<std::size_t>(xmask);
        const double* r0p = re + source0 * leading_shots;
        const double* i0p = im + source0 * leading_shots;
        const double* r1p = re + source1 * leading_shots;
        const double* i1p = im + source1 * leading_shots;
        const double c1r = coeff1_false_real[pair_idx];
        const double c1i = coeff1_false_imag[pair_idx];
        SYMFT_BATCH_SIMD_LOOP
        for (int shot = 0; shot < active_shots; ++shot) {
            const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
            const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
            const double amp_r = kInvSqrt2 * r0p[shot] - prod_r;
            const double amp_i = kInvSqrt2 * i0p[shot] - prod_i;
            prob_true[shot] += amp_r * amp_r + amp_i * amp_i;
        }
    }
}

void scalar_nondiagonal_project(
    const double* re,
    const double* im,
    double* scratch_re,
    double* scratch_im,
    std::size_t leading_shots,
    int active_shots,
    const std::size_t* source0_false,
    const std::size_t* source1_false,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    std::size_t out_dim,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    for (std::size_t idx = 0; idx < out_dim; ++idx) {
        const double* r0p = re + source0_false[idx] * leading_shots;
        const double* i0p = im + source0_false[idx] * leading_shots;
        const double* r1p = re + source1_false[idx] * leading_shots;
        const double* i1p = im + source1_false[idx] * leading_shots;
        double* dst_r = scratch_re + idx * leading_shots;
        double* dst_i = scratch_im + idx * leading_shots;
        const double c1r = coeff1_false_real[idx];
        const double c1i = coeff1_false_imag[idx];
        for (std::size_t word = 0; word < nwords; ++word) {
            const int first = static_cast<int>(word << 6);
            const int last = std::min(active_shots, first + 64);
            const std::uint64_t live = live_word_mask(active_shots, word);
            const std::uint64_t bits = branch_bits[word] & live;
            if (bits == 0 || bits == live) {
                const double branch_sign = bits == 0 ? 1.0 : -1.0;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
                    const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
                    const double n = invnorms[shot];
                    dst_r[shot] = (kInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
                    dst_i[shot] = (kInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
                }
            } else {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const double branch_sign = ((bits >> (shot - first)) & 1ULL) != 0 ? -1.0 : 1.0;
                    const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
                    const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
                    const double n = invnorms[shot];
                    dst_r[shot] = (kInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
                    dst_i[shot] = (kInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
                }
            }
        }
    }
}

void scalar_nondiagonal_xmask_project(
    const double* re,
    const double* im,
    double* scratch_re,
    double* scratch_im,
    std::size_t leading_shots,
    int active_shots,
    std::uint64_t xmask,
    unsigned pair_bit,
    std::size_t npairs,
    const double* coeff1_false_real,
    const double* coeff1_false_imag,
    const std::uint64_t* branch_bits,
    const double* invnorms) {
#if defined(__AVX2__) && defined(__FMA__)
    if (nondiagonal_xmask_project_pitch2_avx2(
            re,
            im,
            scratch_re,
            scratch_im,
            leading_shots,
            active_shots,
            xmask,
            pair_bit,
            npairs,
            coeff1_false_real,
            coeff1_false_imag,
            branch_bits,
            invnorms)) {
        return;
    }
#endif
    const std::size_t nwords = static_cast<std::size_t>((active_shots + 63) >> 6);
    for (std::size_t pair_idx = 0; pair_idx < npairs; ++pair_idx) {
        const std::size_t source0 = insert_zero_bit(pair_idx, pair_bit);
        const std::size_t source1 = source0 ^ static_cast<std::size_t>(xmask);
        const double* r0p = re + source0 * leading_shots;
        const double* i0p = im + source0 * leading_shots;
        const double* r1p = re + source1 * leading_shots;
        const double* i1p = im + source1 * leading_shots;
        double* dst_r = scratch_re + pair_idx * leading_shots;
        double* dst_i = scratch_im + pair_idx * leading_shots;
        const double c1r = coeff1_false_real[pair_idx];
        const double c1i = coeff1_false_imag[pair_idx];
        for (std::size_t word = 0; word < nwords; ++word) {
            const int first = static_cast<int>(word << 6);
            const int last = std::min(active_shots, first + 64);
            const std::uint64_t live = live_word_mask(active_shots, word);
            const std::uint64_t bits = branch_bits[word] & live;
            if (bits == 0 || bits == live) {
                const double branch_sign = bits == 0 ? 1.0 : -1.0;
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
                    const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
                    const double n = invnorms[shot];
                    dst_r[shot] = (kInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
                    dst_i[shot] = (kInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
                }
            } else {
                SYMFT_BATCH_SIMD_LOOP
                for (int shot = first; shot < last; ++shot) {
                    const double branch_sign = ((bits >> (shot - first)) & 1ULL) != 0 ? -1.0 : 1.0;
                    const double prod_r = c1r * r1p[shot] - c1i * i1p[shot];
                    const double prod_i = c1r * i1p[shot] + c1i * r1p[shot];
                    const double n = invnorms[shot];
                    dst_r[shot] = (kInvSqrt2 * r0p[shot] + branch_sign * prod_r) * n;
                    dst_i[shot] = (kInvSqrt2 * i0p[shot] + branch_sign * prod_i) * n;
                }
            }
        }
    }
}

const KernelTable interleaved_kernels = {
    scalar_rotate_uniform_imag_pairs,
    scalar_rotate_uniform_imag_xmask,
    scalar_rotate_uniform_imag_xmask_const,
    scalar_rotate_real_pair_flip,
    scalar_rotate_real_pair_flip_xmask,
    scalar_rotate_real_pair_flip_xmask_const,
    scalar_rotate_diagonal_const,
    scalar_rotate_diagonal_mixed,
    scalar_rotate_general_xmask_const,
    scalar_rotate_general_xmask_mixed,
    scalar_promote_first_dormant_rotation,
    scalar_diagonal_measure_true_prob,
    scalar_diagonal_project,
    scalar_nondiagonal_measure_true_prob,
    scalar_nondiagonal_xmask_measure_true_prob,
    scalar_nondiagonal_project,
    scalar_nondiagonal_xmask_project,
};

} // namespace

const KernelTable& table() {
    return interleaved_kernels;
}

} // namespace symft::batch_interleaved

#undef SYMFT_BATCH_SIMD_LOOP
