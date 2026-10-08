#include "sampler/real_active.hpp"
#include "sampler/active_internal.hpp"
#include "sampler/active_kernels.hpp"
#include "sampler/cpu_profile.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace symft::detail {
namespace {
bool parity(unsigned value) { return std::popcount(value) & 1; }
constexpr double invsqrt2 = 0.707106781186547524400844362104849;
}

bool real_rotation_compatible(const PrecomputedActivePauliRotationKernel& kernel, unsigned gauge) {
    const auto q = kernel.minus_even_coefficient;
    if (q.real() != 0 && q.imag() != 0) return false;
    return parity(gauge & kernel.action.xmask) == (q.imag() != 0);
}

bool real_measurement_compatible(const PrecomputedActivePauliMeasurementKernel& kernel, unsigned gauge) {
    if (kernel.is_diagonal) return true;
    const auto q = kernel.nondiagonal_coefficient1_even;
    if (q.real() != 0 && q.imag() != 0) return false;
    return parity(gauge & kernel.action.xmask) == (q.imag() != 0);
}

unsigned real_measurement_next_gauge(const PrecomputedActivePauliMeasurementKernel& kernel, unsigned gauge) {
    if (kernel.is_diagonal && (gauge & (1U << kernel.pivot))) gauge ^= kernel.action.zmask;
    const unsigned low = (1U << kernel.pivot) - 1;
    return (gauge & low) | ((gauge >> (kernel.pivot + 1)) << kernel.pivot);
}

RealRotationKernel make_real_rotation(const PrecomputedActivePauliRotationKernel& kernel, unsigned gauge) {
    if (!real_rotation_compatible(kernel, gauge)) throw std::runtime_error("rotation has no real gauge");
    RealRotationKernel out;
    out.dim = 1U << kernel.action.nqubits;
    out.xmask = static_cast<unsigned>(kernel.action.xmask);
    out.pair_bit = kernel.pair_bit;
    out.c = kernel.cos_kernel_angle;
    const auto q = kernel.minus_even_coefficient;
    const unsigned phase_mask = static_cast<unsigned>(kernel.action.zmask) ^ (q.imag() != 0 ? gauge : 0);
    const double coefficient = q.imag() != 0 ? q.imag() : q.real();
    const unsigned count = out.xmask == 0 ? out.dim : out.dim / 2;
    if (out.xmask != 0 && coefficient != 0 && !parity(out.xmask & phase_mask)) {
        throw std::runtime_error("non-orthogonal real rotation kernel");
    }
    for (unsigned j = 0; j < count; ++j) {
        const unsigned left = out.xmask == 0 ? j : static_cast<unsigned>(insert_zero_bit(j, out.pair_bit));
        out.left.push_back(left);
        out.right.push_back(left ^ out.xmask);
        out.coefficients.push_back(parity(left & phase_mask) ? -coefficient : coefficient);
    }
    if (out.xmask && out.pair_bit < 3) {
        out.small_pair_dest_coefficients.resize(out.dim);
        for (unsigned j = 0; j < count; ++j) {
            out.small_pair_dest_coefficients[out.left[j]] = -out.coefficients[j];
            out.small_pair_dest_coefficients[out.right[j]] = out.coefficients[j];
        }
    }
    return out;
}

RealMeasurementKernel make_real_measurement(const PrecomputedActivePauliMeasurementKernel& kernel, unsigned gauge) {
    if (!real_measurement_compatible(kernel, gauge)) throw std::runtime_error("measurement has no real gauge");
    RealMeasurementKernel out;
    out.diagonal = kernel.is_diagonal;
    out.pivot = static_cast<unsigned>(kernel.pivot);
    out.xmask = static_cast<unsigned>(kernel.action.xmask);
    out.next_gauge = real_measurement_next_gauge(kernel, gauge);
    for (unsigned j = 0; j < kernel.out_dim; ++j) {
        const unsigned base = static_cast<unsigned>(insert_zero_bit(j, kernel.pivot));
        if (kernel.is_diagonal) {
            out.source0.push_back(static_cast<unsigned>(compact_diagonal_measurement_source(kernel, j, false)));
            out.source1.push_back(static_cast<unsigned>(compact_diagonal_measurement_source(kernel, j, true)));
            const bool odd = (gauge & (1U << kernel.pivot)) && parity(j & out.next_gauge);
            out.diagonal_sign0.push_back(odd && kernel.diagonal_phase_bit ? -1 : 1);
            out.diagonal_sign1.push_back(odd && !kernel.diagonal_phase_bit ? -1 : 1);
        } else {
            out.source0.push_back(base);
            out.source1.push_back(base ^ static_cast<unsigned>(kernel.action.xmask));
            const auto q = kernel.nondiagonal_coefficient1_even;
            double coefficient = q.imag() != 0 ? (parity(base & gauge) ? q.imag() : -q.imag()) : q.real();
            if (parity(base & kernel.action.zmask)) coefficient = -coefficient;
            out.coefficient.push_back(coefficient);
        }
    }
    return out;
}

void rotate_real_active(double* r, const RealRotationKernel& kernel, bool sign) {
    const double direction = sign ? -1 : 1;
    const double c = kernel.c;
    const unsigned n = static_cast<unsigned>(kernel.coefficients.size());
    if (kernel.xmask == 0) {
        for (unsigned j = 0; j < n; ++j) r[j] *= c + direction * kernel.coefficients[j];
        return;
    }
#if defined(__AVX512F__) && defined(__FMA__)
    // The pair bit is the highest set bit of xmask. Within an eight-lane
    // block all partners can be permuted in registers; higher pairs use two
    // contiguous blocks. No gathers, scatter stores, or precision changes.
    if (kernel.dim >= 8) {
        const auto vc = _mm512_set1_pd(c), vd = _mm512_set1_pd(direction);
        const auto lanes = _mm512_xor_si512(_mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0),
                                           _mm512_set1_epi64(kernel.xmask & 7U));
        if (kernel.pair_bit < 3) {
            for (unsigned base = 0; base < kernel.dim; base += 8) {
                const auto a = _mm512_loadu_pd(r + base);
                const auto b = _mm512_permutexvar_pd(lanes, a);
                const auto q = _mm512_mul_pd(vd, _mm512_loadu_pd(kernel.small_pair_dest_coefficients.data() + base));
                _mm512_storeu_pd(r + base, _mm512_fmadd_pd(q, b, _mm512_mul_pd(vc, a)));
            }
        } else {
            const unsigned selector = 1U << kernel.pair_bit;
            const unsigned low_mask = kernel.xmask & (selector - 1);
            for (unsigned block = 0; block < kernel.dim; block += 2 * selector) {
                for (unsigned offset = 0; offset < selector; offset += 8) {
                    const unsigned i0 = block + offset;
                    const unsigned i1 = block + selector + (offset ^ (low_mask & ~7U));
                    const auto a = _mm512_loadu_pd(r + i0);
                    const auto b = _mm512_permutexvar_pd(lanes, _mm512_loadu_pd(r + i1));
                    const auto q = _mm512_mul_pd(vd, _mm512_loadu_pd(kernel.coefficients.data() + block / 2 + offset));
                    const auto u = _mm512_fnmadd_pd(q, b, _mm512_mul_pd(vc, a));
                    const auto v = _mm512_fmadd_pd(q, a, _mm512_mul_pd(vc, b));
                    _mm512_storeu_pd(r + i0, u);
                    _mm512_storeu_pd(r + i1, _mm512_permutexvar_pd(lanes, v));
                }
            }
        }
        return;
    }
#endif
#if defined(__AVX2__) && defined(__FMA__)
    const __m256d vc = _mm256_set1_pd(c), vd = _mm256_set1_pd(direction);
    if (kernel.dim >= 4 && kernel.pair_bit < 2) {
        for (unsigned base = 0; base < kernel.dim; base += 4) {
            const auto a = _mm256_loadu_pd(r + base);
            const auto b = permute_lanes_xor2_inline(a, kernel.xmask);
            const auto q = _mm256_mul_pd(vd, _mm256_loadu_pd(kernel.small_pair_dest_coefficients.data() + base));
            _mm256_storeu_pd(r + base, _mm256_fmadd_pd(q, b, _mm256_mul_pd(vc, a)));
        }
        return;
    }
    if (kernel.pair_bit >= 2) {
        const unsigned selector = 1U << kernel.pair_bit;
        const unsigned low_mask = kernel.xmask & (selector - 1);
        for (unsigned block = 0; block < kernel.dim; block += 2 * selector) {
            for (unsigned offset = 0; offset < selector; offset += 4) {
                const unsigned i0 = block + offset;
                const unsigned i1 = block + selector + (offset ^ (low_mask & ~3U));
                const auto a = _mm256_loadu_pd(r + i0);
                const auto b = permute_lanes_xor2_inline(_mm256_loadu_pd(r + i1), low_mask);
                const auto q = _mm256_mul_pd(vd, _mm256_loadu_pd(kernel.coefficients.data() + block / 2 + offset));
                const auto u = _mm256_fnmadd_pd(q, b, _mm256_mul_pd(vc, a));
                const auto v = _mm256_fmadd_pd(q, a, _mm256_mul_pd(vc, b));
                _mm256_storeu_pd(r + i0, u);
                _mm256_storeu_pd(r + i1, permute_lanes_xor2_inline(v, low_mask));
            }
        }
        return;
    }
#endif
    // Pairs are disjoint, which licenses vectorization across pair indices.
    SYMFT_SINGLE_SIMD_LOOP
    for (unsigned j = 0; j < n; ++j) {
        const unsigned a = kernel.left[j], b = kernel.right[j];
        const double u = r[a], v = r[b], q = direction * kernel.coefficients[j];
        r[a] = c * u - q * v;
        r[b] = c * v + q * u;
    }
}

void promote_real_active(double* r, unsigned dim, unsigned gauge, double c, double q) {
#if defined(__AVX512F__)
    if (dim >= 8) {
        const auto vc = _mm512_set1_pd(c), vq = _mm512_set1_pd(q), vnq = _mm512_set1_pd(-q);
        const unsigned lane_signs = ((gauge & 1) ? 0xaaU : 0) ^
                                    ((gauge & 2) ? 0xccU : 0) ^ ((gauge & 4) ? 0xf0U : 0);
        for (unsigned j = 0; j < dim; j += 8) {
            const auto signs = static_cast<__mmask8>(lane_signs ^ (parity(j & gauge) ? 0xffU : 0));
            const auto u = _mm512_loadu_pd(r + j);
            _mm512_storeu_pd(r + dim + j, _mm512_mul_pd(_mm512_mask_mov_pd(vq, signs, vnq), u));
            _mm512_storeu_pd(r + j, _mm512_mul_pd(vc, u));
        }
        return;
    }
#endif
    for (unsigned j = 0; j < dim; ++j) {
        const double u = r[j];
        r[dim + j] = (parity(j & gauge) ? -q : q) * u;
        r[j] = c * u;
    }
}

double real_measurement_probability(const double* r, const RealMeasurementKernel& kernel) {
    ScopedCpuTimer cpu_timer(CpuPhase::Probability);
    double probability = 0;
    const unsigned n = static_cast<unsigned>(kernel.source0.size());
    unsigned first = 0;
#if defined(__AVX2__) && defined(__FMA__)
    __m256d total = _mm256_setzero_pd();
    const auto h = _mm256_set1_pd(invsqrt2);
    for (; first + 4 <= n; first += 4) {
        __m256d u;
        if (kernel.diagonal) {
            u = _mm256_i32gather_pd(r, _mm_loadu_si128(reinterpret_cast<const __m128i*>(kernel.source1.data() + first)), 8);
        } else {
            __m256d a, b;
            if (kernel.pivot >= 2) {
                const unsigned base = kernel.source0[first];
                a = _mm256_loadu_pd(r + base);
                b = permute_lanes_xor2_inline(_mm256_loadu_pd(r + ((base ^ kernel.xmask) & ~3U)), kernel.xmask);
            } else {
                a = _mm256_i32gather_pd(r, _mm_loadu_si128(reinterpret_cast<const __m128i*>(kernel.source0.data() + first)), 8);
                b = _mm256_i32gather_pd(r, _mm_loadu_si128(reinterpret_cast<const __m128i*>(kernel.source1.data() + first)), 8);
            }
            u = _mm256_fnmadd_pd(_mm256_loadu_pd(kernel.coefficient.data() + first), b, _mm256_mul_pd(h, a));
        }
        total = _mm256_fmadd_pd(u, u, total);
    }
    alignas(32) double sums[4];
    _mm256_store_pd(sums, total);
    probability = (sums[0] + sums[1]) + (sums[2] + sums[3]);
#endif
    if (kernel.diagonal) {
        for (unsigned j = first; j < n; ++j) {
            const double u = r[kernel.source1[j]];
            probability += u * u;
        }
    } else {
        for (unsigned j = first; j < n; ++j) {
            const double u = invsqrt2 * r[kernel.source0[j]] - kernel.coefficient[j] * r[kernel.source1[j]];
            probability += u * u;
        }
    }
    return std::clamp(probability, 0.0, 1.0);
}

void project_real_active(const double* r, double* out, const RealMeasurementKernel& kernel,
                         bool branch, double invnorm) {
    ScopedCpuTimer cpu_timer(CpuPhase::Projection);
    const unsigned n = static_cast<unsigned>(kernel.source0.size());
    unsigned first = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const auto scale = _mm256_set1_pd(invnorm);
    const auto direction = _mm256_set1_pd(branch ? -1 : 1);
    const auto h = _mm256_set1_pd(invsqrt2);
    for (; first + 4 <= n; first += 4) {
        __m256d u;
        if (kernel.diagonal) {
            const auto& sources = branch ? kernel.source1 : kernel.source0;
            const auto& signs = branch ? kernel.diagonal_sign1 : kernel.diagonal_sign0;
            u = _mm256_mul_pd(_mm256_loadu_pd(signs.data() + first),
                _mm256_i32gather_pd(r, _mm_loadu_si128(reinterpret_cast<const __m128i*>(sources.data() + first)), 8));
        } else {
            __m256d a, b;
            if (kernel.pivot >= 2) {
                const unsigned base = kernel.source0[first];
                a = _mm256_loadu_pd(r + base);
                b = permute_lanes_xor2_inline(_mm256_loadu_pd(r + ((base ^ kernel.xmask) & ~3U)), kernel.xmask);
            } else {
                a = _mm256_i32gather_pd(r, _mm_loadu_si128(reinterpret_cast<const __m128i*>(kernel.source0.data() + first)), 8);
                b = _mm256_i32gather_pd(r, _mm_loadu_si128(reinterpret_cast<const __m128i*>(kernel.source1.data() + first)), 8);
            }
            const auto q = _mm256_mul_pd(direction, _mm256_loadu_pd(kernel.coefficient.data() + first));
            u = _mm256_fmadd_pd(q, b, _mm256_mul_pd(h, a));
        }
        _mm256_storeu_pd(out + first, _mm256_mul_pd(scale, u));
    }
#endif
    if (kernel.diagonal) {
        const auto& sources = branch ? kernel.source1 : kernel.source0;
        const auto& signs = branch ? kernel.diagonal_sign1 : kernel.diagonal_sign0;
        SYMFT_SINGLE_SIMD_LOOP
        for (unsigned j = first; j < n; ++j) out[j] = signs[j] * r[sources[j]] * invnorm;
    } else {
        const double direction = branch ? -1 : 1;
        SYMFT_SINGLE_SIMD_LOOP
        for (unsigned j = first; j < n; ++j) {
            out[j] = (invsqrt2 * r[kernel.source0[j]] + direction * kernel.coefficient[j] * r[kernel.source1[j]]) * invnorm;
        }
    }
}
} // namespace symft::detail
