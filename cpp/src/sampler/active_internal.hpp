#pragma once

#include "core/internal.hpp"
#include "sampler/active.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

namespace symft::detail {

inline constexpr std::size_t kNoSource = std::numeric_limits<std::size_t>::max();
inline constexpr std::size_t kSimdPairRotationThreshold = 16384;

#if defined(__clang__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_SINGLE_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_SINGLE_SIMD_LOOP
#endif

inline Complex phase_factor(int phase) {
    switch (phase & 3) {
    case 0:
        return {1.0, 0.0};
    case 1:
        return {0.0, 1.0};
    case 2:
        return {-1.0, 0.0};
    default:
        return {0.0, -1.0};
    }
}

inline std::size_t active_length(int k) {
    if (k < 0 || k >= 62) {
        fail("active qubit count is too large for machine basis indices");
    }
    return std::size_t{1} << k;
}

inline std::size_t insert_zero_bit(std::size_t packed, int bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - std::size_t{1};
    const std::size_t low = packed & low_mask;
    const std::size_t high = packed & ~low_mask;
    return low | (high << 1);
}

inline std::uint64_t active_mask_x(const PauliString& pauli) {
    return pauli.x.empty() ? 0 : pauli.x[0];
}

inline std::uint64_t active_mask_z(const PauliString& pauli) {
    return pauli.z.empty() ? 0 : pauli.z[0];
}

inline bool active_action_phase_odd(const ActivePauliAction& action, std::size_t basis) {
    return is_odd_popcount(static_cast<std::uint64_t>(basis) & action.zmask);
}

inline Complex active_action_phase(const ActivePauliAction& action, std::size_t basis) {
    return active_action_phase_odd(action, basis) ? action.odd_phase : action.even_phase;
}

inline bool is_near_zero(double value) {
    return std::abs(value) < 1e-14;
}

inline bool can_rotate_real_pair_flip(const ActivePauliAction& action) {
    if (action.zmask == 0 || !action.xz_overlap_odd) {
        return false;
    }
    return is_near_zero(action.even_phase.real()) && !is_near_zero(action.even_phase.imag());
}

inline Complex compact_rotation_coefficient(
    const PrecomputedActivePauliRotationKernel& kernel,
    std::size_t source,
    bool sign) {
    const bool odd = active_action_phase_odd(kernel.action, source);
    return sign != odd ? -kernel.minus_even_coefficient : kernel.minus_even_coefficient;
}

inline std::size_t compact_diagonal_measurement_source(
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t packed,
    bool branch) {
    const std::size_t without_pivot = insert_zero_bit(packed, kernel.pivot);
    const int parity = popcount64(static_cast<std::uint64_t>(without_pivot) & kernel.z_without_pivot) & 1;
    const int pivot_value = kernel.diagonal_phase_bit ^ parity ^ static_cast<int>(branch);
    return without_pivot | (std::size_t{static_cast<unsigned>(pivot_value)} << kernel.pivot);
}

inline std::size_t compact_nondiagonal_measurement_source0(
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t packed) {
    return insert_zero_bit(packed, kernel.pivot);
}

inline std::size_t compact_nondiagonal_measurement_source1(
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t packed) {
    return compact_nondiagonal_measurement_source0(kernel, packed) ^ static_cast<std::size_t>(kernel.action.xmask);
}

inline Complex compact_nondiagonal_measurement_coefficient1(
    const PrecomputedActivePauliMeasurementKernel& kernel,
    std::size_t packed,
    bool branch) {
    const bool odd = active_action_phase_odd(
        kernel.action, compact_nondiagonal_measurement_source0(kernel, packed));
    return branch != odd ? -kernel.nondiagonal_coefficient1_even : kernel.nondiagonal_coefficient1_even;
}

} // namespace symft::detail
