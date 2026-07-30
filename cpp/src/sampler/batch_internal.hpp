#pragma once

#include "sampler/active_internal.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/contiguous_active.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/presampled_expression.hpp"
#include "sampler/random.hpp"
#include "simd/batch_interleaved.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace symft {

inline constexpr int kDefaultBatchShots = 2048;
inline constexpr std::size_t kDefaultBatchActiveAmplitudes = std::size_t{1} << 15;
inline constexpr std::size_t kXmaskRotationPairThreshold = 64;
inline constexpr std::size_t kBatchScalarSymbolicEvalThreshold = 32;
inline constexpr int kBatchActiveLaneAlignment = 4;

#if defined(__clang__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BATCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BATCH_SIMD_LOOP
#endif

enum class BatchSignMode {
    AllMinus,
    AllPlus,
    Mixed,
};

using detail::active_length;
using detail::check_probability;
using detail::fail;
using detail::symbol_bit_mask;
using detail::symbol_word_count;
using detail::symbol_word_index;
using detail::trailing_zeros64;

inline std::size_t batch_word_count(int shots) {
    if (shots <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((shots + 63) >> 6);
}

inline int padded_batch_active_pitch(int shot_capacity) {
    if (shot_capacity <= 2) {
        return std::max(shot_capacity, 1);
    }
    const int lanes = std::max(shot_capacity, kBatchActiveLaneAlignment);
    return ((lanes + kBatchActiveLaneAlignment - 1) / kBatchActiveLaneAlignment) * kBatchActiveLaneAlignment;
}

inline std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

inline std::uint64_t batch_live_word_mask(const BatchFactoredExecutorState& runtime, std::size_t word) {
    const int remaining = runtime.active_shots - static_cast<int>(word << 6);
    return low_bits_mask(remaining);
}

inline std::size_t runtime_batch_word_count(const BatchFactoredExecutorState& runtime) {
    return batch_word_count(runtime.active_shots);
}

inline std::uint64_t batch_shot_mask(int shot) {
    return std::uint64_t{1} << (shot & 63);
}

inline std::size_t batch_shot_word(int shot) {
    return static_cast<std::size_t>(shot >> 6);
}

inline void set_batch_bit(std::vector<std::uint64_t>& bits, int shot) {
    bits[batch_shot_word(shot)] |= batch_shot_mask(shot);
}

inline std::size_t batch_active_offset(const BatchFactoredExecutorState& runtime, std::size_t basis, int shot) {
    if (runtime.dense_shot_major_active) {
        return static_cast<std::size_t>(shot) * runtime.active_stride + basis;
    }
    return basis * static_cast<std::size_t>(runtime.active_pitch) + static_cast<std::size_t>(shot);
}

inline double* batch_active_re_for_shot(BatchFactoredExecutorState& runtime, int shot) {
    return runtime.active_re.data() + static_cast<std::size_t>(shot) * runtime.active_stride;
}

inline double* batch_active_im_for_shot(BatchFactoredExecutorState& runtime, int shot) {
    return runtime.active_im.data() + static_cast<std::size_t>(shot) * runtime.active_stride;
}

inline double* batch_scratch_re_for_shot(BatchFactoredExecutorState& runtime, int shot) {
    return runtime.scratch_re.data() + static_cast<std::size_t>(shot) * runtime.active_stride;
}

inline double* batch_scratch_im_for_shot(BatchFactoredExecutorState& runtime, int shot) {
    return runtime.scratch_im.data() + static_cast<std::size_t>(shot) * runtime.active_stride;
}

inline std::size_t batch_condition_offset(const BatchFactoredExecutorState& runtime, int condition, std::size_t word) {
    return static_cast<std::size_t>(condition - 1) * runtime.batch_words + word;
}

inline std::size_t batch_record_offset(const BatchFactoredExecutorState& runtime, int record, std::size_t word) {
    return static_cast<std::size_t>(record - 1) * runtime.batch_words + word;
}

inline std::size_t batch_detector_offset(const BatchFactoredExecutorState& runtime, int detector, std::size_t word) {
    return static_cast<std::size_t>(detector - 1) * runtime.batch_words + word;
}

inline void fill_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime, bool value) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < runtime.batch_words) {
        bits.resize(runtime.batch_words, 0);
    }
    const std::uint64_t fill = value ? std::numeric_limits<std::uint64_t>::max() : 0;
    for (std::size_t word = 0; word < nwords; ++word) {
        bits[word] = fill & batch_live_word_mask(runtime, word);
    }
    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(nwords), bits.end(), 0);
}

inline void mask_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (nwords == 0) {
        return;
    }
    const int live_bits = runtime.active_shots & 63;
    if (live_bits != 0) {
        bits[nwords - 1] &= low_bits_mask(live_bits);
    }
}

inline void invert_batch_bits(std::vector<std::uint64_t>& bits, const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < nwords; ++word) {
        bits[word] = ~bits[word] & batch_live_word_mask(runtime, word);
    }
    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(nwords), bits.end(), 0);
}

inline void fill_batch_random_half_bits(std::vector<std::uint64_t>& bits, BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < runtime.batch_words) {
        bits.resize(runtime.batch_words, 0);
    }
    if ((runtime.active_shots & 63) == 0) {
        for (std::size_t word = 0; word < nwords; ++word) {
            bits[word] = next_random_u64(runtime.rng_state);
        }
    } else {
        for (std::size_t word = 0; word < nwords; ++word) {
            bits[word] = next_random_u64(runtime.rng_state) & batch_live_word_mask(runtime, word);
        }
    }
    std::fill(bits.begin() + static_cast<std::ptrdiff_t>(nwords), bits.end(), 0);
}



void assign_batch_symbol(BatchFactoredExecutorState& runtime, int condition, const std::vector<std::uint64_t>& bits);
void assign_batch_symbol(
    BatchFactoredExecutorState& runtime,
    std::optional<int> condition,
    const std::vector<std::uint64_t>& bits);
void eval_symbolic_bool_batch(
    std::vector<std::uint64_t>& out,
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime);
void xor_symbolic_bool_batch_into(
    std::vector<std::uint64_t>& out,
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime);
void write_batch_measurement_record(
    BatchFactoredExecutorState& runtime,
    std::optional<int> record,
    const std::vector<std::uint64_t>& outcome_bits,
    std::optional<int> record_condition);

inline bool write_direct_branch_measurement_record(
    BatchFactoredExecutorState& runtime,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition) {
    if (outcome_plan.conditions.size() != 1 || outcome_plan.conditions.front() != branch_condition) {
        return false;
    }
    if (outcome_plan.constant) {
        invert_batch_bits(runtime.eval_scratch, runtime);
    }
    write_batch_measurement_record(runtime, record, runtime.eval_scratch, record_condition);
    return true;
}

void write_batch_detector_record(
    BatchFactoredExecutorState& runtime,
    int detector,
    const std::vector<std::uint64_t>& outcome_bits);
void assign_presampled_exogenous_batch(BatchFactoredExecutorState& runtime, const PresampledExogenous& samples);
void assign_presampled_exogenous_batch(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int first_sample_shot);
void sample_exogenous_symbols_batch(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void rotate_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits);
void promote_first_dormant_rotation_batch(
    BatchFactoredExecutorState& runtime,
    double kernel_angle,
    const std::vector<std::uint64_t>& sign_bits);
void measure_precomputed_active_pauli_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition,
    const SymbolicBoolEvaluationPlan& outcome_plan,
    std::optional<int> record,
    std::optional<int> record_condition);
void measure_precomputed_active_pauli_branch_batch(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    int branch_condition);
} // namespace symft
