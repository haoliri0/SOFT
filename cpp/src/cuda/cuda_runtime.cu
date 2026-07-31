#include "cuda/cuda_runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace symft::cuda {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status == cudaSuccess) {
        return;
    }

    const char* raw_message = cudaGetErrorString(status);
    std::string message = std::string(what) + ": " + (raw_message == nullptr ? "unknown CUDA error" : raw_message);
    if (raw_message != nullptr) {
        const std::string detail(raw_message);
        if (detail.find("unsupported toolchain") != std::string::npos ||
            detail.find("unsupported PTX") != std::string::npos) {
            message +=
                ". Rebuild the CUDA extension for this GPU architecture, for example "
                "SYMFT_PY_ENABLE_CUDA=1 SYMFT_PY_CUDA_ARCH=sm_XX "
                "python setup.py build_ext --inplace, or use a CUDA toolkit supported "
                "by the installed NVIDIA driver";
        }
    }
    throw Error(message);
}

std::uint64_t host_mix_u64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

template <typename T>
struct DeviceArray {
    T* ptr = nullptr;
    std::size_t count = 0;
    std::size_t capacity = 0;

    DeviceArray() = default;
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& other) noexcept
        : ptr(other.ptr), count(other.count), capacity(other.capacity) {
        other.ptr = nullptr;
        other.count = 0;
        other.capacity = 0;
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            reset();
            ptr = other.ptr;
            count = other.count;
            capacity = other.capacity;
            other.ptr = nullptr;
            other.count = 0;
            other.capacity = 0;
        }
        return *this;
    }

    ~DeviceArray() {
        reset();
    }

    void reset() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        count = 0;
        capacity = 0;
    }

    void reserve(std::size_t next_capacity) {
        if (next_capacity <= capacity) {
            return;
        }
        reset();
        if (next_capacity != 0) {
            check_cuda(cudaMalloc(&ptr, next_capacity * sizeof(T)), "cudaMalloc");
        }
        capacity = next_capacity;
    }

    void upload(const std::vector<T>& values) {
        reserve(values.size());
        count = values.size();
        if (!values.empty()) {
            check_cuda(
                cudaMemcpy(ptr, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice),
                "cudaMemcpy host-to-device");
        }
    }

    void upload_raw(const T* values, std::size_t n) {
        reserve(n);
        count = n;
        if (n != 0) {
            check_cuda(cudaMemcpy(ptr, values, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy host-to-device");
        }
    }

    void resize_uninitialized(std::size_t n) {
        reserve(n);
        count = n;
    }
};

struct DeviceProgramView {
    int initial_k = 0;
    int max_k = 0;
    int symbol_words = 0;
    int record_words = 0;
    int instruction_count = 0;
    int expression_count = 0;
    int rotation_count = 0;
    int measurement_count = 0;
    int block_expression_count = 0;
    int max_rotation_run_length = 0;
    int logical_group_count = 0;
    int categorical_count = 0;
    int rare_categorical_count = 0;
    int bernoulli_count = 0;
    int low_probability_group_count = 0;

    const CudaInstruction* instructions = nullptr;
    const CudaRotationRunItem* rotation_run_items = nullptr;
    const CudaExpression* expressions = nullptr;
    const CudaWordMask* residual_masks = nullptr;
    const CudaRotationKernel* rotations = nullptr;
    const CudaMeasurementKernel* measurements = nullptr;
    const CudaComplex* complex_table = nullptr;
    const CudaReal* real_table = nullptr;
    const int* source_table = nullptr;
    const int* record_table = nullptr;
    const int* logical_group_offsets = nullptr;
    const int* logical_group_sizes = nullptr;
    const CudaCategoricalDistribution* categorical_distributions = nullptr;
    const CudaRareCategoricalGroup* rare_categorical_groups = nullptr;
    const CudaBernoulliCondition* bernoulli_conditions = nullptr;
    const CudaBernoulliGroup* low_probability_bernoulli_groups = nullptr;
    const int* sample_condition_table = nullptr;
    const std::uint64_t* sample_assignment_table = nullptr;
    const double* sample_probability_table = nullptr;
    const int* sample_event_row_table = nullptr;
};

struct DeviceAuxiliaryView {
    int symbol_count = 0;
    int sampler_count = 0;
    const CudaBlockExpression* block_expression_plans = nullptr;
    const CudaWordMask* block_expression_masks = nullptr;
    const int* block_expression_condition_table = nullptr;
    const CudaConditionSamplerRef* condition_sampler_refs = nullptr;
};

constexpr int kScalarSmallMaxK = 4;
constexpr int kScalarSmallMaxDim = 1 << kScalarSmallMaxK;
constexpr int kScalarSmallMaxScratchDim = kScalarSmallMaxDim >> 1;
constexpr int kScalarSmallMaxSymbolWords = 32;
constexpr int kScalarSmallMaxRecordWords = 8;
constexpr bool kEnableDiagonalRotationSubrunFusion = false;

__device__ void sample_sync() {
    if (blockDim.x <= warpSize) {
        __syncwarp(0xffffffffU);
    } else {
        __syncthreads();
    }
}

__device__ std::uint64_t device_mix_u64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ std::uint64_t device_next_random_u64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    return device_mix_u64(state);
}

struct Philox4x32 {
    std::uint32_t x = 0;
    std::uint32_t y = 0;
    std::uint32_t z = 0;
    std::uint32_t w = 0;
};

__device__ __forceinline__ Philox4x32 device_philox4x32_round(
    Philox4x32 c,
    std::uint32_t key0,
    std::uint32_t key1) {
    constexpr std::uint32_t kPhiloxM0 = 0xD2511F53U;
    constexpr std::uint32_t kPhiloxM1 = 0xCD9E8D57U;
    const auto p0 = static_cast<unsigned long long>(kPhiloxM0) * static_cast<unsigned long long>(c.x);
    const auto p1 = static_cast<unsigned long long>(kPhiloxM1) * static_cast<unsigned long long>(c.z);
    return Philox4x32{
        static_cast<std::uint32_t>(p1 >> 32) ^ c.y ^ key0,
        static_cast<std::uint32_t>(p1),
        static_cast<std::uint32_t>(p0 >> 32) ^ c.w ^ key1,
        static_cast<std::uint32_t>(p0),
    };
}

__device__ __forceinline__ Philox4x32 device_philox4x32_10(
    Philox4x32 c,
    std::uint32_t key0,
    std::uint32_t key1) {
    constexpr std::uint32_t kPhiloxW0 = 0x9E3779B9U;
    constexpr std::uint32_t kPhiloxW1 = 0xBB67AE85U;
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        c = device_philox4x32_round(c, key0, key1);
        key0 += kPhiloxW0;
        key1 += kPhiloxW1;
    }
    return c;
}

__device__ std::uint64_t device_counter_random_u64(
    std::uint64_t seed,
    int shot,
    int sampler_id,
    int draw_index) {
    const auto counter = Philox4x32{
        static_cast<std::uint32_t>(shot),
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(shot) >> 32),
        static_cast<std::uint32_t>(sampler_id),
        static_cast<std::uint32_t>(draw_index),
    };
    const auto out = device_philox4x32_10(
        counter,
        static_cast<std::uint32_t>(seed),
        static_cast<std::uint32_t>(seed >> 32));
    return (static_cast<std::uint64_t>(out.x) << 32) | static_cast<std::uint64_t>(out.z);
}

__device__ double device_rand_float(std::uint64_t& state) {
    return static_cast<double>(device_next_random_u64(state) >> 11) * 0x1.0p-53;
}

__device__ double device_counter_rand_float(
    std::uint64_t seed,
    int shot,
    int sampler_id,
    int& draw_index) {
    const std::uint64_t value = device_counter_random_u64(seed, shot, sampler_id, draw_index);
    ++draw_index;
    return static_cast<double>(value >> 11) * 0x1.0p-53;
}

__device__ bool device_sample_bernoulli(std::uint64_t& state, double p) {
    if (p <= 0.0) {
        return false;
    }
    if (p >= 1.0) {
        return true;
    }
    return device_rand_float(state) < p;
}

__device__ bool device_counter_sample_bernoulli(
    std::uint64_t seed,
    int shot,
    int sampler_id,
    int& draw_index,
    double p) {
    if (p <= 0.0) {
        return false;
    }
    if (p >= 1.0) {
        return true;
    }
    return device_counter_rand_float(seed, shot, sampler_id, draw_index) < p;
}

__device__ int device_sample_categorical_row(std::uint64_t& state, const double* probabilities, int count) {
    const double r = device_rand_float(state);
    double cumulative = 0.0;
    for (int idx = 0; idx < count; ++idx) {
        cumulative += probabilities[idx];
        if (r <= cumulative) {
            return idx;
        }
    }
    return count - 1;
}

__device__ int device_counter_sample_categorical_row(
    std::uint64_t seed,
    int shot,
    int sampler_id,
    int& draw_index,
    const double* probabilities,
    int count) {
    const double r = device_counter_rand_float(seed, shot, sampler_id, draw_index);
    double cumulative = 0.0;
    for (int idx = 0; idx < count; ++idx) {
        cumulative += probabilities[idx];
        if (r <= cumulative) {
            return idx;
        }
    }
    return count - 1;
}

__device__ int device_sample_geometric_gap(std::uint64_t& state, double inverse_log_survival) {
    const double u = fmax(static_cast<double>(device_next_random_u64(state) >> 11) * 0x1.0p-53, 0x1.0p-1022);
    const double gap = floor(log(u) * inverse_log_survival);
    if (!isfinite(gap) || gap >= static_cast<double>(0x7fffffff)) {
        return 0x7fffffff;
    }
    return static_cast<int>(gap);
}

__device__ int device_counter_sample_geometric_gap(
    std::uint64_t seed,
    int shot,
    int sampler_id,
    int& draw_index,
    double inverse_log_survival) {
    const double u = fmax(device_counter_rand_float(seed, shot, sampler_id, draw_index), 0x1.0p-1022);
    const double gap = floor(log(u) * inverse_log_survival);
    if (!isfinite(gap) || gap >= static_cast<double>(0x7fffffff)) {
        return 0x7fffffff;
    }
    return static_cast<int>(gap);
}

__device__ std::size_t device_insert_zero_bit(std::size_t packed, int bit) {
    const std::size_t low_mask = (std::size_t{1} << bit) - std::size_t{1};
    const std::size_t low = packed & low_mask;
    const std::size_t high = packed & ~low_mask;
    return low | (high << 1);
}

__device__ __forceinline__ bool device_get_bit(const std::uint64_t* words, int one_based_id) {
    const int idx = one_based_id - 1;
    return (words[idx >> 6] & (std::uint64_t{1} << (idx & 63))) != 0;
}

__device__ void device_set_bit(std::uint64_t* words, int one_based_id, bool value) {
    if (one_based_id <= 0) {
        return;
    }
    const int idx = one_based_id - 1;
    const std::uint64_t mask = std::uint64_t{1} << (idx & 63);
    if (value) {
        words[idx >> 6] |= mask;
    } else {
        words[idx >> 6] &= ~mask;
    }
}

__device__ void device_atomic_set_bit_true(std::uint64_t* words, int one_based_id) {
    if (one_based_id <= 0) {
        return;
    }
    const int idx = one_based_id - 1;
    const auto mask = static_cast<unsigned long long>(std::uint64_t{1} << (idx & 63));
    atomicOr(reinterpret_cast<unsigned long long*>(words + (idx >> 6)), mask);
}

__device__ void set_assignment_bits(
    const DeviceProgramView& program,
    std::uint64_t* condition_words,
    int condition_offset,
    int nbits,
    std::uint64_t assignment) {
    for (int bit = 0; bit < nbits; ++bit) {
        if (((assignment >> bit) & 1ULL) != 0) {
            const int condition = program.sample_condition_table[condition_offset + bit];
            device_set_bit(condition_words, condition, true);
        }
    }
}

__device__ void set_assignment_bits_atomic(
    const DeviceProgramView& program,
    std::uint64_t* condition_words,
    int condition_offset,
    int nbits,
    std::uint64_t assignment) {
    for (int bit = 0; bit < nbits; ++bit) {
        if (((assignment >> bit) & 1ULL) != 0) {
            const int condition = program.sample_condition_table[condition_offset + bit];
            device_atomic_set_bit_true(condition_words, condition);
        }
    }
}

__device__ void sample_categorical_distribution_counter(
    const DeviceProgramView& program,
    std::uint64_t seed,
    int shot,
    int dist_idx,
    bool use_atomic,
    std::uint64_t* condition_words) {
    const auto& dist = program.categorical_distributions[dist_idx];
    int draw_index = 0;
    const int row = device_counter_sample_categorical_row(
        seed,
        shot,
        dist.sampler_id,
        draw_index,
        program.sample_probability_table + dist.probability_offset,
        dist.row_count);
    const std::uint64_t assignment = program.sample_assignment_table[dist.assignment_offset + row];
    if (use_atomic) {
        set_assignment_bits_atomic(program, condition_words, dist.condition_offset, dist.nbits, assignment);
    } else {
        set_assignment_bits(program, condition_words, dist.condition_offset, dist.nbits, assignment);
    }
}

__device__ void sample_rare_categorical_group_counter(
    const DeviceProgramView& program,
    std::uint64_t seed,
    int shot,
    int group_idx,
    bool use_atomic,
    std::uint64_t* condition_words) {
    const auto& group = program.rare_categorical_groups[group_idx];
    if (group.event_probability <= 0.0 || group.set_count <= 0) {
        return;
    }
    int draw = 0;
    int draw_index = 0;
    while (true) {
        const int gap = device_counter_sample_geometric_gap(
            seed,
            shot,
            group.sampler_id,
            draw_index,
            group.inverse_log_survival);
        if (gap >= group.set_count - draw) {
            break;
        }
        draw += gap;
        const int event_idx = device_counter_sample_categorical_row(
            seed,
            shot,
            group.sampler_id,
            draw_index,
            program.sample_probability_table + group.event_probability_offset,
            group.event_count);
        const int row = program.sample_event_row_table[group.event_row_offset + event_idx];
        const std::uint64_t assignment = program.sample_assignment_table[group.assignment_offset + row];
        const int condition_offset = group.condition_offset + draw * group.nbits;
        if (use_atomic) {
            set_assignment_bits_atomic(program, condition_words, condition_offset, group.nbits, assignment);
        } else {
            set_assignment_bits(program, condition_words, condition_offset, group.nbits, assignment);
        }
        ++draw;
    }
}

__device__ void sample_bernoulli_condition_counter(
    const DeviceProgramView& program,
    std::uint64_t seed,
    int shot,
    int item_idx,
    bool use_atomic,
    std::uint64_t* condition_words) {
    const auto& item = program.bernoulli_conditions[item_idx];
    int draw_index = 0;
    if (device_counter_sample_bernoulli(seed, shot, item.sampler_id, draw_index, item.probability)) {
        if (use_atomic) {
            device_atomic_set_bit_true(condition_words, item.condition);
        } else {
            device_set_bit(condition_words, item.condition, true);
        }
    }
}

__device__ void sample_low_probability_bernoulli_group_counter(
    const DeviceProgramView& program,
    std::uint64_t seed,
    int shot,
    int group_idx,
    bool use_atomic,
    std::uint64_t* condition_words) {
    const auto& group = program.low_probability_bernoulli_groups[group_idx];
    if (group.probability <= 0.0 || group.condition_count <= 0) {
        return;
    }
    int draw = 0;
    int draw_index = 0;
    while (true) {
        const int gap = device_counter_sample_geometric_gap(
            seed,
            shot,
            group.sampler_id,
            draw_index,
            group.inverse_log_survival);
        if (gap >= group.condition_count - draw) {
            break;
        }
        draw += gap;
        const int condition = program.sample_condition_table[group.condition_offset + draw];
        if (use_atomic) {
            device_atomic_set_bit_true(condition_words, condition);
        } else {
            device_set_bit(condition_words, condition, true);
        }
        ++draw;
    }
}

__device__ void sample_exogenous_conditions_parallel_counter(
    const DeviceProgramView& program,
    std::uint64_t seed,
    int shot,
    std::uint64_t* condition_words) {
    for (int dist_idx = threadIdx.x; dist_idx < program.categorical_count; dist_idx += blockDim.x) {
        sample_categorical_distribution_counter(program, seed, shot, dist_idx, true, condition_words);
    }
    for (int group_idx = threadIdx.x; group_idx < program.rare_categorical_count; group_idx += blockDim.x) {
        sample_rare_categorical_group_counter(program, seed, shot, group_idx, true, condition_words);
    }
    for (int idx = threadIdx.x; idx < program.bernoulli_count; idx += blockDim.x) {
        sample_bernoulli_condition_counter(program, seed, shot, idx, true, condition_words);
    }
    for (int group_idx = threadIdx.x; group_idx < program.low_probability_group_count; group_idx += blockDim.x) {
        sample_low_probability_bernoulli_group_counter(program, seed, shot, group_idx, true, condition_words);
    }
}

__device__ void sample_exogenous_conditions(
    const DeviceProgramView& program,
    std::uint64_t& rng_state,
    std::uint64_t* condition_words) {
    for (int dist_idx = 0; dist_idx < program.categorical_count; ++dist_idx) {
        const auto& dist = program.categorical_distributions[dist_idx];
        const int row = device_sample_categorical_row(
            rng_state,
            program.sample_probability_table + dist.probability_offset,
            dist.row_count);
        const std::uint64_t assignment = program.sample_assignment_table[dist.assignment_offset + row];
        set_assignment_bits(program, condition_words, dist.condition_offset, dist.nbits, assignment);
    }

    for (int group_idx = 0; group_idx < program.rare_categorical_count; ++group_idx) {
        const auto& group = program.rare_categorical_groups[group_idx];
        if (group.event_probability <= 0.0 || group.set_count <= 0) {
            continue;
        }
        int draw = 0;
        while (true) {
            const int gap = device_sample_geometric_gap(rng_state, group.inverse_log_survival);
            if (gap >= group.set_count - draw) {
                break;
            }
            draw += gap;
            const int event_idx = device_sample_categorical_row(
                rng_state,
                program.sample_probability_table + group.event_probability_offset,
                group.event_count);
            const int row = program.sample_event_row_table[group.event_row_offset + event_idx];
            const std::uint64_t assignment = program.sample_assignment_table[group.assignment_offset + row];
            const int condition_offset = group.condition_offset + draw * group.nbits;
            set_assignment_bits(program, condition_words, condition_offset, group.nbits, assignment);
            ++draw;
        }
    }

    for (int idx = 0; idx < program.bernoulli_count; ++idx) {
        const auto& item = program.bernoulli_conditions[idx];
        if (device_sample_bernoulli(rng_state, item.probability)) {
            device_set_bit(condition_words, item.condition, true);
        }
    }

    for (int group_idx = 0; group_idx < program.low_probability_group_count; ++group_idx) {
        const auto& group = program.low_probability_bernoulli_groups[group_idx];
        if (group.probability <= 0.0 || group.condition_count <= 0) {
            continue;
        }
        int draw = 0;
        while (true) {
            const int gap = device_sample_geometric_gap(rng_state, group.inverse_log_survival);
            if (gap >= group.condition_count - draw) {
                break;
            }
            draw += gap;
            const int condition = program.sample_condition_table[group.condition_offset + draw];
            device_set_bit(condition_words, condition, true);
            ++draw;
        }
    }
}

__device__ bool sampler_already_sampled(const std::uint64_t* sampled_sampler_words, int sampler_id) {
    if (sampler_id < 0) {
        return true;
    }
    return (sampled_sampler_words[sampler_id >> 6] & (std::uint64_t{1} << (sampler_id & 63))) != 0;
}

__device__ void mark_sampler_sampled(std::uint64_t* sampled_sampler_words, int sampler_id) {
    if (sampler_id < 0) {
        return;
    }
    sampled_sampler_words[sampler_id >> 6] |= std::uint64_t{1} << (sampler_id & 63);
}

__device__ void sample_one_exogenous_sampler_counter(
    const DeviceProgramView& program,
    std::uint64_t seed,
    int shot,
    const CudaConditionSamplerRef& ref,
    std::uint64_t* condition_words) {
    if (ref.kind == static_cast<int>(CudaSamplerKind::Categorical)) {
        sample_categorical_distribution_counter(program, seed, shot, ref.index, false, condition_words);
    } else if (ref.kind == static_cast<int>(CudaSamplerKind::RareCategorical)) {
        sample_rare_categorical_group_counter(program, seed, shot, ref.index, false, condition_words);
    } else if (ref.kind == static_cast<int>(CudaSamplerKind::Bernoulli)) {
        sample_bernoulli_condition_counter(program, seed, shot, ref.index, false, condition_words);
    } else if (ref.kind == static_cast<int>(CudaSamplerKind::LowProbabilityBernoulli)) {
        sample_low_probability_bernoulli_group_counter(program, seed, shot, ref.index, false, condition_words);
    }
}

__device__ void ensure_condition_sampled_lazy(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    std::uint64_t seed,
    int shot,
    int condition,
    std::uint64_t* condition_words,
    std::uint64_t* sampled_sampler_words) {
    if (condition <= 0 || condition > aux.symbol_count || sampled_sampler_words == nullptr) {
        return;
    }
    const auto& ref = aux.condition_sampler_refs[condition - 1];
    if (ref.kind == static_cast<int>(CudaSamplerKind::None) || sampler_already_sampled(sampled_sampler_words, ref.sampler_id)) {
        return;
    }
    mark_sampler_sampled(sampled_sampler_words, ref.sampler_id);
    sample_one_exogenous_sampler_counter(program, seed, shot, ref, condition_words);
}

__device__ void ensure_expression_dependencies_sampled_lazy(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    std::uint64_t seed,
    int shot,
    const CudaExpression& expression,
    std::uint64_t* condition_words,
    std::uint64_t* sampled_sampler_words) {
    for (int idx = 0; idx < expression.residual_count; ++idx) {
        const auto& wm = program.residual_masks[expression.residual_offset + idx];
        std::uint64_t mask = wm.mask;
        while (mask != 0) {
            const int bit = __ffsll(static_cast<long long>(mask)) - 1;
            const int condition = (wm.word << 6) + bit + 1;
            ensure_condition_sampled_lazy(
                program,
                aux,
                seed,
                shot,
                condition,
                condition_words,
                sampled_sampler_words);
            mask &= mask - 1;
        }
    }
}

__device__ CudaComplex cadd(CudaComplex a, CudaComplex b) {
    return CudaComplex{a.re + b.re, a.im + b.im};
}

__device__ CudaComplex cmul(CudaComplex a, CudaComplex b) {
    return CudaComplex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

__device__ CudaComplex cscale(CudaComplex a, CudaReal s) {
    return CudaComplex{a.re * s, a.im * s};
}

__device__ __forceinline__ CudaReal cuda_real_rsqrt(CudaReal x) {
#ifdef SYMFT_CUDA_REAL_DOUBLE
    return static_cast<CudaReal>(1.0) / sqrt(x);
#else
    return rsqrtf(x);
#endif
}

__device__ __forceinline__ bool eval_expression_bit_fast(
    const DeviceProgramView& program,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    const std::uint64_t* condition_words,
    int expression_index) {
    const auto& expression = program.expressions[expression_index];
    bool out = false;
    if (expression.block_expression >= 0 && shot_words != 0) {
        const std::size_t word = static_cast<std::size_t>(shot >> 6);
        const std::uint64_t mask = std::uint64_t{1} << (shot & 63);
        const std::size_t base =
            static_cast<std::size_t>(expression.block_expression) * shot_words + word;
        out = (expression_words[base] & mask) != 0;
    }
    out ^= expression.residual_constant != 0;
    switch (expression.residual_count) {
    case 0:
        return out;
    case 1: {
        const auto& wm = program.residual_masks[expression.residual_offset];
        return out ^ ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
    }
    case 2: {
        const auto& wm0 = program.residual_masks[expression.residual_offset];
        const auto& wm1 = program.residual_masks[expression.residual_offset + 1];
        return out ^
               ((__popcll(condition_words[wm0.word] & wm0.mask) & 1) != 0) ^
               ((__popcll(condition_words[wm1.word] & wm1.mask) & 1) != 0);
    }
    default:
        for (int idx = 0; idx < expression.residual_count; ++idx) {
            const auto& wm = program.residual_masks[expression.residual_offset + idx];
            out ^= ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
        }
        return out;
    }
}

__device__ __forceinline__ bool eval_expression_bit_cached(
    const DeviceProgramView& program,
    const int* block_expression_values,
    const std::uint64_t* condition_words,
    int expression_index) {
    const auto& expression = program.expressions[expression_index];
    bool out = false;
    if (expression.block_expression >= 0) {
        out = block_expression_values[expression.block_expression] != 0;
    }
    out ^= expression.residual_constant != 0;
    switch (expression.residual_count) {
    case 0:
        return out;
    case 1: {
        const auto& wm = program.residual_masks[expression.residual_offset];
        return out ^ ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
    }
    case 2: {
        const auto& wm0 = program.residual_masks[expression.residual_offset];
        const auto& wm1 = program.residual_masks[expression.residual_offset + 1];
        return out ^
               ((__popcll(condition_words[wm0.word] & wm0.mask) & 1) != 0) ^
               ((__popcll(condition_words[wm1.word] & wm1.mask) & 1) != 0);
    }
    default:
        for (int idx = 0; idx < expression.residual_count; ++idx) {
            const auto& wm = program.residual_masks[expression.residual_offset + idx];
            out ^= ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
        }
        return out;
    }
}

__device__ bool eval_block_expression_bit(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int expression_index);

__device__ __forceinline__ bool eval_block_expression_masked_warp(
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int expression_index);

__device__ __forceinline__ bool eval_expression_bit_block_cache_on_demand(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    std::uint64_t* condition_words,
    int* block_expression_values,
    std::uint8_t* block_expression_ready,
    int expression_index) {
    const auto& expression = program.expressions[expression_index];
    bool out = false;
    if (expression.block_expression >= 0) {
        const int block_expression_index = expression.block_expression;
        if (block_expression_ready[block_expression_index] == 0) {
            block_expression_values[block_expression_index] =
                eval_block_expression_bit(program, aux, condition_words, block_expression_index) ? 1 : 0;
            block_expression_ready[block_expression_index] = 1;
        }
        out = block_expression_values[block_expression_index] != 0;
    }
    out ^= expression.residual_constant != 0;
    for (int idx = 0; idx < expression.residual_count; ++idx) {
        const auto& wm = program.residual_masks[expression.residual_offset + idx];
        out ^= ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
    }
    return out;
}

__device__ __forceinline__ bool ensure_block_expression_value_on_demand_coop(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int* block_expression_values,
    std::uint8_t* block_expression_ready,
    int block_expression_index) {
    if (block_expression_ready[block_expression_index] == 0) {
        const auto& expression = aux.block_expression_plans[block_expression_index];
        if (threadIdx.x == 0 && expression.condition_count <= 4) {
            block_expression_values[block_expression_index] =
                eval_block_expression_bit(program, aux, condition_words, block_expression_index) ? 1 : 0;
            block_expression_ready[block_expression_index] = 1;
        } else if (threadIdx.x < warpSize && expression.condition_count > 4) {
            const bool value = eval_block_expression_masked_warp(aux, condition_words, block_expression_index);
            if ((threadIdx.x & (warpSize - 1)) == 0) {
                block_expression_values[block_expression_index] = value ? 1 : 0;
                block_expression_ready[block_expression_index] = 1;
            }
        }
        sample_sync();
    }
    return block_expression_values[block_expression_index] != 0;
}

__device__ __forceinline__ bool eval_expression_bit_block_cache_on_demand_coop(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    std::uint64_t* condition_words,
    int* block_expression_values,
    std::uint8_t* block_expression_ready,
    int expression_index) {
    const auto& expression = program.expressions[expression_index];
    bool out = false;
    if (expression.block_expression >= 0) {
        out = ensure_block_expression_value_on_demand_coop(
            program,
            aux,
            condition_words,
            block_expression_values,
            block_expression_ready,
            expression.block_expression);
    }
    if (threadIdx.x == 0) {
        out ^= expression.residual_constant != 0;
        switch (expression.residual_count) {
        case 0:
            return out;
        case 1: {
            const auto& wm = program.residual_masks[expression.residual_offset];
            return out ^ ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
        }
        case 2: {
            const auto& wm0 = program.residual_masks[expression.residual_offset];
            const auto& wm1 = program.residual_masks[expression.residual_offset + 1];
            return out ^
                   ((__popcll(condition_words[wm0.word] & wm0.mask) & 1) != 0) ^
                   ((__popcll(condition_words[wm1.word] & wm1.mask) & 1) != 0);
        }
        default:
            for (int idx = 0; idx < expression.residual_count; ++idx) {
                const auto& wm = program.residual_masks[expression.residual_offset + idx];
                out ^= ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
            }
            return out;
        }
    }
    return false;
}

__device__ __noinline__ bool eval_expression_bit_lazy(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    std::uint64_t* condition_words,
    int expression_index,
    std::uint64_t exogenous_seed,
    std::uint64_t* sampled_sampler_words) {
    const auto& expression = program.expressions[expression_index];
    ensure_expression_dependencies_sampled_lazy(
        program,
        aux,
        exogenous_seed,
        shot,
        expression,
        condition_words,
        sampled_sampler_words);
    return eval_expression_bit_fast(
        program,
        expression_words,
        shot_words,
        shot,
        condition_words,
        expression_index);
}

__device__ __forceinline__ CudaComplex compact_rotation_coefficient(
    const CudaRotationKernel& kernel,
    std::size_t source,
    bool sign) {
    const bool odd = (__popcll(static_cast<unsigned long long>(source & kernel.zmask)) & 1) != 0;
    const CudaReal direction = sign != odd ? static_cast<CudaReal>(-1.0) : static_cast<CudaReal>(1.0);
    return CudaComplex{direction * kernel.minus_even_re, direction * kernel.minus_even_im};
}


__device__ void apply_rotation_index(
    const DeviceProgramView& program,
    int rotation_index,
    bool sign,
    int k,
    CudaReal* active_re,
    CudaReal* active_im) {
    const auto& kernel = program.rotations[rotation_index];
    const int tid = threadIdx.x;
    const int dim = 1 << k;
    const CudaReal c = kernel.cos_angle;
    if (kernel.kind == CudaRotationKernelKind::Diagonal) {
        for (int basis = tid; basis < dim; basis += blockDim.x) {
            const CudaComplex coefficient = compact_rotation_coefficient(kernel, static_cast<std::size_t>(basis), sign);
            const CudaReal fr = c + coefficient.re;
            const CudaReal fi = coefficient.im;
            const CudaReal r = active_re[basis];
            const CudaReal im = active_im[basis];
            active_re[basis] = fr * r - fi * im;
            active_im[basis] = fr * im + fi * r;
        }
        sample_sync();
        return;
    }
    if (kernel.kind == CudaRotationKernelKind::UniformImagPairs) {
        const CudaReal q = compact_rotation_coefficient(kernel, 0, sign).im;
        for (int pair = tid; pair < kernel.pair_count; pair += blockDim.x) {
            const std::size_t left = device_insert_zero_bit(static_cast<std::size_t>(pair), kernel.pair_bit);
            const std::size_t right = left ^ static_cast<std::size_t>(kernel.xmask);
            const CudaReal r0 = active_re[left];
            const CudaReal i0 = active_im[left];
            const CudaReal r1 = active_re[right];
            const CudaReal i1 = active_im[right];
            active_re[left] = c * r0 - q * i1;
            active_im[left] = c * i0 + q * r1;
            active_re[right] = c * r1 - q * i0;
            active_im[right] = c * i1 + q * r0;
        }
        sample_sync();
        return;
    }
    for (int pair = tid; pair < kernel.pair_count; pair += blockDim.x) {
        const std::size_t left = device_insert_zero_bit(static_cast<std::size_t>(pair), kernel.pair_bit);
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.xmask);
        const bool left_odd = (__popcll(static_cast<unsigned long long>(left & kernel.zmask)) & 1) != 0;
        const CudaReal left_direction = sign != left_odd ? static_cast<CudaReal>(-1.0) : static_cast<CudaReal>(1.0);
        const CudaReal right_direction = kernel.xz_overlap_odd != 0 ? -left_direction : left_direction;
        const CudaComplex left_coefficient{
            left_direction * kernel.minus_even_re,
            left_direction * kernel.minus_even_im,
        };
        const CudaComplex right_coefficient{
            right_direction * kernel.minus_even_re,
            right_direction * kernel.minus_even_im,
        };
        const CudaReal r0 = active_re[left];
        const CudaReal i0 = active_im[left];
        const CudaReal r1 = active_re[right];
        const CudaReal i1 = active_im[right];
        active_re[left] = c * r0 + right_coefficient.re * r1 - right_coefficient.im * i1;
        active_im[left] = c * i0 + right_coefficient.re * i1 + right_coefficient.im * r1;
        active_re[right] = c * r1 + left_coefficient.re * r0 - left_coefficient.im * i0;
        active_im[right] = c * i1 + left_coefficient.re * i0 + left_coefficient.im * r0;
    }
    sample_sync();
}

__device__ void apply_diagonal_rotation_subrun(
    const DeviceProgramView& program,
    const CudaInstruction& instruction,
    int first_run_index,
    int run_count,
    const int* run_signs,
    int k,
    CudaReal* active_re,
    CudaReal* active_im) {
    const int tid = threadIdx.x;
    const int dim = 1 << k;
    for (int basis = tid; basis < dim; basis += blockDim.x) {
        CudaReal factor_re = 1.0f;
        CudaReal factor_im = 0.0f;
        for (int local = 0; local < run_count; ++local) {
            const int idx = first_run_index + local;
            const auto& item = program.rotation_run_items[instruction.rotation_run_offset + idx];
            const auto& kernel = program.rotations[item.rotation];
            const auto coeff = compact_rotation_coefficient(
                kernel, static_cast<std::size_t>(basis), run_signs[idx] != 0);
            const CudaReal next_re = kernel.cos_angle + coeff.re;
            const CudaReal next_im = coeff.im;
            const CudaReal old_re = factor_re;
            const CudaReal old_im = factor_im;
            factor_re = old_re * next_re - old_im * next_im;
            factor_im = old_re * next_im + old_im * next_re;
        }
        const CudaReal r = active_re[basis];
        const CudaReal im = active_im[basis];
        active_re[basis] = factor_re * r - factor_im * im;
        active_im[basis] = factor_re * im + factor_im * r;
    }
    sample_sync();
}

template <bool Lazy, bool BlockCache, bool OnDemandBlockCache>
__device__ void apply_rotation_run(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const CudaInstruction& instruction,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    std::uint64_t* condition_words,
    int* block_expression_values,
    std::uint8_t* block_expression_ready,
    std::uint64_t exogenous_seed,
    std::uint64_t* sampled_sampler_words,
    int k,
    CudaReal* active_re,
    CudaReal* active_im,
    int* run_signs) {
    if constexpr (BlockCache && OnDemandBlockCache) {
        for (int idx = 0; idx < instruction.rotation_run_count; ++idx) {
            const auto& item = program.rotation_run_items[instruction.rotation_run_offset + idx];
            const bool sign = eval_expression_bit_block_cache_on_demand_coop(
                program,
                aux,
                condition_words,
                block_expression_values,
                block_expression_ready,
                item.expression);
            if (threadIdx.x == 0) {
                run_signs[idx] = sign ? 1 : 0;
            }
        }
    } else {
        if constexpr (!Lazy) {
            for (int idx = threadIdx.x; idx < instruction.rotation_run_count; idx += blockDim.x) {
                const auto& item = program.rotation_run_items[instruction.rotation_run_offset + idx];
                bool sign = false;
                if constexpr (BlockCache) {
                    sign = eval_expression_bit_cached(
                        program,
                        block_expression_values,
                        condition_words,
                        item.expression);
                } else {
                    sign = eval_expression_bit_fast(
                        program,
                        expression_words,
                        shot_words,
                        shot,
                        condition_words,
                        item.expression);
                }
                run_signs[idx] = sign ? 1 : 0;
            }
        } else if (threadIdx.x == 0) {
            for (int idx = 0; idx < instruction.rotation_run_count; ++idx) {
                const auto& item = program.rotation_run_items[instruction.rotation_run_offset + idx];
                const bool sign = eval_expression_bit_lazy(
                    program,
                    aux,
                    expression_words,
                    shot_words,
                    shot,
                    condition_words,
                    item.expression,
                    exogenous_seed,
                    sampled_sampler_words);
                run_signs[idx] = sign ? 1 : 0;
            }
        }
    }
    sample_sync();

    for (int idx = 0; idx < instruction.rotation_run_count;) {
        const auto& item = program.rotation_run_items[instruction.rotation_run_offset + idx];
        const auto& kernel = program.rotations[item.rotation];
        if (kEnableDiagonalRotationSubrunFusion && kernel.kind == CudaRotationKernelKind::Diagonal) {
            int end = idx + 1;
            while (end < instruction.rotation_run_count) {
                const auto& next_item = program.rotation_run_items[instruction.rotation_run_offset + end];
                const auto& next_kernel = program.rotations[next_item.rotation];
                if (next_kernel.kind != CudaRotationKernelKind::Diagonal) {
                    break;
                }
                ++end;
            }
            if (end - idx >= 2) {
                apply_diagonal_rotation_subrun(
                    program,
                    instruction,
                    idx,
                    end - idx,
                    run_signs,
                    k,
                    active_re,
                    active_im);
                idx = end;
                continue;
            }
        }
        apply_rotation_index(program, item.rotation, run_signs[idx] != 0, k, active_re, active_im);
        ++idx;
    }
}

__device__ void promote_dormant_rotation(
    const CudaInstruction& instruction,
    bool sign,
    int k,
    CudaReal* active_re,
    CudaReal* active_im) {
    const int tid = threadIdx.x;
    const int dim = 1 << k;
    const CudaReal c = instruction.kernel_cos_angle;
    const CudaReal s = instruction.kernel_sin_angle;
    const CudaReal q = sign ? s : -s;
    for (int basis = tid; basis < dim; basis += blockDim.x) {
        const CudaReal r = active_re[basis];
        const CudaReal im = active_im[basis];
        active_re[basis] = c * r;
        active_im[basis] = c * im;
        active_re[dim + basis] = -q * im;
        active_im[dim + basis] = q * r;
    }
    sample_sync();
}

__device__ __forceinline__ int diagonal_measurement_source(
    const CudaMeasurementKernel& kernel,
    int packed,
    bool branch) {
    const std::size_t base = device_insert_zero_bit(static_cast<std::size_t>(packed), kernel.pivot);
    const int parity = __popcll(static_cast<unsigned long long>(base & kernel.zmask)) & 1;
    const int false_pivot = kernel.diagonal_phase_bit ^ parity;
    const int pivot_value = branch ? (false_pivot ^ 1) : false_pivot;
    return static_cast<int>(base | (static_cast<std::size_t>(pivot_value) << kernel.pivot));
}

__device__ __forceinline__ CudaComplex nondiagonal_measurement_value(
    const CudaMeasurementKernel& kernel,
    const CudaReal* active_re,
    const CudaReal* active_im,
    int packed,
    bool branch) {
    constexpr CudaReal kInvSqrt2 = static_cast<CudaReal>(0.707106781186547524400844362104849039);
    const std::size_t source0 = device_insert_zero_bit(static_cast<std::size_t>(packed), kernel.pivot);
    const std::size_t source1 = source0 ^ static_cast<std::size_t>(kernel.xmask);
    const int phase_sign = (__popcll(static_cast<unsigned long long>(source0 & kernel.zmask)) & 1) != 0 ? -1 : 1;
    const CudaReal qsign = static_cast<CudaReal>(branch ? -phase_sign : phase_sign);
    const CudaReal qre = kInvSqrt2 * qsign * kernel.even_phase_re;
    const CudaReal qim = -kInvSqrt2 * qsign * kernel.even_phase_im;
    const CudaReal r0 = active_re[source0];
    const CudaReal i0 = active_im[source0];
    const CudaReal r1 = active_re[source1];
    const CudaReal i1 = active_im[source1];
    return CudaComplex{
        kInvSqrt2 * r0 + qre * r1 - qim * i1,
        kInvSqrt2 * i0 + qre * i1 + qim * r1,
    };
}

__device__ CudaReal measurement_true_probability(
    const DeviceProgramView& program,
    const CudaMeasurementKernel& kernel,
    const CudaReal* active_re,
    const CudaReal* active_im,
    CudaReal* scratch_re,
    CudaReal* scratch_im,
    CudaReal* reduction) {
    (void)program;
    const int tid = threadIdx.x;
    CudaReal partial = 0.0;
    if (kernel.is_diagonal != 0) {
        for (int idx = tid; idx < kernel.out_dim; idx += blockDim.x) {
            const int source = diagonal_measurement_source(kernel, idx, true);
            const CudaReal re = active_re[source];
            const CudaReal im = active_im[source];
            scratch_re[idx] = re;
            scratch_im[idx] = im;
            partial += re * re + im * im;
        }
    } else {
        for (int idx = tid; idx < kernel.out_dim; idx += blockDim.x) {
            const CudaComplex value = nondiagonal_measurement_value(kernel, active_re, active_im, idx, true);
            scratch_re[idx] = value.re;
            scratch_im[idx] = value.im;
            partial += value.re * value.re + value.im * value.im;
        }
    }

    CudaReal p = 0.0;
    if (blockDim.x <= warpSize) {
        for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffU, partial, offset);
        }
        p = __shfl_sync(0xffffffffU, partial, 0);
    } else {
        reduction[tid] = partial;
        sample_sync();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            sample_sync();
        }
        p = reduction[0];
    }
    if (p < 0.0) {
        p = 0.0;
    } else if (p > 1.0) {
        p = 1.0;
    }
    return p;
}

__device__ void project_cached_true_measurement_branch(
    const CudaMeasurementKernel& kernel,
    CudaReal invnorm,
    CudaReal* active_re,
    CudaReal* active_im,
    CudaReal* scratch_re,
    CudaReal* scratch_im) {
    const int tid = threadIdx.x;
    for (int idx = tid; idx < kernel.out_dim; idx += blockDim.x) {
        active_re[idx] = scratch_re[idx] * invnorm;
        active_im[idx] = scratch_im[idx] * invnorm;
    }
    sample_sync();
}

__device__ void project_measurement_branch(
    const DeviceProgramView& program,
    const CudaMeasurementKernel& kernel,
    bool branch,
    CudaReal invnorm,
    CudaReal* active_re,
    CudaReal* active_im,
    CudaReal* scratch_re,
    CudaReal* scratch_im) {
    (void)program;
    const int tid = threadIdx.x;
    if (kernel.is_diagonal != 0) {
        for (int idx = tid; idx < kernel.out_dim; idx += blockDim.x) {
            const int source = diagonal_measurement_source(kernel, idx, branch);
            scratch_re[idx] = active_re[source] * invnorm;
            scratch_im[idx] = active_im[source] * invnorm;
        }
    } else {
        for (int idx = tid; idx < kernel.out_dim; idx += blockDim.x) {
            const CudaComplex value = nondiagonal_measurement_value(kernel, active_re, active_im, idx, branch);
            scratch_re[idx] = value.re * invnorm;
            scratch_im[idx] = value.im * invnorm;
        }
    }
    sample_sync();
    for (int idx = tid; idx < kernel.out_dim; idx += blockDim.x) {
        active_re[idx] = scratch_re[idx];
        active_im[idx] = scratch_im[idx];
    }
    sample_sync();
}

__device__ void apply_active_measurement(
    const DeviceProgramView& program,
    const CudaInstruction& instruction,
    std::uint64_t& rng_state,
    int& k,
    std::uint64_t* condition_words,
    CudaReal* active_re,
    CudaReal* active_im,
    CudaReal* scratch_re,
    CudaReal* scratch_im,
    CudaReal* reduction) {
    __shared__ int branch_shared;
    __shared__ CudaReal invnorm_shared;

    const auto& kernel = program.measurements[instruction.measurement];
    const CudaReal p_true = measurement_true_probability(
        program,
        kernel,
        active_re,
        active_im,
        scratch_re,
        scratch_im,
        reduction);
    if (threadIdx.x == 0) {
        const bool branch = device_sample_bernoulli(rng_state, static_cast<double>(p_true));
        branch_shared = branch ? 1 : 0;
        device_set_bit(condition_words, instruction.branch_condition, branch);
        const CudaReal probability = branch ? p_true : static_cast<CudaReal>(1.0) - p_true;
        invnorm_shared = probability > 0.0
                             ? cuda_real_rsqrt(probability)
                             : static_cast<CudaReal>(0.0);
    }
    sample_sync();

    if (branch_shared != 0) {
        project_cached_true_measurement_branch(
            kernel,
            invnorm_shared,
            active_re,
            active_im,
            scratch_re,
            scratch_im);
    } else {
        project_measurement_branch(
            program,
            kernel,
            false,
            invnorm_shared,
            active_re,
            active_im,
            scratch_re,
            scratch_im);
    }
    --k;
}

__device__ void write_measurement_record(
    const CudaInstruction& instruction,
    bool outcome,
    std::uint64_t* condition_words,
    std::uint64_t* measurement_words) {
    if (instruction.record > 0) {
        device_set_bit(measurement_words, instruction.record, outcome);
    }
    if (instruction.record_condition > 0) {
        device_set_bit(condition_words, instruction.record_condition, outcome);
    }
}

template <bool Lazy, bool BlockCache, bool OnDemandBlockCache>
__device__ bool detector_outcome(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const CudaInstruction& instruction,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    std::uint64_t* condition_words,
    int* block_expression_values,
    std::uint8_t* block_expression_ready,
    std::uint64_t exogenous_seed,
    std::uint64_t* sampled_sampler_words,
    const std::uint64_t* measurement_words) {
    if (instruction.record_list_count == 0) {
        if constexpr (Lazy) {
            return eval_expression_bit_lazy(
                program,
                aux,
                expression_words,
                shot_words,
                shot,
                condition_words,
                instruction.expression,
                exogenous_seed,
                sampled_sampler_words);
        } else if constexpr (BlockCache) {
            if constexpr (OnDemandBlockCache) {
                return eval_expression_bit_block_cache_on_demand(
                    program,
                    aux,
                    condition_words,
                    block_expression_values,
                    block_expression_ready,
                    instruction.expression);
            } else {
                return eval_expression_bit_cached(
                    program,
                    block_expression_values,
                    condition_words,
                    instruction.expression);
            }
        } else {
            return eval_expression_bit_fast(
                program,
                expression_words,
                shot_words,
                shot,
                condition_words,
                instruction.expression);
        }
    }
    bool out = false;
    for (int idx = 0; idx < instruction.record_list_count; ++idx) {
        const int record = program.record_table[instruction.record_list_offset + idx];
        out ^= device_get_bit(measurement_words, record);
    }
    return out;
}

__device__ bool logical_error_outcome(
    const DeviceProgramView& program,
    const std::uint64_t* measurement_words) {
    bool logical = false;
    for (int group = 0; group < program.logical_group_count; ++group) {
        bool parity = false;
        const int offset = program.logical_group_offsets[group];
        const int count = program.logical_group_sizes[group];
        for (int idx = 0; idx < count; ++idx) {
            parity ^= device_get_bit(measurement_words, program.record_table[offset + idx]);
        }
        logical ^= parity;
    }
    return logical;
}

__device__ __forceinline__ bool scalar_eval_expression_bit_fast(
    const DeviceProgramView& program,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    const std::uint64_t* condition_words,
    int expression_index) {
    return eval_expression_bit_fast(
        program,
        expression_words,
        shot_words,
        shot,
        condition_words,
        expression_index);
}


__device__ void scalar_apply_rotation_index(
    const DeviceProgramView& program,
    int rotation_index,
    bool sign,
    int k,
    CudaReal* active_re,
    CudaReal* active_im) {
    const auto& kernel = program.rotations[rotation_index];
    const int dim = 1 << k;
    const CudaReal c = kernel.cos_angle;
    if (kernel.kind == CudaRotationKernelKind::Diagonal) {
        for (int basis = 0; basis < dim; ++basis) {
            const CudaComplex coefficient = compact_rotation_coefficient(kernel, static_cast<std::size_t>(basis), sign);
            const CudaReal fr = c + coefficient.re;
            const CudaReal fi = coefficient.im;
            const CudaReal r = active_re[basis];
            const CudaReal im = active_im[basis];
            active_re[basis] = fr * r - fi * im;
            active_im[basis] = fr * im + fi * r;
        }
        return;
    }
    if (kernel.kind == CudaRotationKernelKind::UniformImagPairs) {
        const CudaReal q = compact_rotation_coefficient(kernel, 0, sign).im;
        for (int pair = 0; pair < kernel.pair_count; ++pair) {
            const std::size_t left = device_insert_zero_bit(static_cast<std::size_t>(pair), kernel.pair_bit);
            const std::size_t right = left ^ static_cast<std::size_t>(kernel.xmask);
            const CudaReal r0 = active_re[left];
            const CudaReal i0 = active_im[left];
            const CudaReal r1 = active_re[right];
            const CudaReal i1 = active_im[right];
            active_re[left] = c * r0 - q * i1;
            active_im[left] = c * i0 + q * r1;
            active_re[right] = c * r1 - q * i0;
            active_im[right] = c * i1 + q * r0;
        }
        return;
    }
    for (int pair = 0; pair < kernel.pair_count; ++pair) {
        const std::size_t left = device_insert_zero_bit(static_cast<std::size_t>(pair), kernel.pair_bit);
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.xmask);
        const bool left_odd = (__popcll(static_cast<unsigned long long>(left & kernel.zmask)) & 1) != 0;
        const CudaReal left_direction = sign != left_odd ? static_cast<CudaReal>(-1.0) : static_cast<CudaReal>(1.0);
        const CudaReal right_direction = kernel.xz_overlap_odd != 0 ? -left_direction : left_direction;
        const CudaComplex left_coefficient{
            left_direction * kernel.minus_even_re,
            left_direction * kernel.minus_even_im,
        };
        const CudaComplex right_coefficient{
            right_direction * kernel.minus_even_re,
            right_direction * kernel.minus_even_im,
        };
        const CudaReal r0 = active_re[left];
        const CudaReal i0 = active_im[left];
        const CudaReal r1 = active_re[right];
        const CudaReal i1 = active_im[right];
        active_re[left] = c * r0 + right_coefficient.re * r1 - right_coefficient.im * i1;
        active_im[left] = c * i0 + right_coefficient.re * i1 + right_coefficient.im * r1;
        active_re[right] = c * r1 + left_coefficient.re * r0 - left_coefficient.im * i0;
        active_im[right] = c * i1 + left_coefficient.re * i0 + left_coefficient.im * r0;
    }
}

__device__ void scalar_promote_dormant_rotation(
    const CudaInstruction& instruction,
    bool sign,
    int k,
    CudaReal* active_re,
    CudaReal* active_im) {
    const int dim = 1 << k;
    const CudaReal c = instruction.kernel_cos_angle;
    const CudaReal s = instruction.kernel_sin_angle;
    const CudaReal q = sign ? s : -s;
    for (int basis = 0; basis < dim; ++basis) {
        const CudaReal r = active_re[basis];
        const CudaReal im = active_im[basis];
        active_re[basis] = c * r;
        active_im[basis] = c * im;
        active_re[dim + basis] = -q * im;
        active_im[dim + basis] = q * r;
    }
}

__device__ CudaReal scalar_measurement_true_probability(
    const DeviceProgramView& program,
    const CudaMeasurementKernel& kernel,
    const CudaReal* active_re,
    const CudaReal* active_im) {
    (void)program;
    CudaReal p = 0.0f;
    for (int idx = 0; idx < kernel.out_dim; ++idx) {
        CudaComplex value;
        if (kernel.is_diagonal != 0) {
            const int source = diagonal_measurement_source(kernel, idx, true);
            value = CudaComplex{active_re[source], active_im[source]};
        } else {
            value = nondiagonal_measurement_value(kernel, active_re, active_im, idx, true);
        }
        p += value.re * value.re + value.im * value.im;
    }
    if (p < 0.0f) {
        return 0.0f;
    }
    if (p > 1.0f) {
        return 1.0f;
    }
    return p;
}

__device__ void scalar_project_measurement_branch(
    const DeviceProgramView& program,
    const CudaMeasurementKernel& kernel,
    bool branch,
    CudaReal invnorm,
    CudaReal* active_re,
    CudaReal* active_im,
    CudaReal* scratch_re,
    CudaReal* scratch_im) {
    (void)program;
    for (int idx = 0; idx < kernel.out_dim; ++idx) {
        CudaComplex value;
        if (kernel.is_diagonal != 0) {
            const int source = diagonal_measurement_source(kernel, idx, branch);
            value = CudaComplex{active_re[source], active_im[source]};
        } else {
            value = nondiagonal_measurement_value(kernel, active_re, active_im, idx, branch);
        }
        value = cscale(value, invnorm);
        scratch_re[idx] = value.re;
        scratch_im[idx] = value.im;
    }
    for (int idx = 0; idx < kernel.out_dim; ++idx) {
        active_re[idx] = scratch_re[idx];
        active_im[idx] = scratch_im[idx];
    }
}

__device__ void scalar_apply_active_measurement(
    const DeviceProgramView& program,
    const CudaInstruction& instruction,
    std::uint64_t& rng_state,
    int& k,
    std::uint64_t* condition_words,
    CudaReal* active_re,
    CudaReal* active_im,
    CudaReal* scratch_re,
    CudaReal* scratch_im) {
    const auto& kernel = program.measurements[instruction.measurement];
    const CudaReal p_true = scalar_measurement_true_probability(program, kernel, active_re, active_im);
    const bool branch = device_sample_bernoulli(rng_state, p_true);
    device_set_bit(condition_words, instruction.branch_condition, branch);
    const CudaReal probability = branch ? p_true : 1.0f - p_true;
    const CudaReal invnorm = probability > 0.0f ? cuda_real_rsqrt(probability) : 0.0f;
    scalar_project_measurement_branch(
        program,
        kernel,
        branch,
        invnorm,
        active_re,
        active_im,
        scratch_re,
        scratch_im);
    --k;
}

__device__ bool scalar_detector_outcome(
    const DeviceProgramView& program,
    const CudaInstruction& instruction,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    const std::uint64_t* condition_words,
    const std::uint64_t* measurement_words) {
    if (instruction.record_list_count == 0) {
        return scalar_eval_expression_bit_fast(
            program,
            expression_words,
            shot_words,
            shot,
            condition_words,
            instruction.expression);
    }
    bool out = false;
    for (int idx = 0; idx < instruction.record_list_count; ++idx) {
        const int record = program.record_table[instruction.record_list_offset + idx];
        out ^= device_get_bit(measurement_words, record);
    }
    return out;
}

__device__ void scalar_apply_rotation_run(
    const DeviceProgramView& program,
    const CudaInstruction& instruction,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shot,
    const std::uint64_t* condition_words,
    int k,
    CudaReal* active_re,
    CudaReal* active_im) {
    for (int idx = 0; idx < instruction.rotation_run_count; ++idx) {
        const auto& item = program.rotation_run_items[instruction.rotation_run_offset + idx];
        const bool sign = scalar_eval_expression_bit_fast(
            program,
            expression_words,
            shot_words,
            shot,
            condition_words,
            item.expression);
        scalar_apply_rotation_index(program, item.rotation, sign, k, active_re, active_im);
    }
}

extern "C" __global__ void symft_scalar_small_k_sample_kernel(
    DeviceProgramView program,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed,
    int postselect_detectors,
    std::uint8_t* discarded_out,
    std::uint8_t* logical_out) {
    const int shot = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (shot >= shots) {
        return;
    }

    CudaReal active_re[kScalarSmallMaxDim];
    CudaReal active_im[kScalarSmallMaxDim];
    CudaReal scratch_re[kScalarSmallMaxScratchDim];
    CudaReal scratch_im[kScalarSmallMaxScratchDim];
    std::uint64_t condition_words[kScalarSmallMaxSymbolWords];
    std::uint64_t measurement_words[kScalarSmallMaxRecordWords];

    for (int idx = 0; idx < kScalarSmallMaxDim; ++idx) {
        active_re[idx] = 0.0f;
        active_im[idx] = 0.0f;
    }
    for (int idx = 0; idx < kScalarSmallMaxScratchDim; ++idx) {
        scratch_re[idx] = 0.0f;
        scratch_im[idx] = 0.0f;
    }
    for (int idx = 0; idx < program.symbol_words; ++idx) {
        condition_words[idx] = 0;
    }
    for (int idx = 0; idx < program.record_words; ++idx) {
        measurement_words[idx] = 0;
    }
    active_re[0] = 1.0f;

    std::uint64_t rng_state = seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(shot + 1));
    bool detector_any = false;
    int k = program.initial_k;

    for (int instruction_index = 0; instruction_index < program.instruction_count; ++instruction_index) {
        const auto instruction = program.instructions[instruction_index];
        const bool needs_leading_expression =
            instruction.kind == CudaInstructionKind::ActiveRotation ||
            instruction.kind == CudaInstructionKind::PromoteDormantRotation ||
            instruction.kind == CudaInstructionKind::RecordMeasurement ||
            (instruction.kind == CudaInstructionKind::RecordDetector && instruction.record_list_count == 0);
        bool expression_value = false;
        if (needs_leading_expression) {
            expression_value = scalar_eval_expression_bit_fast(
                program,
                expression_words,
                shot_words,
                shot,
                condition_words,
                instruction.expression);
        }

        switch (instruction.kind) {
        case CudaInstructionKind::ActiveRotation:
            scalar_apply_rotation_index(program, instruction.rotation, expression_value, k, active_re, active_im);
            break;
        case CudaInstructionKind::ActiveRotationRun:
            scalar_apply_rotation_run(
                program,
                instruction,
                expression_words,
                shot_words,
                shot,
                condition_words,
                k,
                active_re,
                active_im);
            break;
        case CudaInstructionKind::PromoteDormantRotation:
            scalar_promote_dormant_rotation(instruction, expression_value, k, active_re, active_im);
            ++k;
            break;
        case CudaInstructionKind::RecordMeasurement:
            write_measurement_record(instruction, expression_value, condition_words, measurement_words);
            break;
        case CudaInstructionKind::RecordDetector: {
            const bool outcome = scalar_detector_outcome(
                program,
                instruction,
                expression_words,
                shot_words,
                shot,
                condition_words,
                measurement_words);
            if (outcome) {
                detector_any = true;
                if (postselect_detectors != 0) {
                    discarded_out[shot] = 1;
                    logical_out[shot] = 0;
                    return;
                }
            }
            break;
        }
        case CudaInstructionKind::ActiveMeasurement:
            scalar_apply_active_measurement(
                program,
                instruction,
                rng_state,
                k,
                condition_words,
                active_re,
                active_im,
                scratch_re,
                scratch_im);
            write_measurement_record(
                instruction,
                scalar_eval_expression_bit_fast(
                    program,
                    expression_words,
                    shot_words,
                    shot,
                    condition_words,
                    instruction.expression),
                condition_words,
                measurement_words);
            break;
        case CudaInstructionKind::IntroduceDormantBranch: {
            const bool branch = (device_next_random_u64(rng_state) & 1ULL) != 0;
            device_set_bit(condition_words, instruction.branch_condition, branch);
            const bool outcome = scalar_eval_expression_bit_fast(
                program,
                expression_words,
                shot_words,
                shot,
                condition_words,
                instruction.expression);
            write_measurement_record(instruction, outcome, condition_words, measurement_words);
            break;
        }
        }
    }

    discarded_out[shot] = detector_any ? 1 : 0;
    logical_out[shot] =
        !detector_any && logical_error_outcome(program, measurement_words) ? 1 : 0;
}

__device__ __forceinline__ bool assignment_bit(std::uint64_t assignment, int bit) {
    return ((assignment >> bit) & 1ULL) != 0;
}

__device__ bool sample_condition_value_counter(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    std::uint64_t seed,
    int shot,
    int condition) {
    if (condition <= 0 || condition > aux.symbol_count) {
        return false;
    }
    const auto& ref = aux.condition_sampler_refs[condition - 1];
    if (ref.kind == static_cast<int>(CudaSamplerKind::Categorical)) {
        const auto& dist = program.categorical_distributions[ref.index];
        int draw_index = 0;
        const int row = device_counter_sample_categorical_row(
            seed,
            shot,
            dist.sampler_id,
            draw_index,
            program.sample_probability_table + dist.probability_offset,
            dist.row_count);
        const std::uint64_t assignment = program.sample_assignment_table[dist.assignment_offset + row];
        return assignment_bit(assignment, ref.condition_offset);
    }
    if (ref.kind == static_cast<int>(CudaSamplerKind::RareCategorical)) {
        const auto& group = program.rare_categorical_groups[ref.index];
        if (group.event_probability <= 0.0 || group.set_count <= 0 || group.nbits <= 0) {
            return false;
        }
        const int target_set = ref.condition_offset / group.nbits;
        const int target_bit = ref.condition_offset - target_set * group.nbits;
        int draw = 0;
        int draw_index = 0;
        while (true) {
            const int gap = device_counter_sample_geometric_gap(
                seed,
                shot,
                group.sampler_id,
                draw_index,
                group.inverse_log_survival);
            if (gap >= group.set_count - draw) {
                return false;
            }
            draw += gap;
            const int event_idx = device_counter_sample_categorical_row(
                seed,
                shot,
                group.sampler_id,
                draw_index,
                program.sample_probability_table + group.event_probability_offset,
                group.event_count);
            const int row = program.sample_event_row_table[group.event_row_offset + event_idx];
            if (draw == target_set) {
                const std::uint64_t assignment = program.sample_assignment_table[group.assignment_offset + row];
                return assignment_bit(assignment, target_bit);
            }
            if (draw > target_set) {
                return false;
            }
            ++draw;
        }
    }
    if (ref.kind == static_cast<int>(CudaSamplerKind::Bernoulli)) {
        const auto& item = program.bernoulli_conditions[ref.index];
        int draw_index = 0;
        return device_counter_sample_bernoulli(seed, shot, item.sampler_id, draw_index, item.probability);
    }
    if (ref.kind == static_cast<int>(CudaSamplerKind::LowProbabilityBernoulli)) {
        const auto& group = program.low_probability_bernoulli_groups[ref.index];
        if (group.probability <= 0.0 || group.condition_count <= 0) {
            return false;
        }
        const int target_condition = ref.condition_offset;
        int draw = 0;
        int draw_index = 0;
        while (true) {
            const int gap = device_counter_sample_geometric_gap(
                seed,
                shot,
                group.sampler_id,
                draw_index,
                group.inverse_log_survival);
            if (gap >= group.condition_count - draw) {
                return false;
            }
            draw += gap;
            if (draw == target_condition) {
                return true;
            }
            if (draw > target_condition) {
                return false;
            }
            ++draw;
        }
    }
    return false;
}

__device__ bool eval_block_expression_bit_direct(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    std::uint64_t seed,
    int shot,
    int expression_index) {
    const auto& expression = aux.block_expression_plans[expression_index];
    const int* conditions = aux.block_expression_condition_table + expression.condition_offset;
    bool out = expression.constant != 0;
    switch (expression.condition_count) {
    case 0:
        return out;
    case 1:
        return out ^ sample_condition_value_counter(program, aux, seed, shot, conditions[0]);
    case 2:
        return out ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[0]) ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[1]);
    case 3:
        return out ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[0]) ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[1]) ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[2]);
    case 4:
        return out ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[0]) ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[1]) ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[2]) ^
               sample_condition_value_counter(program, aux, seed, shot, conditions[3]);
    default:
        for (int idx = 0; idx < expression.condition_count; ++idx) {
            out ^= sample_condition_value_counter(program, aux, seed, shot, conditions[idx]);
        }
        return out;
    }
}

__device__ bool eval_block_expression_bit(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int expression_index) {
    (void)program;
    const auto& expression = aux.block_expression_plans[expression_index];
    const int* conditions = aux.block_expression_condition_table + expression.condition_offset;
    bool out = expression.constant != 0;
    switch (expression.condition_count) {
    case 0:
        return out;
    case 1:
        return out ^ device_get_bit(condition_words, conditions[0]);
    case 2:
        return out ^
               device_get_bit(condition_words, conditions[0]) ^
               device_get_bit(condition_words, conditions[1]);
    case 3:
        return out ^
               device_get_bit(condition_words, conditions[0]) ^
               device_get_bit(condition_words, conditions[1]) ^
               device_get_bit(condition_words, conditions[2]);
    case 4:
        return out ^
               device_get_bit(condition_words, conditions[0]) ^
               device_get_bit(condition_words, conditions[1]) ^
               device_get_bit(condition_words, conditions[2]) ^
               device_get_bit(condition_words, conditions[3]);
    default:
        for (int idx = 0; idx < expression.condition_count; ++idx) {
            out ^= device_get_bit(condition_words, conditions[idx]);
        }
        return out;
    }
}

__device__ __forceinline__ bool eval_block_expression_masked(
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int expression_index) {
    const auto& expression = aux.block_expression_plans[expression_index];
    bool out = expression.constant != 0;
    for (int idx = 0; idx < expression.mask_count; ++idx) {
        const auto& wm = aux.block_expression_masks[expression.mask_offset + idx];
        out ^= ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
    }
    return out;
}

__device__ __forceinline__ bool eval_block_expression_masked_warp(
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int expression_index) {
    const auto& expression = aux.block_expression_plans[expression_index];
    const int lane = threadIdx.x & (warpSize - 1);
    bool lane_parity = false;
    for (int idx = lane; idx < expression.mask_count; idx += warpSize) {
        const auto& wm = aux.block_expression_masks[expression.mask_offset + idx];
        lane_parity ^= ((__popcll(condition_words[wm.word] & wm.mask) & 1) != 0);
    }
    const unsigned parity_mask = __ballot_sync(0xffffffffU, lane_parity);
    return ((__popc(parity_mask) & 1) != 0) ^ (expression.constant != 0);
}

__device__ void compute_block_expression_cache(
    const DeviceProgramView& program,
    const DeviceAuxiliaryView& aux,
    const std::uint64_t* condition_words,
    int* block_expression_values) {
    constexpr int kDirectConditionThreshold = 4;
    constexpr int kWarpMaskThreshold = 32;
    const bool use_warp_eval = blockDim.x >= warpSize;

    for (int expression_index = threadIdx.x;
         expression_index < program.block_expression_count;
         expression_index += blockDim.x) {
        const auto& expression = aux.block_expression_plans[expression_index];
        if (!use_warp_eval || expression.mask_count <= kWarpMaskThreshold) {
            const bool value =
                expression.condition_count <= kDirectConditionThreshold
                    ? eval_block_expression_bit(program, aux, condition_words, expression_index)
                    : eval_block_expression_masked(aux, condition_words, expression_index);
            block_expression_values[expression_index] = value ? 1 : 0;
        }
    }

    if (!use_warp_eval) {
        return;
    }

    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int warp_count = blockDim.x >> 5;
    for (int expression_index = warp;
         expression_index < program.block_expression_count;
         expression_index += warp_count) {
        const auto& expression = aux.block_expression_plans[expression_index];
        if (expression.mask_count > kWarpMaskThreshold) {
            const bool value = eval_block_expression_masked_warp(aux, condition_words, expression_index);
            if (lane == 0) {
                block_expression_values[expression_index] = value ? 1 : 0;
            }
        }
    }
}

extern "C" __global__ void symft_generate_expression_words_kernel(
    DeviceProgramView program,
    DeviceAuxiliaryView aux,
    std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed) {
    const int shot = static_cast<int>(blockIdx.x);
    if (shot >= shots) {
        return;
    }

    extern __shared__ unsigned char expression_shared_raw[];
    auto* condition_words = reinterpret_cast<std::uint64_t*>(expression_shared_raw);
    int* block_expression_values = reinterpret_cast<int*>(condition_words + program.symbol_words);
    for (int idx = threadIdx.x; idx < program.symbol_words; idx += blockDim.x) {
        condition_words[idx] = 0;
    }
    sample_sync();

    sample_exogenous_conditions_parallel_counter(program, seed, shot, condition_words);
    sample_sync();

    compute_block_expression_cache(program, aux, condition_words, block_expression_values);
    sample_sync();

    const std::size_t shot_word = static_cast<std::size_t>(shot >> 6);
    const auto mask = static_cast<unsigned long long>(std::uint64_t{1} << (shot & 63));
    for (int expression_index = threadIdx.x;
         expression_index < program.block_expression_count;
         expression_index += blockDim.x) {
        if (block_expression_values[expression_index] != 0) {
            const std::size_t offset =
                static_cast<std::size_t>(expression_index) * shot_words + shot_word;
            atomicOr(reinterpret_cast<unsigned long long*>(expression_words + offset), mask);
        }
    }
}

extern "C" __global__ void symft_generate_expression_words_by_word_kernel(
    DeviceProgramView program,
    DeviceAuxiliaryView aux,
    std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed) {
    const int lane = threadIdx.x;
    if (lane >= 32) {
        return;
    }
    const auto shot_word = static_cast<std::size_t>(blockIdx.x);
    const int expression_index = static_cast<int>(blockIdx.y);
    if (expression_index >= program.block_expression_count || shot_word >= shot_words) {
        return;
    }

    const int shot0 = static_cast<int>(shot_word << 6) + lane;
    const bool value0 = shot0 < shots &&
                        eval_block_expression_bit_direct(
                            program,
                            aux,
                            seed,
                            shot0,
                            expression_index);
    const unsigned low = __ballot_sync(0xffffffffU, value0);

    const int shot1 = shot0 + 32;
    const bool value1 = shot1 < shots &&
                        eval_block_expression_bit_direct(
                            program,
                            aux,
                            seed,
                            shot1,
                            expression_index);
    const unsigned high = __ballot_sync(0xffffffffU, value1);

    if (lane == 0) {
        expression_words[static_cast<std::size_t>(expression_index) * shot_words + shot_word] =
            static_cast<std::uint64_t>(low) | (static_cast<std::uint64_t>(high) << 32);
    }
}

template <bool Lazy, bool BlockCache, bool OnDemandBlockCache>
__device__ __forceinline__ void symft_persistent_sample_body(
    DeviceProgramView program,
    DeviceAuxiliaryView aux,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed,
    int postselect_detectors,
    int sample_exogenous_on_device,
    int sampled_sampler_word_count,
    std::uint8_t* discarded_out,
    std::uint8_t* logical_out) {
    const int shot = static_cast<int>(blockIdx.x);
    if (shot >= shots) {
        return;
    }

    extern __shared__ unsigned char persistent_shared_raw[];
    const int max_dim = 1 << program.max_k;
    const int scratch_dim = max_dim > 1 ? (max_dim >> 1) : 1;
    // One CTA owns one complete shot. Active amplitudes stay in shared memory
    // across the whole instruction stream; only final counts leave the GPU.
    CudaReal* active_re = reinterpret_cast<CudaReal*>(persistent_shared_raw);
    CudaReal* active_im = active_re + max_dim;
    CudaReal* scratch_re = active_im + max_dim;
    CudaReal* scratch_im = scratch_re + scratch_dim;
    auto* condition_words = reinterpret_cast<std::uint64_t*>(scratch_im + scratch_dim);
    auto* measurement_words = condition_words + program.symbol_words;
    auto* sampled_sampler_words = measurement_words + program.record_words;
    CudaReal* reduction = reinterpret_cast<CudaReal*>(sampled_sampler_words + sampled_sampler_word_count);
    int* rotation_run_signs = reinterpret_cast<int*>(reduction + blockDim.x);
    int* block_expression_values = rotation_run_signs + program.max_rotation_run_length;
    auto* block_expression_ready = reinterpret_cast<std::uint8_t*>(
        block_expression_values + program.block_expression_count);
    const std::uint64_t exogenous_seed = device_mix_u64(seed ^ 0x51ed5eed1234abcdULL);

    __shared__ std::uint64_t rng_state;
    __shared__ int detector_any;
    __shared__ int dead;
    __shared__ int expression_shared;

    const int initial_dim = 1 << program.initial_k;
    for (int idx = threadIdx.x; idx < initial_dim; idx += blockDim.x) {
        active_re[idx] = idx == 0 ? 1.0 : 0.0;
        active_im[idx] = 0.0;
    }
    for (int idx = threadIdx.x; idx < program.symbol_words; idx += blockDim.x) {
        condition_words[idx] = 0;
    }
    for (int idx = threadIdx.x; idx < program.record_words; idx += blockDim.x) {
        measurement_words[idx] = 0;
    }
    if constexpr (Lazy) {
        for (int idx = threadIdx.x; idx < sampled_sampler_word_count; idx += blockDim.x) {
            sampled_sampler_words[idx] = 0;
        }
    }
    if constexpr (OnDemandBlockCache) {
        for (int idx = threadIdx.x; idx < sampled_sampler_word_count; idx += blockDim.x) {
            sampled_sampler_words[idx] = 0;
        }
        for (int idx = threadIdx.x; idx < program.block_expression_count; idx += blockDim.x) {
            block_expression_ready[idx] = 0;
        }
    }
    sample_sync();
    if (threadIdx.x == 0) {
        rng_state = seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(shot + 1));
        detector_any = 0;
        dead = 0;
    }
    sample_sync();

    if (sample_exogenous_on_device != 0) {
        if constexpr (BlockCache && OnDemandBlockCache) {
            sample_exogenous_conditions_parallel_counter(program, exogenous_seed, shot, condition_words);
        } else if (threadIdx.x == 0) {
            sample_exogenous_conditions(program, rng_state, condition_words);
        }
    }
    sample_sync();

    if constexpr (BlockCache && !OnDemandBlockCache) {
        compute_block_expression_cache(program, aux, condition_words, block_expression_values);
        sample_sync();
    }

    int k = program.initial_k;
    for (int instruction_index = 0; instruction_index < program.instruction_count; ++instruction_index) {
        const auto instruction = program.instructions[instruction_index];
        const bool needs_leading_expression =
            instruction.kind == CudaInstructionKind::ActiveRotation ||
            instruction.kind == CudaInstructionKind::PromoteDormantRotation ||
            instruction.kind == CudaInstructionKind::RecordMeasurement ||
            (instruction.kind == CudaInstructionKind::RecordDetector && instruction.record_list_count == 0);
        if (needs_leading_expression) {
            if constexpr (BlockCache && OnDemandBlockCache) {
                const bool expression_value = eval_expression_bit_block_cache_on_demand_coop(
                    program,
                    aux,
                    condition_words,
                    block_expression_values,
                    block_expression_ready,
                    instruction.expression);
                if (threadIdx.x == 0) {
                    expression_shared = expression_value ? 1 : 0;
                }
            } else {
                if (threadIdx.x == 0) {
                    bool expression_value = false;
                    if constexpr (Lazy) {
                        expression_value = eval_expression_bit_lazy(
                            program,
                            aux,
                            expression_words,
                            shot_words,
                            shot,
                            condition_words,
                            instruction.expression,
                            exogenous_seed,
                            sampled_sampler_words);
                    } else if constexpr (BlockCache) {
                        expression_value = eval_expression_bit_cached(
                            program,
                            block_expression_values,
                            condition_words,
                            instruction.expression);
                    } else {
                        expression_value = eval_expression_bit_fast(
                            program,
                            expression_words,
                            shot_words,
                            shot,
                            condition_words,
                            instruction.expression);
                    }
                    expression_shared = expression_value ? 1 : 0;
                }
            }
            sample_sync();
        }

        switch (instruction.kind) {
        case CudaInstructionKind::ActiveRotation:
            apply_rotation_index(program, instruction.rotation, expression_shared != 0, k, active_re, active_im);
            break;
        case CudaInstructionKind::ActiveRotationRun:
            apply_rotation_run<Lazy, BlockCache, OnDemandBlockCache>(
                program,
                aux,
                instruction,
                expression_words,
                shot_words,
                shot,
                condition_words,
                block_expression_values,
                block_expression_ready,
                exogenous_seed,
                sampled_sampler_words,
                k,
                active_re,
                active_im,
                rotation_run_signs);
            break;
        case CudaInstructionKind::PromoteDormantRotation:
            promote_dormant_rotation(instruction, expression_shared != 0, k, active_re, active_im);
            ++k;
            break;
        case CudaInstructionKind::RecordMeasurement:
            if (threadIdx.x == 0) {
                write_measurement_record(instruction, expression_shared != 0, condition_words, measurement_words);
            }
            sample_sync();
            break;
        case CudaInstructionKind::RecordDetector:
            if (threadIdx.x == 0) {
                const bool outcome =
                    instruction.record_list_count == 0
                        ? expression_shared != 0
                        : detector_outcome<Lazy, BlockCache, OnDemandBlockCache>(
                              program,
                              aux,
                              instruction,
                              expression_words,
                              shot_words,
                              shot,
                              condition_words,
                              block_expression_values,
                              block_expression_ready,
                              exogenous_seed,
                              sampled_sampler_words,
                              measurement_words);
                if (outcome) {
                    detector_any = 1;
                    if (postselect_detectors != 0) {
                        dead = 1;
                    }
                }
            }
            sample_sync();
            if (dead != 0) {
                if (threadIdx.x == 0) {
                    discarded_out[shot] = 1;
                    logical_out[shot] = 0;
                }
                return;
            }
            break;
        case CudaInstructionKind::ActiveMeasurement:
            apply_active_measurement(
                program,
                instruction,
                rng_state,
                k,
                condition_words,
                active_re,
                active_im,
                scratch_re,
                scratch_im,
                reduction);
            if constexpr (BlockCache && OnDemandBlockCache) {
                const bool outcome = eval_expression_bit_block_cache_on_demand_coop(
                    program,
                    aux,
                    condition_words,
                    block_expression_values,
                    block_expression_ready,
                    instruction.expression);
                if (threadIdx.x == 0) {
                    write_measurement_record(instruction, outcome, condition_words, measurement_words);
                }
            } else {
                if (threadIdx.x == 0) {
                    bool outcome = false;
                    if constexpr (Lazy) {
                        outcome = eval_expression_bit_lazy(
                            program,
                            aux,
                            expression_words,
                            shot_words,
                            shot,
                            condition_words,
                            instruction.expression,
                            exogenous_seed,
                            sampled_sampler_words);
                    } else if constexpr (BlockCache) {
                        outcome = eval_expression_bit_cached(
                            program,
                            block_expression_values,
                            condition_words,
                            instruction.expression);
                    } else {
                        outcome = eval_expression_bit_fast(
                            program,
                            expression_words,
                            shot_words,
                            shot,
                            condition_words,
                            instruction.expression);
                    }
                    write_measurement_record(instruction, outcome, condition_words, measurement_words);
                }
            }
            sample_sync();
            break;
        case CudaInstructionKind::IntroduceDormantBranch:
            if constexpr (BlockCache && OnDemandBlockCache) {
                if (threadIdx.x == 0) {
                    const bool branch = (device_next_random_u64(rng_state) & 1ULL) != 0;
                    device_set_bit(condition_words, instruction.branch_condition, branch);
                }
                sample_sync();
                const bool outcome = eval_expression_bit_block_cache_on_demand_coop(
                    program,
                    aux,
                    condition_words,
                    block_expression_values,
                    block_expression_ready,
                    instruction.expression);
                if (threadIdx.x == 0) {
                    write_measurement_record(instruction, outcome, condition_words, measurement_words);
                }
            } else {
                if (threadIdx.x == 0) {
                    const bool branch = (device_next_random_u64(rng_state) & 1ULL) != 0;
                    device_set_bit(condition_words, instruction.branch_condition, branch);
                    bool outcome = false;
                    if constexpr (Lazy) {
                        outcome = eval_expression_bit_lazy(
                            program,
                            aux,
                            expression_words,
                            shot_words,
                            shot,
                            condition_words,
                            instruction.expression,
                            exogenous_seed,
                            sampled_sampler_words);
                    } else if constexpr (BlockCache) {
                        outcome = eval_expression_bit_cached(
                            program,
                            block_expression_values,
                            condition_words,
                            instruction.expression);
                    } else {
                        outcome = eval_expression_bit_fast(
                            program,
                            expression_words,
                            shot_words,
                            shot,
                            condition_words,
                            instruction.expression);
                    }
                    write_measurement_record(instruction, outcome, condition_words, measurement_words);
                }
            }
            sample_sync();
            break;
        }
    }

    if (threadIdx.x == 0) {
        const bool discarded = detector_any != 0;
        discarded_out[shot] = discarded ? 1 : 0;
        logical_out[shot] = !discarded && logical_error_outcome(program, measurement_words) ? 1 : 0;
    }
}

extern "C" __global__ void symft_persistent_sample_kernel_fast(
    DeviceProgramView program,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed,
    int postselect_detectors,
    int sample_exogenous_on_device,
    std::uint8_t* discarded_out,
    std::uint8_t* logical_out) {
    DeviceAuxiliaryView aux;
    symft_persistent_sample_body<false, false, false>(
        program,
        aux,
        expression_words,
        shot_words,
        shots,
        seed,
        postselect_detectors,
        sample_exogenous_on_device,
        0,
        discarded_out,
        logical_out);
}

extern "C" __global__ void symft_persistent_sample_kernel_cached(
    DeviceProgramView program,
    DeviceAuxiliaryView aux,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed,
    int postselect_detectors,
    int sampled_sampler_word_count,
    std::uint8_t* discarded_out,
    std::uint8_t* logical_out) {
    symft_persistent_sample_body<false, true, true>(
        program,
        aux,
        expression_words,
        shot_words,
        shots,
        seed,
        postselect_detectors,
        1,
        0,
        discarded_out,
        logical_out);
}

extern "C" __global__ void symft_persistent_sample_kernel_lazy(
    DeviceProgramView program,
    DeviceAuxiliaryView aux,
    const std::uint64_t* expression_words,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed,
    int postselect_detectors,
    int sample_exogenous_on_device,
    int sampled_sampler_word_count,
    std::uint8_t* discarded_out,
    std::uint8_t* logical_out) {
    symft_persistent_sample_body<true, false, false>(
        program,
        aux,
        expression_words,
        shot_words,
        shots,
        seed,
        postselect_detectors,
        sample_exogenous_on_device,
        sampled_sampler_word_count,
        discarded_out,
        logical_out);
}

} // namespace

struct CudaRuntimeProgram::Impl {
    CudaProgramData host;
    DeviceArray<CudaInstruction> instructions;
    DeviceArray<CudaRotationRunItem> rotation_run_items;
    DeviceArray<CudaExpression> expressions;
    DeviceArray<CudaBlockExpression> block_expression_plans;
    DeviceArray<CudaWordMask> block_expression_masks;
    DeviceArray<CudaWordMask> residual_masks;
    DeviceArray<CudaRotationKernel> rotations;
    DeviceArray<CudaMeasurementKernel> measurements;
    DeviceArray<CudaComplex> complex_table;
    DeviceArray<CudaReal> real_table;
    DeviceArray<int> source_table;
    DeviceArray<int> record_table;
    DeviceArray<int> block_expression_condition_table;
    DeviceArray<int> logical_group_offsets;
    DeviceArray<int> logical_group_sizes;
    DeviceArray<CudaCategoricalDistribution> categorical_distributions;
    DeviceArray<CudaRareCategoricalGroup> rare_categorical_groups;
    DeviceArray<CudaBernoulliCondition> bernoulli_conditions;
    DeviceArray<CudaBernoulliGroup> low_probability_bernoulli_groups;
    DeviceArray<int> sample_condition_table;
    DeviceArray<std::uint64_t> sample_assignment_table;
    DeviceArray<double> sample_probability_table;
    DeviceArray<int> sample_event_row_table;
    DeviceArray<CudaConditionSamplerRef> condition_sampler_refs;
    DeviceArray<std::uint64_t> expression_words;
    DeviceArray<std::uint8_t> discarded_flags;
    DeviceArray<std::uint8_t> logical_flags;
    std::vector<std::uint8_t> host_discarded;
    std::vector<std::uint8_t> host_logical;

    explicit Impl(const CudaProgramData& program) : host(program) {
        instructions.upload(host.instructions);
        rotation_run_items.upload(host.rotation_run_items);
        expressions.upload(host.expressions);
        block_expression_plans.upload(host.block_expression_plans);
        block_expression_masks.upload(host.block_expression_masks);
        residual_masks.upload(host.residual_masks);
        rotations.upload(host.rotations);
        measurements.upload(host.measurements);
        complex_table.upload(host.complex_table);
        real_table.upload(host.real_table);
        source_table.upload(host.source_table);
        record_table.upload(host.record_table);
        block_expression_condition_table.upload(host.block_expression_condition_table);
        logical_group_offsets.upload(host.logical_group_offsets);
        logical_group_sizes.upload(host.logical_group_sizes);
        categorical_distributions.upload(host.categorical_distributions);
        rare_categorical_groups.upload(host.rare_categorical_groups);
        bernoulli_conditions.upload(host.bernoulli_conditions);
        low_probability_bernoulli_groups.upload(host.low_probability_bernoulli_groups);
        sample_condition_table.upload(host.sample_condition_table);
        sample_assignment_table.upload(host.sample_assignment_table);
        sample_probability_table.upload(host.sample_probability_table);
        sample_event_row_table.upload(host.sample_event_row_table);
        condition_sampler_refs.upload(host.condition_sampler_refs);
    }

    DeviceProgramView view() const {
        DeviceProgramView out;
        out.initial_k = host.initial_k;
        out.max_k = host.max_k;
        out.symbol_words = host.symbol_words;
        out.record_words = host.record_words;
        out.instruction_count = static_cast<int>(host.instructions.size());
        out.expression_count = static_cast<int>(host.expressions.size());
        out.rotation_count = static_cast<int>(host.rotations.size());
        out.measurement_count = static_cast<int>(host.measurements.size());
        out.block_expression_count = host.block_expression_count;
        out.max_rotation_run_length = host.max_rotation_run_length;
        out.logical_group_count = static_cast<int>(host.logical_group_offsets.size());
        out.categorical_count = static_cast<int>(host.categorical_distributions.size());
        out.rare_categorical_count = static_cast<int>(host.rare_categorical_groups.size());
        out.bernoulli_count = static_cast<int>(host.bernoulli_conditions.size());
        out.low_probability_group_count = static_cast<int>(host.low_probability_bernoulli_groups.size());
        out.instructions = instructions.ptr;
        out.rotation_run_items = rotation_run_items.ptr;
        out.expressions = expressions.ptr;
        out.residual_masks = residual_masks.ptr;
        out.rotations = rotations.ptr;
        out.measurements = measurements.ptr;
        out.complex_table = complex_table.ptr;
        out.real_table = real_table.ptr;
        out.source_table = source_table.ptr;
        out.record_table = record_table.ptr;
        out.logical_group_offsets = logical_group_offsets.ptr;
        out.logical_group_sizes = logical_group_sizes.ptr;
        out.categorical_distributions = categorical_distributions.ptr;
        out.rare_categorical_groups = rare_categorical_groups.ptr;
        out.bernoulli_conditions = bernoulli_conditions.ptr;
        out.low_probability_bernoulli_groups = low_probability_bernoulli_groups.ptr;
        out.sample_condition_table = sample_condition_table.ptr;
        out.sample_assignment_table = sample_assignment_table.ptr;
        out.sample_probability_table = sample_probability_table.ptr;
        out.sample_event_row_table = sample_event_row_table.ptr;
        return out;
    }

    DeviceAuxiliaryView aux_view() const {
        DeviceAuxiliaryView out;
        out.symbol_count = host.symbol_count;
        out.sampler_count = host.sampler_count;
        out.block_expression_plans = block_expression_plans.ptr;
        out.block_expression_masks = block_expression_masks.ptr;
        out.block_expression_condition_table = block_expression_condition_table.ptr;
        out.condition_sampler_refs = condition_sampler_refs.ptr;
        return out;
    }
};

CudaRuntimeProgram::CudaRuntimeProgram(const CudaProgramData& program)
    : impl_(std::make_unique<Impl>(program)) {}

CudaRuntimeProgram::~CudaRuntimeProgram() = default;

CudaRuntimeProgram::CudaRuntimeProgram(CudaRuntimeProgram&& other) noexcept = default;

CudaRuntimeProgram& CudaRuntimeProgram::operator=(CudaRuntimeProgram&& other) noexcept = default;

CudaKernelRunResult CudaRuntimeProgram::run(
    const std::uint64_t* expression_words_host,
    std::size_t expression_word_count,
    std::size_t shot_words,
    int shots,
    std::uint64_t seed,
    const CudaLaunchOptions& options) {
    if (shots < 0) {
        throw Error("CUDA shot count must be nonnegative");
    }
    CudaKernelRunResult result;
    if (shots == 0) {
        return result;
    }
    const int direct_device_modes =
        (options.sample_exogenous_on_device ? 1 : 0) +
        (options.lazy_exogenous_on_device ? 1 : 0) +
        (options.on_demand_expression_blocks ? 1 : 0);
    if (direct_device_modes > 1) {
        throw Error("CUDA direct device exogenous modes are mutually exclusive");
    }
    if (options.generate_expressions_on_device &&
        direct_device_modes != 0) {
        throw Error("CUDA expression generation is mutually exclusive with direct device exogenous sampling");
    }

    const bool uses_host_expression_words =
        !options.sample_exogenous_on_device &&
        !options.lazy_exogenous_on_device &&
        !options.on_demand_expression_blocks &&
        !options.generate_expressions_on_device;
    if (options.on_demand_expression_blocks && impl_->host.block_expression_count <= 0) {
        throw Error("CUDA on-demand expression mode requires block expressions");
    }
    if (options.generate_expressions_on_device) {
        if (impl_->host.block_expression_count <= 0) {
            throw Error("CUDA generated expression mode requires block expressions");
        }
        const std::size_t expected_shot_words = static_cast<std::size_t>((shots + 63) >> 6);
        if (shot_words != expected_shot_words) {
            throw Error("CUDA generated expression mode received an invalid shot_words value");
        }
        expression_word_count = static_cast<std::size_t>(impl_->host.block_expression_count) * shot_words;
    } else if (uses_host_expression_words) {
        if (expression_word_count != static_cast<std::size_t>(impl_->host.block_expression_count) * shot_words) {
            throw Error("CUDA expression block shape does not match program");
        }
    } else if (expression_word_count != 0 || shot_words != 0) {
        throw Error("CUDA direct device exogenous modes do not accept expression words");
    }

    const int threads =
        options.threads_per_block > 0 ? options.threads_per_block : (options.generate_expressions_on_device ? 128 : 32);
    if (threads < 32) {
        throw Error("CUDA threads_per_block must be at least one warp (32 threads)");
    }
    if ((threads & (threads - 1)) != 0) {
        throw Error("CUDA threads_per_block must be a power of two");
    }
    const int sampler_threads = options.generate_expressions_on_device ? 32 : threads;
    const int max_dim = 1 << impl_->host.max_k;
    const bool use_scalar_small_k_sampler =
        options.generate_expressions_on_device &&
        impl_->host.max_k <= kScalarSmallMaxK &&
        impl_->host.symbol_words <= kScalarSmallMaxSymbolWords &&
        impl_->host.record_words <= kScalarSmallMaxRecordWords;
    const int sampled_sampler_word_count =
        options.lazy_exogenous_on_device ? ((impl_->host.sampler_count + 63) >> 6) : 0;
    const bool use_on_demand_block_cache = options.on_demand_expression_blocks;
    const std::size_t sampler_shared_bytes =
        (2 * static_cast<std::size_t>(max_dim) +
         2 * static_cast<std::size_t>(std::max(1, max_dim >> 1))) * sizeof(CudaReal) +
        static_cast<std::size_t>(impl_->host.symbol_words + impl_->host.record_words) * sizeof(std::uint64_t) +
        static_cast<std::size_t>(sampled_sampler_word_count) * sizeof(std::uint64_t) +
        static_cast<std::size_t>(sampler_threads) * sizeof(CudaReal) +
        static_cast<std::size_t>(impl_->host.max_rotation_run_length) * sizeof(int) +
        (use_on_demand_block_cache
             ? static_cast<std::size_t>(impl_->host.block_expression_count) * (sizeof(int) + sizeof(std::uint8_t))
             : 0);
    const std::size_t generator_shared_bytes =
        options.generate_expressions_on_device
            ? static_cast<std::size_t>(impl_->host.symbol_words) * sizeof(std::uint64_t) +
                  static_cast<std::size_t>(impl_->host.block_expression_count) * sizeof(int)
            : 0;

    int device = 0;
    check_cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    const std::size_t shared_limit = static_cast<std::size_t>(prop.sharedMemPerBlockOptin);
    if (sampler_shared_bytes > shared_limit || generator_shared_bytes > shared_limit) {
        throw Error("CUDA sampler needs more shared memory than this device allows for the program max_k");
    }
    if (sampler_shared_bytes > static_cast<std::size_t>(prop.sharedMemPerBlock)) {
        if (options.lazy_exogenous_on_device) {
            check_cuda(
                cudaFuncSetAttribute(
                    symft_persistent_sample_kernel_lazy,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(sampler_shared_bytes)),
                "cudaFuncSetAttribute");
        } else if (options.on_demand_expression_blocks) {
            check_cuda(
                cudaFuncSetAttribute(
                    symft_persistent_sample_kernel_cached,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(sampler_shared_bytes)),
                "cudaFuncSetAttribute");
        } else {
            check_cuda(
                cudaFuncSetAttribute(
                    symft_persistent_sample_kernel_fast,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(sampler_shared_bytes)),
                "cudaFuncSetAttribute");
        }
    }
    if (generator_shared_bytes > static_cast<std::size_t>(prop.sharedMemPerBlock)) {
        check_cuda(
            cudaFuncSetAttribute(
                symft_generate_expression_words_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(generator_shared_bytes)),
            "cudaFuncSetAttribute");
    }
    if (uses_host_expression_words) {
        impl_->expression_words.upload_raw(expression_words_host, expression_word_count);
    } else if (options.generate_expressions_on_device) {
        impl_->expression_words.resize_uninitialized(expression_word_count);
    } else {
        impl_->expression_words.resize_uninitialized(0);
    }
    impl_->discarded_flags.reserve(static_cast<std::size_t>(shots));
    impl_->logical_flags.reserve(static_cast<std::size_t>(shots));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate");
    check_cuda(cudaEventRecord(start), "cudaEventRecord");
    if (options.generate_expressions_on_device) {
        check_cuda(
            cudaMemset(
                impl_->expression_words.ptr,
                0,
                expression_word_count * sizeof(std::uint64_t)),
            "cudaMemset expression words");
        symft_generate_expression_words_kernel<<<shots, threads, generator_shared_bytes>>>(
            impl_->view(),
            impl_->aux_view(),
            impl_->expression_words.ptr,
            shot_words,
            shots,
            host_mix_u64(seed ^ 0x51ed5eed1234abcdULL));
        check_cuda(cudaGetLastError(), "CUDA expression generation launch");
        if (use_scalar_small_k_sampler) {
            const int scalar_blocks = (shots + threads - 1) / threads;
            symft_scalar_small_k_sample_kernel<<<scalar_blocks, threads>>>(
                impl_->view(),
                impl_->expression_words.ptr,
                shot_words,
                shots,
                seed,
                options.postselect_detectors ? 1 : 0,
                impl_->discarded_flags.ptr,
                impl_->logical_flags.ptr);
        } else {
            symft_persistent_sample_kernel_fast<<<shots, sampler_threads, sampler_shared_bytes>>>(
                impl_->view(),
                impl_->expression_words.ptr,
                shot_words,
                shots,
                seed,
                options.postselect_detectors ? 1 : 0,
                0,
                impl_->discarded_flags.ptr,
                impl_->logical_flags.ptr);
        }
    } else if (options.on_demand_expression_blocks) {
        symft_persistent_sample_kernel_cached<<<shots, threads, sampler_shared_bytes>>>(
            impl_->view(),
            impl_->aux_view(),
            impl_->expression_words.ptr,
            shot_words,
            shots,
            seed,
            options.postselect_detectors ? 1 : 0,
            0,
            impl_->discarded_flags.ptr,
            impl_->logical_flags.ptr);
    } else if (options.lazy_exogenous_on_device) {
        symft_persistent_sample_kernel_lazy<<<shots, threads, sampler_shared_bytes>>>(
            impl_->view(),
            impl_->aux_view(),
            impl_->expression_words.ptr,
            shot_words,
            shots,
            seed,
            options.postselect_detectors ? 1 : 0,
            options.sample_exogenous_on_device ? 1 : 0,
            sampled_sampler_word_count,
            impl_->discarded_flags.ptr,
            impl_->logical_flags.ptr);
    } else {
        symft_persistent_sample_kernel_fast<<<shots, sampler_threads, sampler_shared_bytes>>>(
            impl_->view(),
            impl_->expression_words.ptr,
            shot_words,
            shots,
            seed,
            options.postselect_detectors ? 1 : 0,
            options.sample_exogenous_on_device ? 1 : 0,
            impl_->discarded_flags.ptr,
            impl_->logical_flags.ptr);
    }
    check_cuda(cudaGetLastError(), "CUDA sampler launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    result.elapsed_s = static_cast<double>(elapsed_ms) * 1.0e-3;

    impl_->host_discarded.resize(static_cast<std::size_t>(shots));
    impl_->host_logical.resize(static_cast<std::size_t>(shots));
    check_cuda(
        cudaMemcpy(
            impl_->host_discarded.data(),
            impl_->discarded_flags.ptr,
            static_cast<std::size_t>(shots) * sizeof(std::uint8_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy device-to-host");
    check_cuda(
        cudaMemcpy(
            impl_->host_logical.data(),
            impl_->logical_flags.ptr,
            static_cast<std::size_t>(shots) * sizeof(std::uint8_t),
            cudaMemcpyDeviceToHost),
        "cudaMemcpy device-to-host");

    for (int shot = 0; shot < shots; ++shot) {
        if (impl_->host_discarded[static_cast<std::size_t>(shot)] != 0) {
            ++result.discarded;
        } else {
            ++result.accepted;
            result.logical_errors += impl_->host_logical[static_cast<std::size_t>(shot)] != 0 ? 1 : 0;
        }
    }
    return result;
}

const char* cuda_sampler_backend_name() {
    return "cuda_persistent_shot_block";
}

} // namespace symft::cuda
