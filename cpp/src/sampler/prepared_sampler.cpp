#include "sampler/prepared_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <thread>
#include <utility>

namespace symft {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::runtime_error("division by zero in chunk sizing");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

std::uint64_t block_seed(std::uint64_t base, std::uint64_t stream_id, std::uint64_t block_index) {
    return base ^
           (std::uint64_t{0x9e3779b97f4a7c15} * (block_index + 1)) ^
           (std::uint64_t{0xbf58476d1ce4e5b9} * (stream_id + 1));
}

int popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

std::size_t batch_word_count(int shots) {
    return shots <= 0 ? 0 : (static_cast<std::size_t>(shots) + 63) >> 6;
}

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return ~std::uint64_t{0};
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

std::size_t measurement_offset(std::size_t stride_words, int record, std::size_t word) {
    if (record <= 0) {
        throw std::runtime_error("record ids must be positive");
    }
    return static_cast<std::size_t>(record - 1) * stride_words + word;
}

void xor_records_into(
    std::vector<std::uint64_t>& out,
    const std::vector<std::uint64_t>& measurements,
    std::size_t stride_words,
    std::size_t nwords,
    const std::vector<int>& records) {
    std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (int record : records) {
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= measurements[measurement_offset(stride_words, record, word)];
        }
    }
}

bool record_parity(const std::vector<std::uint64_t>& words, const std::vector<int>& records) {
    bool parity = false;
    for (int record : records) {
        if (record <= 0) {
            throw std::runtime_error("record ids must be positive");
        }
        parity ^= packed_bit(words, record - 1);
    }
    return parity;
}

void accumulate_accepted_single_counts(
    CircuitSamplingCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::vector<int>>& logical_records) {
    ++counts.accepted;
    bool logical = false;
    for (const auto& records : logical_records) {
        logical ^= record_parity(measurement_words, records);
    }
    if (logical) {
        ++counts.logical_errors;
    }
}

void accumulate_single_counts(
    CircuitSamplingCounts& counts,
    const std::vector<std::uint64_t>& measurement_words,
    const std::vector<std::uint64_t>& detector_words,
    int ndetectors,
    const std::vector<std::vector<int>>& logical_records) {
    for (int detector = 0; detector < ndetectors; ++detector) {
        if (packed_bit(detector_words, detector)) {
            ++counts.discarded;
            return;
        }
    }
    accumulate_accepted_single_counts(counts, measurement_words, logical_records);
}

void accumulate_block_counts(
    CircuitSamplingCounts& counts,
    const BatchFactoredExecutorState& runtime,
    int block,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& discard_bits,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    const std::size_t stride_words = runtime.batch_words;
    const std::size_t nwords = batch_word_count(block);
    for (std::size_t word = 0; word < nwords; ++word) {
        discard_bits[word] = runtime.detector_any_words[word] & live_word_mask(block, word);
    }

    bool any_accepted = false;
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask(block, word);
        const std::uint64_t discarded_word = discard_bits[word] & live;
        const std::uint64_t accepted_word = (~discarded_word) & live;
        any_accepted = any_accepted || accepted_word != 0;
        counts.discarded += static_cast<std::uint64_t>(popcount64(discarded_word));
        counts.accepted += static_cast<std::uint64_t>(popcount64(accepted_word));
    }
    if (!any_accepted || logical_records.empty()) {
        return;
    }

    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (const auto& records : logical_records) {
        xor_records_into(scratch, runtime.measurement_words, stride_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            logical_bits[word] ^= scratch[word];
        }
    }

    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask(block, word);
        const std::uint64_t accepted_word = (~discard_bits[word]) & live;
        counts.logical_errors += static_cast<std::uint64_t>(popcount64(logical_bits[word] & accepted_word));
    }
}

void accumulate_logical_counts_for_survivors(
    CircuitSamplingCounts& counts,
    const BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& logical_records,
    std::vector<std::uint64_t>& logical_bits,
    std::vector<std::uint64_t>& scratch) {
    counts.accepted += static_cast<std::uint64_t>(runtime.active_shots);
    if (runtime.active_shots == 0 || logical_records.empty()) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    std::fill(logical_bits.begin(), logical_bits.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (const auto& records : logical_records) {
        xor_records_into(scratch, runtime.measurement_words, runtime.batch_words, nwords, records);
        for (std::size_t word = 0; word < nwords; ++word) {
            logical_bits[word] ^= scratch[word] & live_word_mask(runtime.active_shots, word);
        }
    }
    for (std::size_t word = 0; word < nwords; ++word) {
        counts.logical_errors += static_cast<std::uint64_t>(
            popcount64(logical_bits[word] & live_word_mask(runtime.active_shots, word)));
    }
}

CircuitSamplingInfo make_info(
    const FactoredInstructionProgram& program,
    int observable,
    int observable_includes,
    const CircuitSamplingOptions& options,
    int batch_size,
    int sample_chunk_shots,
    int threads) {
    CircuitSamplingInfo info;
    info.n = program.n;
    info.records = program.nrecords;
    info.detectors = program.ndetectors;
    info.observable_includes = observable_includes;
    info.observable = observable;
    info.max_k = program.max_k;
    info.batch_size = batch_size;
    info.sample_chunk_shots = sample_chunk_shots;
    info.threads = threads;
    info.active_components = program.use_active_components;
    info.detector_postselection = options.postselect_detectors;
    info.batch_mask_threshold_denominator =
        options.postselect_detectors ? options.batch_mask_threshold_denominator : 0;
    return info;
}

int sample_chunk_or_default(int requested, int min_auto_shots = 1) {
    if (requested > 0) {
        return requested;
    }
    return std::max(default_single_shot_sample_chunk_shots(), min_auto_shots);
}

int batch_size_or_default(
    int requested,
    const FactoredInstructionProgram& program) {
    if (requested > 0) {
        return requested;
    }
    return default_batch_count(program.max_k);
}

struct PreparedWorkerBuffers {
    CircuitSamplingCounts counts;
    CircuitSamplingTiming timing;
    PackedPresampledExogenous samples;
    PresampledExpressionBlock expression_block;
};

void prepare_expression_plan(
    PresampledExpressionPlan& expression_plan,
    const FactoredInstructionProgram& program) {
    PackedPresampledExogenous samples;
    prepare_presampled_exogenous_packed(samples, program);
    prepare_presampled_expression_plan(expression_plan, program, samples);
}

void prepare_worker_buffers(
    PreparedWorkerBuffers& worker,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    int sample_chunk_shots) {
    prepare_presampled_exogenous_packed(worker.samples, program);
    const std::size_t chunk_words = batch_word_count(sample_chunk_shots);
    worker.samples.value_words.reserve(
        static_cast<std::size_t>(program.nsymbols) * chunk_words);
    worker.expression_block.expression_words.reserve(
        expression_plan.block_expressions.size() * chunk_words);
}

int active_worker_count(int requested_threads, std::uint64_t nchunks) {
    const std::uint64_t available_workers = std::max<std::uint64_t>(1, nchunks);
    return static_cast<int>(std::min<std::uint64_t>(
        static_cast<std::uint64_t>(requested_threads),
        available_workers));
}

template <typename WorkerFn>
void run_workers(int worker_count, WorkerFn&& worker) {
    if (worker_count == 1) {
        worker(0);
        return;
    }

    std::vector<std::exception_ptr> failures(static_cast<std::size_t>(worker_count));
    {
        std::vector<std::jthread> threads;
        threads.reserve(static_cast<std::size_t>(worker_count));
        for (int worker_id = 0; worker_id < worker_count; ++worker_id) {
            threads.emplace_back([&, worker_id] {
                try {
                    worker(worker_id);
                } catch (...) {
                    failures[static_cast<std::size_t>(worker_id)] = std::current_exception();
                }
            });
        }
    }
    for (const auto& failure : failures) {
        if (failure != nullptr) {
            std::rethrow_exception(failure);
        }
    }
}

template <typename WorkerContext>
void merge_worker_result(
    CircuitSamplingRunResult& result,
    const WorkerContext& worker) {
    result.counts.shots += worker.counts.shots;
    result.counts.discarded += worker.counts.discarded;
    result.counts.accepted += worker.counts.accepted;
    result.counts.logical_errors += worker.counts.logical_errors;
    result.timing.presample_s += worker.timing.presample_s;
    result.timing.execute_s += worker.timing.execute_s;
    result.timing.accumulate_s += worker.timing.accumulate_s;
}

} // namespace

std::vector<std::vector<int>> logical_records_for_observable(
    const std::vector<CircuitObservableInclude>& observables,
    int observable) {
    std::vector<std::vector<int>> out;
    for (const auto& include : observables) {
        if (include.index == observable) {
            out.push_back(include.records);
        }
    }
    return out;
}

CircuitSamplingInput make_circuit_sampling_input(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    int observable,
    int observable_includes,
    CircuitSamplingTiming preprocessing_timing) {
    CircuitSamplingInput input;
    input.program = std::move(program);
    input.logical_records = std::move(logical_records);
    input.observable = observable;
    input.observable_includes = observable_includes;
    input.preprocessing_timing = preprocessing_timing;
    return input;
}

struct PreparedCircuitSingleShotSampler::WorkerContext : PreparedWorkerBuffers {
    FactoredExecutorState runtime;

    explicit WorkerContext(const FactoredInstructionProgram& program)
        : runtime(program) {}
};

PreparedCircuitSingleShotSampler::PreparedCircuitSingleShotSampler(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    CircuitSamplingOptions options)
    : PreparedCircuitSingleShotSampler(
          make_circuit_sampling_input(
              std::move(program),
              std::move(logical_records),
              options.observable),
          options) {}

PreparedCircuitSingleShotSampler::PreparedCircuitSingleShotSampler(
    CircuitSamplingInput input,
    CircuitSamplingOptions options)
    : options_(options) {
    program_ = std::move(input.program);
    logical_records_ = std::move(input.logical_records);
    preprocessing_timing_ = input.preprocessing_timing;

    options_.sample_chunk_shots = sample_chunk_or_default(options_.sample_chunk_shots);
    options_.threads = std::max(1, options_.threads);
    info_ = make_info(
        program_,
        input.observable,
        input.observable_includes,
        options_,
        0,
        options_.sample_chunk_shots,
        options_.threads);

    prepare_expression_plan(expression_plan_, program_);
    workers_.reserve(static_cast<std::size_t>(options_.threads));
    for (int worker_id = 0; worker_id < options_.threads; ++worker_id) {
        auto context = std::make_unique<WorkerContext>(program_);
        prepare_worker_buffers(
            *context,
            program_,
            expression_plan_,
            options_.sample_chunk_shots);
        workers_.push_back(std::move(context));
    }
}

PreparedCircuitSingleShotSampler::~PreparedCircuitSingleShotSampler() = default;
PreparedCircuitSingleShotSampler::PreparedCircuitSingleShotSampler(
    PreparedCircuitSingleShotSampler&&) noexcept = default;
PreparedCircuitSingleShotSampler& PreparedCircuitSingleShotSampler::operator=(
    PreparedCircuitSingleShotSampler&&) noexcept = default;

CircuitSamplingRunResult PreparedCircuitSingleShotSampler::sample(std::uint64_t shots) {
    return sample(shots, next_stream_id_++);
}

CircuitSamplingRunResult PreparedCircuitSingleShotSampler::sample(
    std::uint64_t shots,
    std::uint64_t stream_id) {
    CircuitSamplingRunResult result;
    const std::uint64_t nchunks = ceil_div_u64(
        shots,
        static_cast<std::uint64_t>(options_.sample_chunk_shots));
    const int active_threads = active_worker_count(options_.threads, nchunks);
    result.active_threads = active_threads;

    const auto sample_start = Clock::now();
    auto run_worker = [&](int worker_id) {
        auto& context = *workers_[static_cast<std::size_t>(worker_id)];
        context.counts = {};
        context.timing = {};
        for (std::uint64_t chunk_index = static_cast<std::uint64_t>(worker_id);
             chunk_index < nchunks;
             chunk_index += static_cast<std::uint64_t>(active_threads)) {
            const std::uint64_t offset =
                chunk_index * static_cast<std::uint64_t>(options_.sample_chunk_shots);
            const int chunk = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(options_.sample_chunk_shots),
                shots - offset));

            const auto presample_start = Clock::now();
            resample_prepared_exogenous_packed_in_place(
                context.samples,
                program_,
                chunk,
                block_seed(0x7eed0000ULL, stream_id, chunk_index));
            evaluate_presampled_expression_block(
                context.expression_block,
                expression_plan_,
                context.samples);
            const auto presample_stop = Clock::now();
            context.timing.presample_s += seconds_between(presample_start, presample_stop);

            const auto execute_start = Clock::now();
            context.runtime.rng_state = block_seed(0x5eed1234ULL, stream_id, chunk_index);
            for (int shot = 0; shot < chunk; ++shot) {
                reset_executor(context.runtime, program_, !options_.postselect_detectors);
                if (options_.postselect_detectors) {
                    const bool survived = execute_postselected_in_place(
                        context.runtime,
                        program_,
                        expression_plan_,
                        context.expression_block,
                        shot);
                    if (!survived) {
                        ++context.counts.discarded;
                    } else {
                        accumulate_accepted_single_counts(
                            context.counts,
                            context.runtime.measurement_words,
                            logical_records_);
                    }
                    continue;
                }
                execute_in_place(
                    context.runtime,
                    program_,
                    expression_plan_,
                    context.expression_block,
                    shot);
                accumulate_single_counts(
                    context.counts,
                    context.runtime.measurement_words,
                    context.runtime.detector_words,
                    program_.ndetectors,
                    logical_records_);
            }
            context.timing.execute_s += seconds_between(execute_start, Clock::now());
            context.counts.shots += static_cast<std::uint64_t>(chunk);
        }
    };

    run_workers(active_threads, run_worker);

    for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
        merge_worker_result(result, *workers_[static_cast<std::size_t>(worker_id)]);
    }
    result.timing.sample_s = seconds_between(sample_start, Clock::now());
    return result;
}

struct PreparedCircuitBatchSampler::WorkerContext : PreparedWorkerBuffers {
    BatchFactoredExecutorState runtime;
    std::vector<std::uint64_t> discard_bits;
    std::vector<std::uint64_t> logical_bits;
    std::vector<std::uint64_t> scratch;
    BatchDetectorPostselectionScratch postselection_scratch;

    WorkerContext(
        const FactoredInstructionProgram& program,
        int batch_size)
        : runtime(program, batch_size),
          discard_bits(runtime.batch_words, 0),
          logical_bits(runtime.batch_words, 0),
          scratch(runtime.batch_words, 0) {}
};

PreparedCircuitBatchSampler::PreparedCircuitBatchSampler(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    CircuitSamplingOptions options)
    : PreparedCircuitBatchSampler(
          make_circuit_sampling_input(
              std::move(program),
              std::move(logical_records),
              options.observable),
          options) {}

PreparedCircuitBatchSampler::PreparedCircuitBatchSampler(
    CircuitSamplingInput input,
    CircuitSamplingOptions options)
    : options_(options) {
    program_ = std::move(input.program);
    logical_records_ = std::move(input.logical_records);
    preprocessing_timing_ = input.preprocessing_timing;

    options_.batch_size = batch_size_or_default(
        options_.batch_size,
        program_);
    options_.sample_chunk_shots = sample_chunk_or_default(
        options_.sample_chunk_shots,
        options_.batch_size);
    options_.threads = std::max(1, options_.threads);
    postselection_options_.mask_dead_shots_min_fraction_denominator =
        options_.batch_mask_threshold_denominator;
    postselection_options_.retained_record_uses = &logical_records_;

    info_ = make_info(
        program_,
        input.observable,
        input.observable_includes,
        options_,
        options_.batch_size,
        options_.sample_chunk_shots,
        options_.threads);

    prepare_expression_plan(expression_plan_, program_);
    workers_.reserve(static_cast<std::size_t>(options_.threads));
    for (int worker_id = 0; worker_id < options_.threads; ++worker_id) {
        auto context = std::make_unique<WorkerContext>(
            program_,
            options_.batch_size);
        context->runtime.store_detector_records = false;
        context->runtime.dense_shot_major_active = true;
        if (options_.postselect_detectors) {
            prepare_batch_detector_postselection_scratch(
                context->postselection_scratch,
                context->runtime,
                program_,
                postselection_options_);
        }
        prepare_worker_buffers(
            *context,
            program_,
            expression_plan_,
            options_.sample_chunk_shots);
        workers_.push_back(std::move(context));
    }
}

PreparedCircuitBatchSampler::~PreparedCircuitBatchSampler() = default;
PreparedCircuitBatchSampler::PreparedCircuitBatchSampler(
    PreparedCircuitBatchSampler&& other) noexcept
    : options_(std::move(other.options_)),
      program_(std::move(other.program_)),
      logical_records_(std::move(other.logical_records_)),
      info_(std::move(other.info_)),
      preprocessing_timing_(std::move(other.preprocessing_timing_)),
      expression_plan_(std::move(other.expression_plan_)),
      postselection_options_(std::move(other.postselection_options_)),
      workers_(std::move(other.workers_)),
      next_stream_id_(other.next_stream_id_) {
    rebind_after_move();
}

PreparedCircuitBatchSampler& PreparedCircuitBatchSampler::operator=(
    PreparedCircuitBatchSampler&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    options_ = std::move(other.options_);
    program_ = std::move(other.program_);
    logical_records_ = std::move(other.logical_records_);
    info_ = std::move(other.info_);
    preprocessing_timing_ = std::move(other.preprocessing_timing_);
    expression_plan_ = std::move(other.expression_plan_);
    postselection_options_ = std::move(other.postselection_options_);
    workers_ = std::move(other.workers_);
    next_stream_id_ = other.next_stream_id_;
    rebind_after_move();
    return *this;
}

void PreparedCircuitBatchSampler::rebind_after_move() noexcept {
    postselection_options_.retained_record_uses = &logical_records_;
}

CircuitSamplingRunResult PreparedCircuitBatchSampler::sample(std::uint64_t shots) {
    return sample(shots, next_stream_id_++);
}

CircuitSamplingRunResult PreparedCircuitBatchSampler::sample(
    std::uint64_t shots,
    std::uint64_t stream_id) {
    CircuitSamplingRunResult result;
    const std::uint64_t nchunks = ceil_div_u64(
        shots,
        static_cast<std::uint64_t>(options_.sample_chunk_shots));
    const std::uint64_t blocks_per_chunk = ceil_div_u64(
        static_cast<std::uint64_t>(options_.sample_chunk_shots),
        static_cast<std::uint64_t>(options_.batch_size));
    const int active_threads = active_worker_count(options_.threads, nchunks);
    result.active_threads = active_threads;

    const auto sample_start = Clock::now();
    auto run_worker = [&](int worker_id) {
        auto& context = *workers_[static_cast<std::size_t>(worker_id)];
        context.counts = {};
        context.timing = {};
        for (std::uint64_t chunk_index = static_cast<std::uint64_t>(worker_id);
             chunk_index < nchunks;
             chunk_index += static_cast<std::uint64_t>(active_threads)) {
            const std::uint64_t chunk_offset =
                chunk_index * static_cast<std::uint64_t>(options_.sample_chunk_shots);
            const int chunk_shots = static_cast<int>(std::min<std::uint64_t>(
                static_cast<std::uint64_t>(options_.sample_chunk_shots),
                shots - chunk_offset));

            const auto presample_start = Clock::now();
            resample_prepared_exogenous_packed_in_place(
                context.samples,
                program_,
                chunk_shots,
                block_seed(0x7eed0000ULL, stream_id, chunk_index));
            evaluate_presampled_expression_block(
                context.expression_block,
                expression_plan_,
                context.samples);
            const auto presample_stop = Clock::now();
            context.timing.presample_s += seconds_between(presample_start, presample_stop);

            const auto execute_start = Clock::now();
            for (int chunk_local_offset = 0, local_block_index = 0;
                 chunk_local_offset < chunk_shots;
                 chunk_local_offset += options_.batch_size, ++local_block_index) {
                const int block = std::min(options_.batch_size, chunk_shots - chunk_local_offset);
                const std::uint64_t block_index =
                    chunk_index * blocks_per_chunk + static_cast<std::uint64_t>(local_block_index);
                reset_batch_executor(
                    context.runtime,
                    program_,
                    block,
                    !options_.postselect_detectors);
                context.runtime.rng_state = block_seed(0x5eed1234ULL, stream_id, block_index);
                if (options_.postselect_detectors) {
                    const auto postselection_result = execute_batch_postselected_in_place(
                        context.runtime,
                        program_,
                        expression_plan_,
                        context.expression_block,
                        chunk_local_offset,
                        context.postselection_scratch,
                        postselection_options_);
                    context.counts.discarded +=
                        static_cast<std::uint64_t>(postselection_result.discarded);
                } else {
                    execute_batch_in_place(
                        context.runtime,
                        program_,
                        expression_plan_,
                        context.expression_block,
                        chunk_local_offset);
                }
                if (options_.postselect_detectors) {
                    accumulate_logical_counts_for_survivors(
                        context.counts,
                        context.runtime,
                        logical_records_,
                        context.logical_bits,
                        context.scratch);
                } else {
                    accumulate_block_counts(
                        context.counts,
                        context.runtime,
                        block,
                        logical_records_,
                        context.discard_bits,
                        context.logical_bits,
                        context.scratch);
                }
                context.counts.shots += static_cast<std::uint64_t>(block);
            }
            context.timing.execute_s += seconds_between(execute_start, Clock::now());
        }
    };

    run_workers(active_threads, run_worker);

    for (int worker_id = 0; worker_id < active_threads; ++worker_id) {
        merge_worker_result(result, *workers_[static_cast<std::size_t>(worker_id)]);
    }
    result.timing.sample_s = seconds_between(sample_start, Clock::now());
    return result;
}

} // namespace symft
