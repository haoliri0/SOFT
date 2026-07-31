#include "cuda/cuda_sampler.hpp"

#include "cuda/cuda_program.hpp"
#include "cuda/cuda_runtime.hpp"
#include "sampler/exogenous.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace symft::cuda {
namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw Error("division by zero in CUDA chunk sizing");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

std::uint64_t block_seed(std::uint64_t base, std::uint64_t stream_id, std::uint64_t block_index) {
    return base ^
           (std::uint64_t{0x9e3779b97f4a7c15} * (block_index + 1)) ^
           (std::uint64_t{0xbf58476d1ce4e5b9} * (stream_id + 1));
}

int normalized_shots_per_launch(int requested) {
    if (requested <= 0) {
        return 1 << 20;
    }
    return requested;
}

CircuitSamplingInfo make_cuda_info(
    const FactoredInstructionProgram& program,
    const CircuitSamplingInput& input,
    const CudaSamplingOptions& options) {
    CircuitSamplingInfo info;
    info.n = program.n;
    info.records = program.nrecords;
    info.detectors = program.ndetectors;
    info.observable_includes = input.observable_includes;
    info.observable = input.observable;
    info.max_k = program.max_k;
    info.batch_size = 0;
    info.sample_chunk_shots = options.shots_per_launch;
    info.threads = 1;
    info.active_components = false;
    info.detector_postselection = options.postselect_detectors;
    info.batch_mask_threshold_denominator = 0;
    return info;
}

} // namespace

struct PreparedCircuitCudaSampler::Impl {
    CudaSamplingOptions options;
    FactoredInstructionProgram program;
    std::vector<std::vector<int>> logical_records;
    CircuitSamplingInfo info;
    CircuitSamplingTiming preprocessing_timing;
    PackedPresampledExogenous samples;
    PresampledExpressionPlan expression_plan;
    PresampledExpressionBlock expression_block;
    CudaProgramData cuda_program;
    std::unique_ptr<CudaRuntimeProgram> runtime;
    std::uint64_t next_stream_id = 0;

    Impl(CircuitSamplingInput input, CudaSamplingOptions options_)
        : options(options_) {
        const int max_k_hint = input.program.max_k;
        if (options.gpu_presample_expressions ||
            options.lazy_exogenous_on_device ||
            options.gpu_on_demand_expressions) {
            options.sample_exogenous_on_device = false;
        }
        options.shots_per_launch = normalized_shots_per_launch(options.shots_per_launch);
        if (options.threads_per_block <= 0) {
            options.threads_per_block =
                (options.gpu_presample_expressions || options.gpu_on_demand_expressions)
                    ? (max_k_hint <= 4 ? 32 : 128)
                    : 32;
        }
        program = std::move(input.program);
        logical_records = std::move(input.logical_records);
        preprocessing_timing = input.preprocessing_timing;

        const int device_modes =
            (options.sample_exogenous_on_device ? 1 : 0) +
            (options.gpu_presample_expressions ? 1 : 0) +
            (options.lazy_exogenous_on_device ? 1 : 0) +
            (options.gpu_on_demand_expressions ? 1 : 0);
        if (device_modes > 1) {
            throw Error("CUDA exogenous modes are mutually exclusive");
        }

        if (options.sample_exogenous_on_device || options.lazy_exogenous_on_device) {
            cuda_program = build_cuda_program_data_device_exogenous(program, logical_records);
        } else {
            prepare_presampled_exogenous_packed(samples, program);
            prepare_presampled_expression_plan(expression_plan, program, samples);
            cuda_program = build_cuda_program_data(program, expression_plan, logical_records);
        }
        runtime = std::make_unique<CudaRuntimeProgram>(cuda_program);
        info = make_cuda_info(program, input, options);
    }
};

PreparedCircuitCudaSampler::PreparedCircuitCudaSampler(CircuitSamplingInput input, CudaSamplingOptions options)
    : impl_(std::make_unique<Impl>(std::move(input), options)) {}

PreparedCircuitCudaSampler::PreparedCircuitCudaSampler(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records,
    CudaSamplingOptions options)
    : PreparedCircuitCudaSampler(
          symft::make_circuit_sampling_input(
              std::move(program),
              std::move(logical_records)),
          options) {}

PreparedCircuitCudaSampler::~PreparedCircuitCudaSampler() = default;
PreparedCircuitCudaSampler::PreparedCircuitCudaSampler(PreparedCircuitCudaSampler&&) noexcept = default;
PreparedCircuitCudaSampler& PreparedCircuitCudaSampler::operator=(PreparedCircuitCudaSampler&&) noexcept = default;

const CircuitSamplingInfo& PreparedCircuitCudaSampler::info() const {
    return impl_->info;
}

const CircuitSamplingTiming& PreparedCircuitCudaSampler::preprocessing_timing() const {
    return impl_->preprocessing_timing;
}

const FactoredInstructionProgram& PreparedCircuitCudaSampler::program() const {
    return impl_->program;
}

CircuitSamplingRunResult PreparedCircuitCudaSampler::sample(std::uint64_t shots) {
    return sample(shots, impl_->next_stream_id++);
}

CircuitSamplingRunResult PreparedCircuitCudaSampler::sample(
    std::uint64_t shots,
    std::uint64_t stream_id) {
    CircuitSamplingRunResult result;
    result.active_threads = 1;
    const std::uint64_t nchunks =
        ceil_div_u64(shots, static_cast<std::uint64_t>(impl_->options.shots_per_launch));
    const auto sample_start = Clock::now();

    for (std::uint64_t chunk_index = 0; chunk_index < nchunks; ++chunk_index) {
        const std::uint64_t offset =
            chunk_index * static_cast<std::uint64_t>(impl_->options.shots_per_launch);
        const int chunk = static_cast<int>(std::min<std::uint64_t>(
            static_cast<std::uint64_t>(impl_->options.shots_per_launch),
            shots - offset));

        const auto presample_start = Clock::now();
        const bool cpu_presample_exogenous =
            !impl_->options.sample_exogenous_on_device &&
            !impl_->options.gpu_presample_expressions &&
            !impl_->options.lazy_exogenous_on_device &&
            !impl_->options.gpu_on_demand_expressions;
        if (cpu_presample_exogenous) {
            resample_prepared_exogenous_packed_in_place(
                impl_->samples,
                impl_->program,
                chunk,
                block_seed(0x7eed0000ULL, stream_id, chunk_index));
            evaluate_presampled_expression_block(
                impl_->expression_block,
                impl_->expression_plan,
                impl_->samples);
        }
        result.timing.presample_s += seconds_between(presample_start, Clock::now());

        const auto execute_start = Clock::now();
        CudaLaunchOptions launch_options;
        launch_options.postselect_detectors = impl_->options.postselect_detectors;
        launch_options.sample_exogenous_on_device = impl_->options.sample_exogenous_on_device;
        launch_options.generate_expressions_on_device = impl_->options.gpu_presample_expressions;
        launch_options.lazy_exogenous_on_device = impl_->options.lazy_exogenous_on_device;
        launch_options.on_demand_expression_blocks = impl_->options.gpu_on_demand_expressions;
        launch_options.threads_per_block = impl_->options.threads_per_block;
        const std::uint64_t* expression_words =
            cpu_presample_exogenous ? impl_->expression_block.expression_words.data() : nullptr;
        const std::size_t expression_word_count =
            cpu_presample_exogenous ? impl_->expression_block.expression_words.size() : 0;
        const std::size_t shot_words =
            cpu_presample_exogenous
                ? impl_->expression_block.shot_words
                : (impl_->options.gpu_presample_expressions ? static_cast<std::size_t>((chunk + 63) >> 6) : 0);
        const auto kernel_result = impl_->runtime->run(
            expression_words,
            expression_word_count,
            shot_words,
            chunk,
            block_seed(
                cpu_presample_exogenous ? 0x5eed1234ULL : 0x7eed0000ULL,
                stream_id,
                chunk_index),
            launch_options);
        result.timing.execute_s += seconds_between(execute_start, Clock::now());
        result.timing.accumulate_s += kernel_result.elapsed_s;

        result.counts.shots += static_cast<std::uint64_t>(chunk);
        result.counts.discarded += kernel_result.discarded;
        result.counts.accepted += kernel_result.accepted;
        result.counts.logical_errors += kernel_result.logical_errors;
    }

    result.timing.sample_s = seconds_between(sample_start, Clock::now());
    return result;
}

} // namespace symft::cuda
