#pragma once

#include "circuit/circuit.hpp"
#include "sampler/batch_sampler.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/single_shot.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace symft {

struct CircuitSamplingCounts {
    std::uint64_t shots = 0;
    std::uint64_t discarded = 0;
    std::uint64_t accepted = 0;
    std::uint64_t logical_errors = 0;
};

struct CircuitSamplingTiming {
    double parse_s = 0.0;
    double plan_s = 0.0;
    double presample_s = 0.0;
    double execute_s = 0.0;
    double accumulate_s = 0.0;
    double sample_s = 0.0;
};

struct CircuitSamplingOptions {
    int observable = 0;
    bool postselect_detectors = false;
    int sample_chunk_shots = 0;
    int batch_size = 0;
    int batch_mask_threshold_denominator = 2;
    int threads = 1;
};

struct CircuitSamplingInfo {
    int n = 0;
    int records = 0;
    int detectors = 0;
    int observable_includes = 0;
    int observable = 0;
    int max_k = 0;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int threads = 1;
    bool active_components = false;
    bool detector_postselection = false;
    int batch_mask_threshold_denominator = 0;
};

struct CircuitSamplingRunResult {
    CircuitSamplingCounts counts;
    CircuitSamplingTiming timing;
    int active_threads = 1;
};

struct CircuitSamplingInput {
    FactoredInstructionProgram program;
    std::vector<std::vector<int>> logical_records;
    int observable = 0;
    int observable_includes = 0;
    CircuitSamplingTiming preprocessing_timing;
};

class PreparedCircuitSingleShotSampler {
  public:
    PreparedCircuitSingleShotSampler(CircuitSamplingInput input, CircuitSamplingOptions options = {});
    PreparedCircuitSingleShotSampler(
        FactoredInstructionProgram program,
        std::vector<std::vector<int>> logical_records = {},
        CircuitSamplingOptions options = {});

    PreparedCircuitSingleShotSampler(const PreparedCircuitSingleShotSampler&) = delete;
    PreparedCircuitSingleShotSampler& operator=(const PreparedCircuitSingleShotSampler&) = delete;
    ~PreparedCircuitSingleShotSampler();
    PreparedCircuitSingleShotSampler(PreparedCircuitSingleShotSampler&&) noexcept;
    PreparedCircuitSingleShotSampler& operator=(PreparedCircuitSingleShotSampler&&) noexcept;

    const CircuitSamplingInfo& info() const { return info_; }
    const CircuitSamplingTiming& preprocessing_timing() const { return preprocessing_timing_; }
    const FactoredInstructionProgram& program() const { return program_; }

    // Reuses worker storage; calls on the same sampler instance must not overlap.
    CircuitSamplingRunResult sample(std::uint64_t shots);
    CircuitSamplingRunResult sample(std::uint64_t shots, std::uint64_t stream_id);

  private:
    struct WorkerContext;

    CircuitSamplingOptions options_;
    FactoredInstructionProgram program_;
    std::vector<std::vector<int>> logical_records_;
    CircuitSamplingInfo info_;
    CircuitSamplingTiming preprocessing_timing_;
    PresampledExpressionPlan expression_plan_;
    std::vector<std::unique_ptr<WorkerContext>> workers_;
    std::uint64_t next_stream_id_ = 0;
};

class PreparedCircuitBatchSampler {
  public:
    PreparedCircuitBatchSampler(CircuitSamplingInput input, CircuitSamplingOptions options = {});
    PreparedCircuitBatchSampler(
        FactoredInstructionProgram program,
        std::vector<std::vector<int>> logical_records = {},
        CircuitSamplingOptions options = {});
    ~PreparedCircuitBatchSampler();

    PreparedCircuitBatchSampler(const PreparedCircuitBatchSampler&) = delete;
    PreparedCircuitBatchSampler& operator=(const PreparedCircuitBatchSampler&) = delete;
    PreparedCircuitBatchSampler(PreparedCircuitBatchSampler&&) noexcept;
    PreparedCircuitBatchSampler& operator=(PreparedCircuitBatchSampler&&) noexcept;

    const CircuitSamplingInfo& info() const { return info_; }
    const CircuitSamplingTiming& preprocessing_timing() const { return preprocessing_timing_; }
    const FactoredInstructionProgram& program() const { return program_; }

    // Reuses worker storage; calls on the same sampler instance must not overlap.
    CircuitSamplingRunResult sample(std::uint64_t shots);
    CircuitSamplingRunResult sample(std::uint64_t shots, std::uint64_t stream_id);

  private:
    struct WorkerContext;

    void rebind_after_move() noexcept;

    CircuitSamplingOptions options_;
    FactoredInstructionProgram program_;
    std::vector<std::vector<int>> logical_records_;
    CircuitSamplingInfo info_;
    CircuitSamplingTiming preprocessing_timing_;
    PresampledExpressionPlan expression_plan_;
    BatchDetectorPostselectionOptions postselection_options_;
    std::vector<std::unique_ptr<WorkerContext>> workers_;
    std::uint64_t next_stream_id_ = 0;
};

std::vector<std::vector<int>> logical_records_for_observable(
    const std::vector<CircuitObservableInclude>& observables,
    int observable);
CircuitSamplingInput make_circuit_sampling_input(
    FactoredInstructionProgram program,
    std::vector<std::vector<int>> logical_records = {},
    int observable = 0,
    int observable_includes = 0,
    CircuitSamplingTiming preprocessing_timing = {});

} // namespace symft
