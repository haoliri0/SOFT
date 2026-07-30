#pragma once

#include "sampler/prepared_sampler.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace symft::cuda {

struct CudaSamplingOptions {
    bool postselect_detectors = false;
    // Default fast path: sample exogenous noise inside each shot block instead of
    // materializing program.nsymbols * shots packed bits on the CPU.
    bool sample_exogenous_on_device = true;
    bool gpu_presample_expressions = false;
    bool lazy_exogenous_on_device = false;
    bool gpu_on_demand_expressions = false;
    int shots_per_launch = 1 << 20;
    int threads_per_block = 0;
};

class PreparedCircuitCudaSampler {
  public:
    PreparedCircuitCudaSampler(CircuitSamplingInput input, CudaSamplingOptions options = {});
    PreparedCircuitCudaSampler(
        FactoredInstructionProgram program,
        std::vector<std::vector<int>> logical_records = {},
        CudaSamplingOptions options = {});
    ~PreparedCircuitCudaSampler();

    PreparedCircuitCudaSampler(const PreparedCircuitCudaSampler&) = delete;
    PreparedCircuitCudaSampler& operator=(const PreparedCircuitCudaSampler&) = delete;
    PreparedCircuitCudaSampler(PreparedCircuitCudaSampler&&) noexcept;
    PreparedCircuitCudaSampler& operator=(PreparedCircuitCudaSampler&&) noexcept;

    const CircuitSamplingInfo& info() const;
    const CircuitSamplingTiming& preprocessing_timing() const;
    const FactoredInstructionProgram& program() const;

    CircuitSamplingRunResult sample(std::uint64_t shots);
    CircuitSamplingRunResult sample(std::uint64_t shots, std::uint64_t stream_id);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace symft::cuda
