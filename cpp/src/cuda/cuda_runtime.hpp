#pragma once

#include "cuda/cuda_program.hpp"

#include <cstdint>
#include <memory>

namespace symft::cuda {

struct CudaKernelRunResult {
    std::uint64_t discarded = 0;
    std::uint64_t accepted = 0;
    std::uint64_t logical_errors = 0;
    double elapsed_s = 0.0;
};

struct CudaLaunchOptions {
    bool postselect_detectors = false;
    bool sample_exogenous_on_device = false;
    bool generate_expressions_on_device = false;
    bool lazy_exogenous_on_device = false;
    bool on_demand_expression_blocks = false;
    int threads_per_block = 32;
};

class CudaRuntimeProgram {
  public:
    explicit CudaRuntimeProgram(const CudaProgramData& program);
    ~CudaRuntimeProgram();

    CudaRuntimeProgram(const CudaRuntimeProgram&) = delete;
    CudaRuntimeProgram& operator=(const CudaRuntimeProgram&) = delete;
    CudaRuntimeProgram(CudaRuntimeProgram&& other) noexcept;
    CudaRuntimeProgram& operator=(CudaRuntimeProgram&& other) noexcept;

    CudaKernelRunResult run(
        const std::uint64_t* expression_words,
        std::size_t expression_word_count,
        std::size_t shot_words,
        int shots,
        std::uint64_t seed,
        const CudaLaunchOptions& options);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* cuda_sampler_backend_name();

} // namespace symft::cuda
