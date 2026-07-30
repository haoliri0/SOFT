#include "cuda/cuda_sampler.hpp"
#include "frontend/stim_prepared_sampler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

enum class CudaMode {
    GpuExogenous,
    GpuPresampleExpressions,
    GpuOnDemandExpressions,
    LazyExogenous,
};

std::string write_temp_stim(const std::string& name, const std::string& text) {
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    std::ofstream out(path);
    out << text;
    return path;
}

std::string fixture_path(const std::string& name) {
    {
        std::ifstream in(name);
        if (in.good()) {
            return name;
        }
    }
    const std::string parent = "../" + name;
    {
        std::ifstream in(parent);
        if (in.good()) {
            return parent;
        }
    }
    return name;
}

symft::CircuitSamplingRunResult run_cuda_file(
    const std::string& path,
    std::uint64_t shots,
    bool postselect_detectors = true,
    CudaMode mode = CudaMode::GpuExogenous) {
    symft::CircuitSamplingOptions frontend_options;
    auto input = symft::make_stim_circuit_sampling_input_from_file(path, frontend_options);
    symft::cuda::CudaSamplingOptions options;
    options.postselect_detectors = postselect_detectors;
    options.sample_exogenous_on_device = mode == CudaMode::GpuExogenous;
    options.gpu_presample_expressions = mode == CudaMode::GpuPresampleExpressions;
    options.gpu_on_demand_expressions = mode == CudaMode::GpuOnDemandExpressions;
    options.lazy_exogenous_on_device = mode == CudaMode::LazyExogenous;
    options.shots_per_launch = static_cast<int>(shots);
    options.threads_per_block =
        (mode == CudaMode::GpuPresampleExpressions || mode == CudaMode::GpuOnDemandExpressions) ? 128 : 32;
    symft::cuda::PreparedCircuitCudaSampler sampler(std::move(input), options);
    return sampler.sample(shots, 0);
}

symft::CircuitSamplingRunResult run_cpu_batch_file(
    const std::string& path,
    std::uint64_t shots,
    bool postselect_detectors = true) {
    symft::CircuitSamplingOptions options;
    options.postselect_detectors = postselect_detectors;
    options.sample_chunk_shots = static_cast<int>(shots);
    options.threads = 1;
    auto sampler = symft::prepare_batch_sampler_from_stim_file(path, options);
    return sampler.sample(shots, 0);
}

double discard_rate(const symft::CircuitSamplingRunResult& run) {
    return static_cast<double>(run.counts.discarded) / static_cast<double>(run.counts.shots);
}

void require_close(double actual, double expected, double tolerance, const std::string& message) {
    require(std::abs(actual - expected) <= tolerance, message);
}

void test_deterministic_postselection() {
    const std::string accept_path = write_temp_stim(
        "symft_cuda_accept.stim",
        "M 0\n"
        "DETECTOR rec[-1]\n");
    const auto accept = run_cuda_file(accept_path, 256);
    require(accept.counts.shots == 256, "CUDA deterministic accept shot count");
    require(accept.counts.discarded == 0, "CUDA deterministic quiet detector accepted");
    require(accept.counts.accepted == 256, "CUDA deterministic accepted count");

    const std::string reject_path = write_temp_stim(
        "symft_cuda_reject.stim",
        "M !0\n"
        "DETECTOR rec[-1]\n");
    const auto reject = run_cuda_file(reject_path, 256);
    require(reject.counts.shots == 256, "CUDA deterministic reject shot count");
    require(reject.counts.discarded == 256, "CUDA deterministic fired detector discarded");
    require(reject.counts.accepted == 0, "CUDA deterministic rejected accepted count");

    std::remove(accept_path.c_str());
    std::remove(reject_path.c_str());
}

void test_fixture_discard_rates() {
    {
        const std::string path = fixture_path("benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim");
        const auto cpu = run_cpu_batch_file(path, 20000);
        const auto gpu = run_cuda_file(path, 20000);
        const auto gpu_presample = run_cuda_file(path, 20000, true, CudaMode::GpuPresampleExpressions);
        const auto gpu_on_demand = run_cuda_file(path, 20000, true, CudaMode::GpuOnDemandExpressions);
        const auto gpu_lazy = run_cuda_file(path, 20000, true, CudaMode::LazyExogenous);
        require_close(discard_rate(gpu), discard_rate(cpu), 0.02, "CUDA d3 discard rate matches CPU");
        require_close(
            discard_rate(gpu_presample),
            discard_rate(cpu),
            0.02,
            "CUDA d3 gpu-presampled expression discard rate matches CPU");
        require_close(
            discard_rate(gpu_on_demand),
            discard_rate(cpu),
            0.02,
            "CUDA d3 on-demand expression discard rate matches CPU");
        require_close(discard_rate(gpu_lazy), discard_rate(cpu), 0.02, "CUDA d3 lazy discard rate matches CPU");
    }
    {
        const std::string path = fixture_path("benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim");
        const auto cpu = run_cpu_batch_file(path, 30000);
        const auto gpu = run_cuda_file(path, 30000);
        const auto gpu_presample = run_cuda_file(path, 30000, true, CudaMode::GpuPresampleExpressions);
        const auto gpu_on_demand = run_cuda_file(path, 30000, true, CudaMode::GpuOnDemandExpressions);
        const auto gpu_lazy = run_cuda_file(path, 30000, true, CudaMode::LazyExogenous);
        require_close(discard_rate(gpu), discard_rate(cpu), 0.02, "CUDA d5 discard rate matches CPU");
        require_close(
            discard_rate(gpu_presample),
            discard_rate(cpu),
            0.02,
            "CUDA d5 gpu-presampled expression discard rate matches CPU");
        require_close(
            discard_rate(gpu_on_demand),
            discard_rate(cpu),
            0.02,
            "CUDA d5 on-demand expression discard rate matches CPU");
        require_close(discard_rate(gpu_lazy), discard_rate(cpu), 0.02, "CUDA d5 lazy discard rate matches CPU");
    }
}

} // namespace

int main() {
    test_deterministic_postselection();
    test_fixture_discard_rates();
    std::cout << "symft_cuda_tests passed\n";
    return 0;
}
