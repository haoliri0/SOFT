#include "cuda/cuda_sampler.hpp"
#include "frontend/stim_prepared_sampler.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

struct Options {
    std::string path = "benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim";
    std::uint64_t shots = 1000000ULL;
    int repeats = 1;
    int observable = 0;
    // Tuned default for MSC_d5. All three values remain independently
    // overridable through the command line for other circuits or GPUs.
    int shots_per_launch = 1 << 18;
    int threads_per_block = 128;
    bool postselect_detectors = false;
    bool sample_exogenous_on_device = false;
    bool gpu_presample_expressions = true;
    bool lazy_exogenous_on_device = false;
    bool gpu_on_demand_expressions = false;
    bool print_program_stats = false;
};

std::uint64_t parse_u64(const char* raw, const char* name) {
    if (raw[0] == '-') {
        throw std::runtime_error(std::string(name) + " must be nonnegative");
    }
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
    return value;
}

int parse_positive_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must be in 1:Int32Max");
    }
    return static_cast<int>(value);
}

int parse_nonnegative_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must fit in Int32Max");
    }
    return static_cast<int>(value);
}

std::string require_value(const std::string& option, const std::string& inline_value, int& idx, int argc, char** argv) {
    if (!inline_value.empty()) {
        return inline_value;
    }
    if (idx + 1 >= argc) {
        throw std::runtime_error(option + " requires a value");
    }
    return argv[++idx];
}

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " --circuit benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim --shots 1000000 [options]\n"
        << "  --postselect-detectors / --no-postselect-detectors\n"
        << "  --shots-per-launch N\n"
        << "  --threads-per-block N\n"
        << "  --gpu-exogenous / --gpu-presample-expressions / --gpu-on-demand-expressions"
           " / --gpu-lazy-exogenous / --cpu-presample-exogenous\n"
        << "  MSC_d5 default: --gpu-presample-expressions --threads-per-block 128"
           " --shots-per-launch 262144\n"
        << "  --repeats N\n"
        << "  --observable N\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    int positional = 0;
    for (int idx = 1; idx < argc; ++idx) {
        std::string arg(argv[idx]);
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg.rfind("--", 0) != 0) {
            if (positional == 0) {
                options.path = arg;
            } else if (positional == 1) {
                options.shots = parse_u64(arg.c_str(), "shots");
            } else {
                throw std::runtime_error("too many positional arguments");
            }
            ++positional;
            continue;
        }

        std::string name = arg;
        std::string value;
        const std::size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            name = arg.substr(0, eq);
            value = arg.substr(eq + 1);
        }

        if (name == "--circuit" || name == "--file") {
            options.path = require_value(name, value, idx, argc, argv);
        } else if (name == "--shots") {
            const std::string raw = require_value(name, value, idx, argc, argv);
            options.shots = parse_u64(raw.c_str(), "shots");
        } else if (name == "--repeats") {
            const std::string raw = require_value(name, value, idx, argc, argv);
            options.repeats = parse_positive_int(raw.c_str(), "repeats");
        } else if (name == "--observable") {
            const std::string raw = require_value(name, value, idx, argc, argv);
            options.observable = parse_nonnegative_int(raw.c_str(), "observable");
        } else if (name == "--shots-per-launch") {
            const std::string raw = require_value(name, value, idx, argc, argv);
            options.shots_per_launch = parse_positive_int(raw.c_str(), "shots_per_launch");
        } else if (name == "--threads-per-block") {
            const std::string raw = require_value(name, value, idx, argc, argv);
            options.threads_per_block = parse_positive_int(raw.c_str(), "threads_per_block");
        } else if (name == "--postselect-detectors") {
            options.postselect_detectors = true;
        } else if (name == "--no-postselect-detectors") {
            options.postselect_detectors = false;
        } else if (name == "--gpu-exogenous") {
            options.sample_exogenous_on_device = true;
            options.gpu_presample_expressions = false;
            options.lazy_exogenous_on_device = false;
            options.gpu_on_demand_expressions = false;
        } else if (name == "--gpu-presample-expressions") {
            options.sample_exogenous_on_device = false;
            options.gpu_presample_expressions = true;
            options.lazy_exogenous_on_device = false;
            options.gpu_on_demand_expressions = false;
        } else if (name == "--gpu-on-demand-expressions") {
            options.sample_exogenous_on_device = false;
            options.gpu_presample_expressions = false;
            options.lazy_exogenous_on_device = false;
            options.gpu_on_demand_expressions = true;
        } else if (name == "--gpu-lazy-exogenous") {
            options.sample_exogenous_on_device = false;
            options.gpu_presample_expressions = false;
            options.lazy_exogenous_on_device = true;
            options.gpu_on_demand_expressions = false;
        } else if (name == "--cpu-presample-exogenous") {
            options.sample_exogenous_on_device = false;
            options.gpu_presample_expressions = false;
            options.lazy_exogenous_on_device = false;
            options.gpu_on_demand_expressions = false;
        } else if (name == "--print-program-stats") {
            options.print_program_stats = true;
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    return options;
}

void print_result(
    const Options& options,
    const symft::CircuitSamplingRunResult& run,
    double parse_s,
    double plan_s,
    int max_k) {
    const double shots_per_s = run.timing.sample_s > 0.0
                                   ? static_cast<double>(options.shots) / run.timing.sample_s
                                   : 0.0;
    const double discard_rate = run.counts.shots == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(run.counts.discarded) / static_cast<double>(run.counts.shots);
    const double logical_rate = run.counts.accepted == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(run.counts.logical_errors) / static_cast<double>(run.counts.accepted);

    std::cout << "sampler cuda\n";
    std::cout << "file " << options.path << "\n";
    std::cout << "shots " << options.shots << "\n";
    std::cout << "sampled_shots " << run.counts.shots << "\n";
    std::cout << "detector_postselection " << (options.postselect_detectors ? "enabled" : "disabled") << "\n";
    std::cout << "max_k " << max_k << "\n";
    std::cout << "shots_per_launch " << options.shots_per_launch << "\n";
    const int effective_threads_per_block =
        options.threads_per_block > 0
            ? options.threads_per_block
            : ((options.gpu_presample_expressions || options.gpu_on_demand_expressions)
                   ? (max_k <= 4 ? 32 : 128)
                   : 32);
    std::cout << "threads_per_block " << effective_threads_per_block << "\n";
    const char* exogenous_mode = options.sample_exogenous_on_device
                                     ? "gpu"
                                     : (options.gpu_presample_expressions
                                            ? "gpu_presampled_expressions"
                                            : (options.gpu_on_demand_expressions
                                                   ? "gpu_on_demand_expressions"
                                                   : (options.lazy_exogenous_on_device ? "gpu_lazy" : "cpu_presampled")));
    std::cout << "exogenous_sampling " << exogenous_mode << "\n";
    std::cout << "repeats " << options.repeats << "\n";
    std::cout << "sample_s_avg " << run.timing.sample_s << "\n";
    std::cout << "sample_shots_per_s " << shots_per_s << "\n";
    std::cout << "parse_s " << parse_s << "\n";
    std::cout << "plan_s " << plan_s << "\n";
    std::cout << "presample_s_avg " << run.timing.presample_s << "\n";
    std::cout << "execute_s_avg " << run.timing.execute_s << "\n";
    std::cout << "kernel_s_avg " << run.timing.accumulate_s << "\n";
    std::cout << "discarded " << run.counts.discarded << "\n";
    std::cout << "accepted " << run.counts.accepted << "\n";
    std::cout << "logical_errors " << run.counts.logical_errors << "\n";
    std::cout << "discard_rate " << discard_rate << "\n";
    std::cout << "logical_error_rate " << logical_rate << "\n";
}

void print_program_stats(const symft::FactoredInstructionProgram& program) {
    int active_rotations = 0;
    int promote_rotations = 0;
    int active_measurements = 0;
    int diagonal_measurements = 0;
    int nondiagonal_measurements = 0;
    int detectors = 0;
    int record_measurements = 0;
    int dormant_branches = 0;
    int rotation_runs = 0;
    int rotation_run_items = 0;
    int same_shape_adjacent = 0;
    int same_kind_adjacent = 0;
    int diagonal_adjacent = 0;
    int uniform_adjacent = 0;
    int real_pair_adjacent = 0;
    int general_adjacent = 0;
    int max_run = 0;

    auto same_shape = [](const symft::PrecomputedActivePauliRotationKernel& a,
                         const symft::PrecomputedActivePauliRotationKernel& b) {
        return a.is_diagonal == b.is_diagonal &&
               a.uniform_imag_pairs == b.uniform_imag_pairs &&
               a.real_pair_flip == b.real_pair_flip &&
               a.pair_bit == b.pair_bit &&
               a.action.xmask == b.action.xmask;
    };
    auto same_kind = [](const symft::PrecomputedActivePauliRotationKernel& a,
                        const symft::PrecomputedActivePauliRotationKernel& b) {
        return a.is_diagonal == b.is_diagonal &&
               a.uniform_imag_pairs == b.uniform_imag_pairs &&
               a.real_pair_flip == b.real_pair_flip;
    };

    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const auto& instruction = program.instructions[idx];
        if (const auto* rotation = std::get_if<symft::ApplyPrecomputedActivePauliRotation>(&instruction)) {
            (void)rotation;
            ++active_rotations;
        } else if (std::get_if<symft::PromoteDormantRotation>(&instruction) != nullptr) {
            ++promote_rotations;
        } else if (const auto* measurement = std::get_if<symft::MeasurePrecomputedActivePauli>(&instruction)) {
            ++active_measurements;
            if (measurement->kernel.is_diagonal) {
                ++diagonal_measurements;
            } else {
                ++nondiagonal_measurements;
            }
        } else if (std::get_if<symft::RecordDetector>(&instruction) != nullptr) {
            ++detectors;
        } else if (std::get_if<symft::RecordMeasurement>(&instruction) != nullptr) {
            ++record_measurements;
        } else if (std::get_if<symft::IntroduceDormantMeasurementBranch>(&instruction) != nullptr) {
            ++dormant_branches;
        }
    }

    for (std::size_t idx = 0; idx < program.instructions.size();) {
        const auto* first = std::get_if<symft::ApplyPrecomputedActivePauliRotation>(&program.instructions[idx]);
        if (first == nullptr) {
            ++idx;
            continue;
        }
        std::size_t end = idx + 1;
        while (end < program.instructions.size() &&
               std::get_if<symft::ApplyPrecomputedActivePauliRotation>(&program.instructions[end]) != nullptr) {
            ++end;
        }
        const int run_len = static_cast<int>(end - idx);
        if (run_len >= 2) {
            ++rotation_runs;
            rotation_run_items += run_len;
            max_run = std::max(max_run, run_len);
            for (std::size_t j = idx + 1; j < end; ++j) {
                const auto& prev = std::get<symft::ApplyPrecomputedActivePauliRotation>(program.instructions[j - 1]).rotation_kernel;
                const auto& curr = std::get<symft::ApplyPrecomputedActivePauliRotation>(program.instructions[j]).rotation_kernel;
                if (same_kind(prev, curr)) {
                    ++same_kind_adjacent;
                }
                if (same_shape(prev, curr)) {
                    ++same_shape_adjacent;
                    if (curr.is_diagonal) {
                        ++diagonal_adjacent;
                    } else if (curr.uniform_imag_pairs) {
                        ++uniform_adjacent;
                    } else if (curr.real_pair_flip) {
                        ++real_pair_adjacent;
                    } else {
                        ++general_adjacent;
                    }
                }
            }
        }
        idx = end;
    }

    std::cout << "instructions " << program.instructions.size() << "\n";
    std::cout << "max_k " << program.max_k << "\n";
    std::cout << "active_rotations " << active_rotations << "\n";
    std::cout << "promote_rotations " << promote_rotations << "\n";
    std::cout << "active_measurements " << active_measurements << "\n";
    std::cout << "diagonal_measurements " << diagonal_measurements << "\n";
    std::cout << "nondiagonal_measurements " << nondiagonal_measurements << "\n";
    std::cout << "detectors " << detectors << "\n";
    std::cout << "record_measurements " << record_measurements << "\n";
    std::cout << "dormant_branches " << dormant_branches << "\n";
    std::cout << "rotation_runs " << rotation_runs << "\n";
    std::cout << "rotation_run_items " << rotation_run_items << "\n";
    std::cout << "max_rotation_run " << max_run << "\n";
    std::cout << "same_kind_adjacent_in_runs " << same_kind_adjacent << "\n";
    std::cout << "same_shape_adjacent_in_runs " << same_shape_adjacent << "\n";
    std::cout << "diagonal_same_shape_adjacent " << diagonal_adjacent << "\n";
    std::cout << "uniform_same_shape_adjacent " << uniform_adjacent << "\n";
    std::cout << "real_pair_same_shape_adjacent " << real_pair_adjacent << "\n";
    std::cout << "general_same_shape_adjacent " << general_adjacent << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

        symft::CircuitSamplingOptions frontend_options;
        frontend_options.observable = options.observable;
        const auto input = symft::make_stim_circuit_sampling_input_from_file(options.path, frontend_options);
        if (options.print_program_stats) {
            print_program_stats(input.program);
            return 0;
        }

        symft::cuda::CudaSamplingOptions cuda_options;
        cuda_options.postselect_detectors = options.postselect_detectors;
        cuda_options.sample_exogenous_on_device = options.sample_exogenous_on_device;
        cuda_options.gpu_presample_expressions = options.gpu_presample_expressions;
        cuda_options.lazy_exogenous_on_device = options.lazy_exogenous_on_device;
        cuda_options.gpu_on_demand_expressions = options.gpu_on_demand_expressions;
        cuda_options.shots_per_launch = options.shots_per_launch;
        cuda_options.threads_per_block = options.threads_per_block;
        symft::cuda::PreparedCircuitCudaSampler sampler(input, cuda_options);

        symft::CircuitSamplingRunResult total;
        for (int repeat = 0; repeat < options.repeats; ++repeat) {
            const auto run = sampler.sample(options.shots, static_cast<std::uint64_t>(repeat));
            total.counts.shots += run.counts.shots;
            total.counts.discarded += run.counts.discarded;
            total.counts.accepted += run.counts.accepted;
            total.counts.logical_errors += run.counts.logical_errors;
            total.timing.presample_s += run.timing.presample_s;
            total.timing.execute_s += run.timing.execute_s;
            total.timing.accumulate_s += run.timing.accumulate_s;
            total.timing.sample_s += run.timing.sample_s;
        }
        const double inv_repeats = 1.0 / static_cast<double>(options.repeats);
        total.timing.presample_s *= inv_repeats;
        total.timing.execute_s *= inv_repeats;
        total.timing.accumulate_s *= inv_repeats;
        total.timing.sample_s *= inv_repeats;
        print_result(
            options,
            total,
            input.preprocessing_timing.parse_s,
            input.preprocessing_timing.plan_s,
            sampler.info().max_k);
    } catch (const std::exception& ex) {
        std::cerr << "symft_cuda_rate_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
