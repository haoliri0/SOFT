#include "frontend/stim_prepared_sampler.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

enum class SamplerMode {
    Single,
    Batch,
    Both,
};

enum class ActiveComponentMode {
    Auto,
    Enabled,
    Disabled,
};

struct BenchResult {
    std::string path;
    std::string sampler;
    std::uint64_t requested_shots = 0;
    symft::CircuitSamplingCounts counts;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int repeats = 1;
    int threads = 1;
    int requested_threads = 1;
    bool active_components = false;
    bool detector_postselection = false;
    int batch_mask_threshold_denominator = 0;
    symft::CircuitSamplingTiming timing;
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
    return static_cast<std::uint64_t>(value);
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

bool parse_bool_flag(const char* raw, const char* name) {
    const std::string value(raw);
    if (value == "1" || value == "true" || value == "yes" || value == "on" || value == "postselect") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off" || value == "none") {
        return false;
    }
    throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
}

int parse_batch_size(const char* raw) {
    if (std::string(raw) == "auto") {
        return 0;
    }
    return parse_positive_int(raw, "batch_size");
}

int parse_sample_chunk_shots(const char* raw) {
    const std::string value(raw);
    if (value == "auto" || value == "none" || value == "off") {
        return 0;
    }
    return parse_positive_int(raw, "sample_chunk_shots");
}

int parse_threads(const char* raw) {
    const std::string value(raw);
    if (value == "auto") {
        const unsigned int detected = std::thread::hardware_concurrency();
        return std::max(1, static_cast<int>(detected == 0 ? 1 : detected));
    }
    return parse_positive_int(raw, "threads");
}

SamplerMode parse_sampler_mode(const char* raw) {
    const std::string value(raw);
    if (value == "single" || value == "single-shot" || value == "single_shot") {
        return SamplerMode::Single;
    }
    if (value == "batch" || value == "batched") {
        return SamplerMode::Batch;
    }
    if (value == "both" || value == "all") {
        return SamplerMode::Both;
    }
    throw std::runtime_error(std::string("invalid sampler: ") + raw);
}

ActiveComponentMode parse_active_component_mode(const char* raw) {
    const std::string value(raw);
    if (value == "auto") {
        return ActiveComponentMode::Auto;
    }
    if (value == "1" || value == "on" || value == "enabled") {
        return ActiveComponentMode::Enabled;
    }
    if (value == "0" || value == "off" || value == "disabled" ||
        value == "dense") {
        return ActiveComponentMode::Disabled;
    }
    throw std::runtime_error(
        std::string("invalid active-components mode: ") + raw);
}

struct Options {
    std::string path = "benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim";
    std::uint64_t shots = 100000000ULL;
    int batch_size = 0;
    int sample_chunk_shots = 0;
    int repeats = 1;
    int observable = 0;
    int threads = 1;
    bool postselect_detectors = false;
    int batch_mask_threshold_denominator = 2;
    SamplerMode sampler = SamplerMode::Batch;
    ActiveComponentMode active_components = ActiveComponentMode::Auto;
};

void print_usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " --circuit PATH --shots N [options]\n"
        << "       " << argv0 << " --sampler batch --circuit benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim --shots 1000000 --batch-size auto --postselect-detectors\n"
        << "       " << argv0 << " --sampler both --circuit benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim --shots 1000000 --batch-size auto\n"
        << "\n"
        << "common options:\n"
        << "  --circuit PATH, --file PATH        Stim circuit path\n"
        << "  --shots N                         Number of shots\n"
        << "  --sampler single|batch|both       Sampler to benchmark; default: batch\n"
        << "  --sample-chunk-shots N|auto       Presample requested shots in sampler-owned chunks; default: auto\n"
        << "  --repeats N                       Number of repeats\n"
        << "  --observable N                    Observable index\n"
        << "  --postselect-detectors            Enable detector postselection inside the sampler\n"
        << "  --no-postselect-detectors         Disable detector postselection\n"
        << "  --threads N|auto                  Parallel worker threads over independent chunks\n"
        << "  --active-components auto|on|off   Override automatic product-component selection\n"
        << "\n"
        << "batch-only options:\n"
        << "  --batch-size N|auto               Batch size; auto chooses the sampler default\n"
        << "  --batch-mask-threshold-denominator N  Batch postselection compaction threshold; default: 2\n";
}

std::string require_option_value(
    const std::string& option,
    const std::string& inline_value,
    int& idx,
    int argc,
    char** argv) {
    if (!inline_value.empty()) {
        return inline_value;
    }
    if (idx + 1 >= argc) {
        throw std::runtime_error(option + " requires a value");
    }
    return argv[++idx];
}

void apply_positional_option(Options& options, int position, const char* value) {
    switch (position) {
    case 0:
        options.path = value;
        break;
    case 1:
        options.shots = parse_u64(value, "shots");
        break;
    case 2:
        options.batch_size = parse_batch_size(value);
        break;
    case 3:
        options.repeats = parse_positive_int(value, "repeats");
        break;
    case 4:
        options.observable = parse_nonnegative_int(value, "observable");
        break;
    case 5:
        options.postselect_detectors = parse_bool_flag(value, "postselect_detectors");
        break;
    default:
        throw std::runtime_error("too many positional arguments");
    }
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
            apply_positional_option(options, positional++, argv[idx]);
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
            options.path = require_option_value(name, value, idx, argc, argv);
        } else if (name == "--shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.shots = parse_u64(raw.c_str(), "shots");
        } else if (name == "--sampler") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.sampler = parse_sampler_mode(raw.c_str());
        } else if (name == "--batch-size") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.batch_size = parse_batch_size(raw.c_str());
        } else if (name == "--sample-chunk-shots") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.sample_chunk_shots = parse_sample_chunk_shots(raw.c_str());
        } else if (name == "--repeats") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.repeats = parse_positive_int(raw.c_str(), "repeats");
        } else if (name == "--observable") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.observable = parse_nonnegative_int(raw.c_str(), "observable");
        } else if (name == "--threads") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.threads = parse_threads(raw.c_str());
        } else if (name == "--active-components") {
            const std::string raw =
                require_option_value(name, value, idx, argc, argv);
            options.active_components =
                parse_active_component_mode(raw.c_str());
        } else if (name == "--postselect-detectors") {
            options.postselect_detectors = value.empty() ? true : parse_bool_flag(value.c_str(), "postselect_detectors");
        } else if (name == "--no-postselect-detectors") {
            options.postselect_detectors = false;
        } else if (name == "--batch-mask-threshold-denominator") {
            const std::string raw = require_option_value(name, value, idx, argc, argv);
            options.batch_mask_threshold_denominator =
                parse_positive_int(raw.c_str(), "batch_mask_threshold_denominator");
        } else {
            throw std::runtime_error("unknown option: " + name);
        }
    }
    return options;
}

symft::CircuitSamplingOptions make_sampler_options(const Options& options) {
    symft::CircuitSamplingOptions out;
    out.observable = options.observable;
    out.postselect_detectors = options.postselect_detectors;
    out.sample_chunk_shots = options.sample_chunk_shots;
    out.batch_size = options.batch_size;
    out.batch_mask_threshold_denominator = options.batch_mask_threshold_denominator;
    out.threads = options.threads;
    return out;
}

symft::CircuitSamplingInput make_benchmark_input(
    const Options& options,
    const symft::CircuitSamplingOptions& sampler_options) {
    auto input =
        symft::make_stim_circuit_sampling_input_from_file(
            options.path,
            sampler_options);
    if (options.active_components == ActiveComponentMode::Enabled) {
        input.program.use_active_components = true;
    } else if (
        options.active_components == ActiveComponentMode::Disabled) {
        input.program.use_active_components = false;
    }
    return input;
}

BenchResult make_result_shell(
    const Options& options,
    const symft::CircuitSamplingInfo& info,
    const std::string& sampler_name) {
    BenchResult result;
    result.path = options.path;
    result.sampler = sampler_name;
    result.requested_shots = options.shots;
    result.batch_size = info.batch_size;
    result.sample_chunk_shots = info.sample_chunk_shots;
    result.repeats = options.repeats;
    result.threads = info.threads;
    result.requested_threads = options.threads;
    result.active_components = info.active_components;
    result.detector_postselection = info.detector_postselection;
    result.batch_mask_threshold_denominator = info.batch_mask_threshold_denominator;
    return result;
}

void accumulate_run(BenchResult& result, const symft::CircuitSamplingRunResult& run) {
    result.counts.shots += run.counts.shots;
    result.counts.discarded += run.counts.discarded;
    result.counts.accepted += run.counts.accepted;
    result.counts.logical_errors += run.counts.logical_errors;
    result.timing.presample_s += run.timing.presample_s;
    result.timing.execute_s += run.timing.execute_s;
    result.timing.sample_s += run.timing.sample_s;
    result.threads = run.active_threads;
}

void average_repeated_timings(BenchResult& result) {
    const double inv_repeats = 1.0 / static_cast<double>(result.repeats);
    result.timing.presample_s *= inv_repeats;
    result.timing.execute_s *= inv_repeats;
    result.timing.sample_s *= inv_repeats;
}

BenchResult run_single_sampler(const Options& options) {
    const std::string sampler_name = options.postselect_detectors ? "single_postselected" : "single";
    const auto sampler_options = make_sampler_options(options);
    auto sampler = symft::PreparedCircuitSingleShotSampler(
        make_benchmark_input(options, sampler_options),
        sampler_options);
    BenchResult result = make_result_shell(
        options,
        sampler.info(),
        sampler_name);

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        accumulate_run(result, sampler.sample(options.shots, repeat));
    }
    average_repeated_timings(result);
    return result;
}

BenchResult run_batch_sampler(const Options& options) {
    const std::string sampler_name = options.postselect_detectors ? "batch_postselected" : "batch";
    const auto sampler_options = make_sampler_options(options);
    auto sampler = symft::PreparedCircuitBatchSampler(
        make_benchmark_input(options, sampler_options),
        sampler_options);
    BenchResult result = make_result_shell(
        options,
        sampler.info(),
        sampler_name);

    for (int repeat = 0; repeat < options.repeats; ++repeat) {
        accumulate_run(result, sampler.sample(options.shots, repeat));
    }
    average_repeated_timings(result);
    return result;
}

void print_result(const BenchResult& result) {
    const bool is_batch_sampler = result.sampler == "batch" || result.sampler == "batch_postselected";
    const double shots_per_s = result.timing.sample_s > 0.0
                                   ? static_cast<double>(result.requested_shots) / result.timing.sample_s
                                   : 0.0;
    const double discard_rate = result.counts.shots == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.discarded) / static_cast<double>(result.counts.shots);
    const double logical_rate = result.counts.accepted == 0
                                    ? std::numeric_limits<double>::quiet_NaN()
                                    : static_cast<double>(result.counts.logical_errors) / static_cast<double>(result.counts.accepted);

    std::cout << "sampler " << result.sampler << "\n";
    std::cout << "file " << result.path << "\n";
    std::cout << "shots " << result.requested_shots << "\n";
    std::cout << "sampled_shots " << result.counts.shots << "\n";
    std::cout << "active_components "
              << (result.active_components ? "enabled" : "dense_fallback")
              << "\n";
    std::cout << "detector_postselection " << (result.detector_postselection ? "enabled" : "disabled") << "\n";
    if (is_batch_sampler) {
        std::cout << "batch_size " << result.batch_size << "\n";
    }
    std::cout << "sample_chunk_shots " << result.sample_chunk_shots << "\n";
    std::cout << "repeats " << result.repeats << "\n";
    std::cout << "threads " << result.threads << "\n";
    if (result.requested_threads != result.threads) {
        std::cout << "requested_threads " << result.requested_threads << "\n";
    }
    if (result.sampler == "batch_postselected") {
        std::cout << "batch_mask_threshold_denominator "
                  << result.batch_mask_threshold_denominator << "\n";
    }
    std::cout << "sample_s_avg " << result.timing.sample_s << "\n";
    std::cout << "sample_shots_per_s " << shots_per_s << "\n";
    if (result.threads == 1) {
        std::cout << "presample_s_avg " << result.timing.presample_s << "\n";
        std::cout << "execute_s_avg " << result.timing.execute_s << "\n";
    }
    std::cout << "discarded " << result.counts.discarded << "\n";
    std::cout << "accepted " << result.counts.accepted << "\n";
    std::cout << "logical_errors " << result.counts.logical_errors << "\n";
    std::cout << "discard_rate " << discard_rate << "\n";
    std::cout << "logical_error_rate " << logical_rate << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.sampler == SamplerMode::Single) {
            print_result(run_single_sampler(options));
        } else if (options.sampler == SamplerMode::Batch) {
            print_result(run_batch_sampler(options));
        } else {
            print_result(run_single_sampler(options));
            std::cout << "\n";
            print_result(run_batch_sampler(options));
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_rate_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
