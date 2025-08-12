#include <ctime>
#include <cstdio>
#include <iostream>
#include <functional>

#include "./datatype.cuh"
#include "./simulator.hpp"
#include "./print.hpp"
#include "./read.hpp"

using namespace StnCuda;

// cli args

struct CliArgs {
    Sid shots_n = 1;
    Qid qubits_n = 4;
    Kid amps_m = 4;
    Rid results_m = 4;
    unsigned long long seed = 42;
    unsigned int mode = 1;
};

enum class ParseCliArgsError {
    Success,
    IllegalArg,
    IllegalKey,
    IllegalValue,
};

static
ParseCliArgsError parse_cli_args(const int argc, const char **argv, CliArgs &args) {
    for (unsigned int i = 1; i < argc;) {
        const char *arg_key = argv[i++];
        if (match(arg_key, "--shots_n")) {
            const char *arg_value = argv[i++];
            args.shots_n = strtoul(arg_value, nullptr, 10);
            if (args.shots_n == 0) {
                fprintf(stderr, "Illegal value: shots_n=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (match(arg_key, "--qubits_n")) {
            const char *arg_value = argv[i++];
            args.qubits_n = strtoul(arg_value, nullptr, 10);
            if (args.qubits_n == 0) {
                fprintf(stderr, "Illegal value: qubits_n=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (match(arg_key, "--amps_m")) {
            const char *arg_value = argv[i++];
            args.amps_m = strtoul(arg_value, nullptr, 10);
            if (args.amps_m == 0) {
                fprintf(stderr, "Illegal value: amps_m=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (match(arg_key, "--results_m")) {
            const char *arg_value = argv[i++];
            args.results_m = strtoul(arg_value, nullptr, 10);
            if (args.results_m == 0) {
                fprintf(stderr, "Illegal value: results_m=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (match(arg_key, "--seed")) {
            const char *arg_value = argv[i++];
            args.seed = strtoull(arg_value, nullptr, 10);
            continue;
        }
        if (match(arg_key, "--mode")) {
            const char *arg_value = argv[i++];
            args.mode = strtoul(arg_value, nullptr, 10);
            if (args.mode > 2) {
                fprintf(stderr, "Illegal value: mode=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (match_head(arg_key, "-")) {
            fprintf(stderr, "Unrecognized option: %s\n", arg_key);
            return ParseCliArgsError::IllegalKey;
        }
        fprintf(stderr, "Unexpected arg: %s\n", arg_key);
        return ParseCliArgsError::IllegalArg;
    }
    return ParseCliArgsError::Success;
}

// circuit ops

enum class ParseCircuitLineError {
    Success,
    IOError,
    IllegalOp,
    IllegalArg,
    ExecFailed,
};

static
ParseCircuitLineError read_arg(std::istream &istream, int &arg) {
    skip_whitespace(istream);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    istream >> arg;
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    return ParseCircuitLineError::Success;
}

static
ParseCircuitLineError read_arg(std::istream &istream, Qid &arg) {
    skip_whitespace(istream);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    istream >> arg;
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    return ParseCircuitLineError::Success;
}

static
ParseCircuitLineError read_arg(std::istream &istream, Flt &arg) {
    skip_whitespace(istream);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    istream >> arg;
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    return ParseCircuitLineError::Success;
}

static
ParseCircuitLineError read_arg(std::istream &istream, Bit &arg) {
    skip_whitespace(istream);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    unsigned int value;
    istream >> value;
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    if (value != 0 && value != 1) {
        istream.setstate(std::istream::failbit);
        return ParseCircuitLineError::IllegalArg;
    }

    arg = value;
    return ParseCircuitLineError::Success;
}

static
ParseCircuitLineError execute_op(
    std::istream &istream,
    const std::function<ParseCircuitLineError()> &op
) noexcept {
    const ParseCircuitLineError error = op();
    skip_whitespace_line(istream);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;
    return error;
}

static
ParseCircuitLineError execute_op(
    std::istream &istream,
    const std::function<void()> &op
) noexcept {
    op();

    skip_whitespace_line(istream);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.fail()) return ParseCircuitLineError::IllegalArg;

    return ParseCircuitLineError::Success;
}

template<typename Ret, typename Arg0, typename... Args>
static
ParseCircuitLineError execute_op(
    std::istream &istream,
    const std::function<Ret (Arg0, Args...)> &op
) noexcept {
    Arg0 arg0;
    const ParseCircuitLineError error = read_arg(istream, arg0);
    if (error != ParseCircuitLineError::Success) return error;

    const std::function wrapped = [op, arg0](Args... args) { return op(arg0, args...); };
    return execute_op(istream, wrapped);
}

template<typename Receiver, typename... Args>
static
ParseCircuitLineError execute_op(
    std::istream &istream,
    const Receiver receiver,
    void (Receiver::*op)(Args...) const noexcept
) noexcept {
    std::function wrapped = [receiver, op](Args... args) { (receiver.*op)(args...); };
    return execute_op(istream, wrapped);
}

ParseCircuitLineError perform_read_op(
    const Simulator &simulator,
    const int result_i
) {
    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const Kid results_m = shots_state_ptr.results_m;
    const Sid shots_n = shots_state_ptr.shots_n;

    cudaError cuda_err = cudaStreamSynchronize(simulator.stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs when executing previous operations!\n");
        fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
        return ParseCircuitLineError::ExecFailed;
    }

    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

        Err error;
        cuda_err = cudaMemcpy(&error, shot_state_ptr.get_results_ptr().get_error_ptr(),
            sizeof(Err), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) break;

        Rvl results_value[results_m];
        cuda_err = cudaMemcpy(results_value, shot_state_ptr.get_results_ptr().get_values_ptr(),
            results_m * sizeof(Rvl), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) break;

        Flt results_prob[results_m];
        cuda_err = cudaMemcpy(results_prob, shot_state_ptr.get_results_ptr().get_probs_ptr(),
            results_m * sizeof(Flt), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) break;

        Rid result_j;
        if (result_i < 0) {
            Rid results_n;
            cuda_err = cudaMemcpy(&results_n, shot_state_ptr.get_results_ptr().get_results_n_ptr(),
                sizeof(Rid), cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) break;
            result_j = (results_n + result_i) % results_m;
        } else {
            result_j = result_i % results_m;
        }

        const Rvl value = results_value[result_j];
        const Flt prob = results_prob[result_j];
        printf("%u,%u,%u,%f\n", shot_i, error, value, prob);
    }

    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs when reading result %d!\n", result_i);
        fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
        return ParseCircuitLineError::ExecFailed;
    }

    return ParseCircuitLineError::Success;
}

static
ParseCircuitLineError execute_line(
    const Simulator &simulator,
    std::istream &istream
) noexcept {
    constexpr size_t name_limit = 16;
    char name[name_limit];
    read_word(istream, name_limit, name);
    if (istream.bad()) return ParseCircuitLineError::IOError;
    if (istream.eof()) return ParseCircuitLineError::Success;

    if (match(name, ""))
        return execute_op(istream, [] {});
    if (match(name, "X"))
        return execute_op(istream, simulator, &Simulator::apply_x);
    if (match(name, "Y"))
        return execute_op(istream, simulator, &Simulator::apply_y);
    if (match(name, "Z"))
        return execute_op(istream, simulator, &Simulator::apply_z);
    if (match(name, "H"))
        return execute_op(istream, simulator, &Simulator::apply_h);
    if (match(name, "S"))
        return execute_op(istream, simulator, &Simulator::apply_s);
    if (match(name, "SDG"))
        return execute_op(istream, simulator, &Simulator::apply_sdg);
    if (match(name, "T"))
        return execute_op(istream, simulator, &Simulator::apply_t);
    if (match(name, "TDG"))
        return execute_op(istream, simulator, &Simulator::apply_tdg);
    if (match(name, "CX"))
        return execute_op(istream, simulator, &Simulator::apply_cx);

    if (match(name, "M"))
        return execute_op(istream, simulator, &Simulator::apply_measure);
    if (match(name, "D"))
        return execute_op(istream, simulator, &Simulator::apply_desire);
    if (match(name, "R"))
        return execute_op(istream, std::function([simulator](const Qid target) {
            simulator.apply_assign(target, false);
        }));
    if (match(name, "XERR"))
        return execute_op(istream, simulator, &Simulator::apply_noise_x);
    if (match(name, "ZERR"))
        return execute_op(istream, simulator, &Simulator::apply_noise_z);
    if (match(name, "DEP1"))
        return execute_op(istream, simulator, &Simulator::apply_noise_depo1);
    if (match(name, "DEP2"))
        return execute_op(istream, simulator, &Simulator::apply_noise_depo2);

    if (match(name, "READ"))
        return execute_op(istream, std::function([simulator](const int result_i) {
            return perform_read_op(simulator, result_i);
        }));

    return ParseCircuitLineError::IllegalOp;
}

// main

int main(const int argc, const char **argv) {
    CliArgs args{};
    ParseCliArgsError const args_err = parse_cli_args(argc, argv, args);
    if (args_err != ParseCliArgsError::Success) return 1;
    fprintf(stderr, "Cli Args:\n");
    fprintf(stderr, "\tshots_n=%u\n", args.shots_n);
    fprintf(stderr, "\tqubits_n=%u\n", args.qubits_n);
    fprintf(stderr, "\tamps_m=%u\n", args.amps_m);
    fprintf(stderr, "\tresults_m=%u\n", args.results_m);
    fprintf(stderr, "\trseed=%llu\n", args.seed);

    Simulator simulator;
    cudaError cuda_err = cudaSuccess;
    auto line_err = ParseCircuitLineError::Success;
    do {
        fprintf(stderr, "Creating Simulator\n");

        cuda_err = simulator.create(args.shots_n, args.qubits_n, args.amps_m, args.results_m, args.seed);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Error occurs when creating simulator.\n");
            fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
            break;
        }

        cuda_err = cudaStreamSynchronize(simulator.stream);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Error occurs when creating simulator.\n");
            fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
            break;
        }

        fprintf(stderr, "Executing Circuit\n");
        const clock_t time_start = clock();

        size_t lines_n = 0;
        std::istream &istream = std::cin;
        istream >> std::noskipws;
        while (istream.good()) {
            line_err = execute_line(simulator, istream);
            if (line_err != ParseCircuitLineError::Success) {
                fprintf(stderr, "Error occurs when parsing line %lu.\n", lines_n + 1);
                break;
            }

            if (args.mode >= 2)
                print_simulator(simulator);

            lines_n += 1;
        }

        if (line_err != ParseCircuitLineError::Success)
            break;

        const clock_t time_end = clock();
        const clock_t time_span = time_end - time_start;
        const float time_span_seconds = static_cast<float>(time_span) / CLOCKS_PER_SEC;
        const float shots_per_second = static_cast<float>(args.shots_n) / time_span_seconds;

        fprintf(stderr, "Finished Circuit\n");
        fprintf(stderr, "\tspan_time: %f s\n", time_span_seconds);
        fprintf(stderr, "\tavg_speed: %f shot/s\n", shots_per_second);
    } while (false);

    simulator.destroy();

    if (line_err != ParseCircuitLineError::Success)
        return -2;

    if (cuda_err != cudaSuccess)
        return -3;

    return 0;
}
