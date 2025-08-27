#include <ctime>
#include <cstdio>
#include <iostream>
#include <functional>

#include "./datatype.cuh"
#include "./simulator.hpp"
#include "./cleaner.hpp"
#include "./print.hpp"
#include "./read.hpp"

using namespace StnCuda;

// cli args

struct CliArgs {
    Sid shots_n = 1;
    Qid qubits_n = 4;
    Eid entries_m = 4;
    Rid results_m = 4;
    unsigned long long seed = 42;
};

static
unsigned long parse_cli_ul(const char *s) {
    char *ptr;
    const unsigned long v = strtoul(s, &ptr, 10);

    if (errno != 0 || ptr == s || *ptr != '\0') {
        fprintf(stderr, "Failed to parse '%s' with errno=%d", s,errno);
        throw CliArgsException(CliArgsError::IllegalValue);
    }

    return v;
}

static
unsigned long long parse_cli_ull(const char *s) {
    char *ptr;
    const unsigned long long v = strtoull(s, &ptr, 10);

    if (errno != 0 || ptr == s || *ptr != '\0') {
        fprintf(stderr, "Failed to parse '%s' with errno=%d", s,errno);
        throw CliArgsException(CliArgsError::IllegalValue);
    }

    return v;
}

static
void parse_cli_args(const int argc, const char **argv, CliArgs &args) {
    for (unsigned int i = 1; i < argc;) {
        const char *arg_key = argv[i++];
        if (match(arg_key, "--shots_n")) {
            const char *arg_value = argv[i++];
            args.shots_n = parse_cli_ul(arg_value);
            if (args.shots_n == 0) {
                fprintf(stderr, "Illegal value: shots_n=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--qubits_n")) {
            const char *arg_value = argv[i++];
            args.qubits_n = parse_cli_ul(arg_value);
            if (args.qubits_n == 0) {
                fprintf(stderr, "Illegal value: qubits_n=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--entries_m")) {
            const char *arg_value = argv[i++];
            args.entries_m = parse_cli_ul(arg_value);
            if (args.entries_m == 0) {
                fprintf(stderr, "Illegal value: entries_m=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--results_m")) {
            const char *arg_value = argv[i++];
            args.results_m = parse_cli_ul(arg_value);
            if (args.results_m == 0) {
                fprintf(stderr, "Illegal value: results_m=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--seed")) {
            const char *arg_value = argv[i++];
            args.seed = parse_cli_ull(arg_value);
            continue;
        }
        if (match_head(arg_key, "-")) {
            fprintf(stderr, "Unrecognized option: %s\n", arg_key);
            throw CliArgsException(CliArgsError::IllegalKey);
        }
        fprintf(stderr, "Unexpected arg: %s\n", arg_key);
        throw CliArgsException(CliArgsError::IllegalArg);
    }
}

// circuit ops

static
void execute_op(
    std::istream &istream,
    const std::function<void()> &op
) {
    op();
    skip_whitespace_line(istream);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);
}

template<typename Ret>
static
Ret execute_op(
    std::istream &istream,
    const std::function<Ret()> &op
) {
    const Ret ret = op();
    skip_whitespace_line(istream);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.fail()) throw ExecException(ExecError::IllegalArg);
    return ret;
}

template<typename Ret, typename Arg0, typename... Args>
static
Ret execute_op(
    std::istream &istream,
    const std::function<Ret (Arg0, Args...)> &op
) {
    Arg0 arg0;
    read_arg(istream, arg0);
    const std::function wrapped = [op, arg0](Args... args) { return op(arg0, args...); };
    return execute_op(istream, wrapped);
}

template<typename Receiver, typename... Args>
static
void execute_op(
    std::istream &istream,
    const Receiver &receiver,
    void (Receiver::*op)(Args...) const
) {
    std::function wrapped = [&receiver, op](Args... args) { (receiver.*op)(args...); };
    return execute_op(istream, wrapped);
}

template<typename Receiver, typename... Args>
static
void execute_op(
    std::istream &istream,
    const Receiver &receiver,
    void (*op)(const Receiver &, Args...)
) {
    std::function wrapped = [&receiver, op](Args... args) { (*op)(receiver, args...); };
    return execute_op(istream, wrapped);
}

void perform_read_op(
    const Simulator &simulator,
    const int result_i
) {
    cuda_check(cudaStreamSynchronize(simulator.stream));

    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const Rid results_m = shots_state_ptr.results_m;
    const Sid shots_n = shots_state_ptr.shots_n;
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

        Err error;
        cuda_check(cudaMemcpy(&error, shot_state_ptr.get_error_ptr(),
            sizeof(Err), cudaMemcpyDeviceToHost));

        auto const results_value = new Rvl[results_m];
        Cleaner results_value_cleaner([results_value] { delete[] results_value; });
        cuda_check(cudaMemcpy(results_value, shot_state_ptr.get_results_ptr().get_values_ptr(),
            results_m * sizeof(Rvl), cudaMemcpyDeviceToHost));

        auto const results_prob = new Flt[results_m];
        Cleaner results_prob_cleaner([results_prob] { delete[] results_prob; });
        cuda_check(cudaMemcpy(results_prob, shot_state_ptr.get_results_ptr().get_probs_ptr(),
            results_m * sizeof(Flt), cudaMemcpyDeviceToHost));

        Rid result_j;
        if (result_i < 0) {
            Rid results_n;
            cuda_check(cudaMemcpy(&results_n, shot_state_ptr.get_results_ptr().get_results_n_ptr(),
                sizeof(Rid), cudaMemcpyDeviceToHost));
            result_j = (results_n + result_i) % results_m;
        } else {
            result_j = result_i % results_m;
        }

        const Rvl value = results_value[result_j];
        const Flt prob = results_prob[result_j];
        printf("%u,%u,%u,%f\n", shot_i, error, value, prob);
    }
}

void perform_state_op(
    const Simulator &simulator
) {
    cuda_check(cudaStreamSynchronize(simulator.stream));

    auto const buffer = new char[simulator.shots_state_ptr.get_size_bytes_n()];
    Cleaner buffer_cleaner([buffer] { delete[] buffer; });
    cuda_check(cudaMemcpy(buffer, simulator.shots_state_ptr.ptr,
        simulator.shots_state_ptr.get_size_bytes_n(), cudaMemcpyDeviceToHost));
    ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    shots_state_ptr.ptr = buffer;

    printf("state:\n");
    for (Sid shot_i = 0; shot_i < shots_state_ptr.shots_n; ++shot_i) {
        ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
        print_indent(1);
        printf("shot %u:\n", shot_i);
        print_error(*shot_state_ptr.get_error_ptr(), 2);
        print_table(shot_state_ptr.get_table_ptr(), 2);
        print_entries(shot_state_ptr.get_entries_ptr(), false, 2);
    }
}

static
void execute_line(
    const Simulator &simulator,
    std::istream &istream
) {
    constexpr size_t name_limit = 16;
    char name[name_limit];
    read_word(istream, name_limit, name);
    if (istream.bad()) throw ExecException(ExecError::IOError);
    if (istream.eof()) return;

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

    if (match(name, "MEASURE"))
        return execute_op(istream, simulator, &Simulator::apply_measure);
    if (match(name, "DESIRE"))
        return execute_op(istream, simulator, &Simulator::apply_desire);
    if (match(name, "RESET"))
        return execute_op(istream, simulator, &Simulator::apply_reset);

    if (match(name, "XERR"))
        return execute_op(istream, simulator, &Simulator::apply_noise_x);
    if (match(name, "ZERR"))
        return execute_op(istream, simulator, &Simulator::apply_noise_z);
    if (match(name, "DEP1"))
        return execute_op(istream, simulator, &Simulator::apply_noise_depo1);
    if (match(name, "DEP2"))
        return execute_op(istream, simulator, &Simulator::apply_noise_depo2);

    if (match(name, "READ"))
        return execute_op(istream, simulator, perform_read_op);
    if (match(name, "STATE"))
        return execute_op(istream, simulator, perform_state_op);

    fprintf(stderr, "Unknown op: %s\n", name);
    throw ExecException(ExecError::IllegalOp);
}

// main

int main(const int argc, const char **argv) {
    CliArgs args{};
    parse_cli_args(argc, argv, args);
    fprintf(stderr, "args:\n");
    fprintf(stderr, "  shots_n=%u\n", args.shots_n);
    fprintf(stderr, "  qubits_n=%u\n", args.qubits_n);
    fprintf(stderr, "  entries_m=%u\n", args.entries_m);
    fprintf(stderr, "  results_m=%u\n", args.results_m);
    fprintf(stderr, "  seed=%llu\n", args.seed);
    fprintf(stderr, "\n");

    Simulator simulator;
    Cleaner simulator_cleaner([&simulator] { simulator.destroy(); });

    fprintf(stderr, "Creating\n");
    cudaError cuda_err = cudaSuccess;

    cuda_err = simulator.create(args.shots_n, args.qubits_n, args.entries_m, args.results_m, args.seed);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs when creating simulator.\n");
        fprint_cuda_error(stderr, cuda_err);
        throw CudaException(cuda_err);
    }

    cuda_err = cudaStreamSynchronize(simulator.stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs when creating simulator.\n");
        fprint_cuda_error(stderr, cuda_err);
        throw CudaException(cuda_err);
    }

    fprintf(stderr, "Executing\n");
    const clock_t time_start = clock();

    size_t lines_n = 0;
    std::istream &istream = std::cin;
    istream >> std::noskipws;
    while (istream.good()) {
        try {
            execute_line(simulator, istream);
        } catch (CudaException &exception) {
            fprintf(stderr, "Error occurs when parsing line %lu.\n", lines_n + 1);
            fprint_cuda_error(stderr, exception.error);
            throw;
        } catch (ExecException &exception) {
            fprintf(stderr, "Error occurs when parsing line %lu.\n", lines_n + 1);
            fprint_exec_error(stderr, exception.error);
            throw;
        }
        lines_n += 1;
    }

    cuda_err = cudaStreamSynchronize(simulator.stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs when executing circuit.\n");
        fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
        throw CudaException(cuda_err);
    }

    const clock_t time_end = clock();
    const clock_t time_span = time_end - time_start;
    const float time_span_seconds = static_cast<float>(time_span) / CLOCKS_PER_SEC;
    const float shots_per_second = static_cast<float>(args.shots_n) / time_span_seconds;

    fprintf(stderr, "Finished\n");
    fprintf(stderr, "performance:\n");
    fprintf(stderr, "  span_time=%f s\n", time_span_seconds);
    fprintf(stderr, "  avg_speed=%f shot/s\n", shots_per_second);
    return 0;
}
