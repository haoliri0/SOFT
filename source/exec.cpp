#include <span>
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
    Eid entries_m = 16;
    Mid mem_ints_m = 1024;
    Mid mem_flts_m = 1024;
    unsigned long long seed = 42;
};

static
void parse_cli_args(const std::span<const char *> span, CliArgs &args) {
    for (auto iter = ++span.begin(); iter != span.end(); ++iter) {
        const char *arg_key = *iter;
        if (match(arg_key, "--shots_n")) {
            const char *arg_value = *++iter;
            parse_value(arg_value, args.shots_n);
            if (args.shots_n == 0) {
                fprintf(stderr, "Illegal value: shots_n=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--qubits_n")) {
            const char *arg_value = *++iter;
            parse_value(arg_value, args.qubits_n);
            if (args.qubits_n == 0) {
                fprintf(stderr, "Illegal value: qubits_n=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--entries_m")) {
            const char *arg_value = *++iter;
            parse_value(arg_value, args.entries_m);
            if (args.entries_m == 0) {
                fprintf(stderr, "Illegal value: entries_m=%s\n", arg_value);
                throw CliArgsException(CliArgsError::IllegalValue);
            }
            continue;
        }
        if (match(arg_key, "--mem_ints_m")) {
            const char *arg_value = *++iter;
            parse_value(arg_value, args.mem_ints_m);
            continue;
        }
        if (match(arg_key, "--mem_flts_m")) {
            const char *arg_value = *++iter;
            parse_value(arg_value, args.mem_flts_m);
            continue;
        }
        if (match(arg_key, "--seed")) {
            const char *arg_value = *++iter;
            parse_value(arg_value, args.seed);
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

static
void parse_cli_args(const int argc, const char **argv, CliArgs &args) {
    parse_cli_args(std::span(argv, argc), args);
}

// custom ops

void perform_print_state(const Simulator &simulator, const unsigned int print_i) {
    cuda_check(cudaStreamSynchronize(simulator.stream));

    auto const buffer = new char[simulator.shots_state_ptr.get_size_bytes_n()];
    Cleaner buffer_cleaner([buffer] { delete[] buffer; });
    cuda_check(cudaMemcpy(buffer, simulator.shots_state_ptr.ptr,
        simulator.shots_state_ptr.get_size_bytes_n(), cudaMemcpyDeviceToHost));
    ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    shots_state_ptr.ptr = buffer;

    printf("print_%u:\n", print_i);
    for (Sid shot_i = 0; shot_i < shots_state_ptr.shots_n; ++shot_i) {
        ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
        print_indent(1);
        printf("shot %u:\n", shot_i);

        print_indent(2);
        const Err err = *shot_state_ptr.get_work_ptr().get_err_ptr();
        printf("error: %d\n", err);
        if (!err) {
            print_table(shot_state_ptr.get_table_ptr(), 2);
            print_entries(shot_state_ptr.get_entries_ptr(), false, 2);
        }
    }
}

void perform_print_int(const Simulator &simulator, const unsigned int print_i) {
    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const size_t pitch = shots_state_ptr.get_shot_size_bytes_n() + shots_state_ptr.get_shot_pad_bytes_n();
    const void *shot_value_ptr = shots_state_ptr.get_shot_ptr(0).get_work_ptr().get_int_ptr();
    const Sid shots_n = shots_state_ptr.shots_n;

    auto const shots_value = new Int[shots_n];
    Cleaner shots_value_cleaner([shots_value] { delete[] shots_value; });

    cuda_check(cudaMemcpy2DAsync(
        shots_value,
        sizeof(Int),
        shot_value_ptr,
        pitch,
        sizeof(Int),
        shots_n,
        cudaMemcpyDeviceToHost,
        simulator.stream));

    cuda_check(cudaStreamSynchronize(simulator.stream));

    printf("print_%u:\n", print_i);
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        print_indent(1);
        printf("shot_%u: %d\n", shot_i, shots_value[shot_i]);
    }
}

void perform_print_flt(const Simulator &simulator, const unsigned int print_i) {
    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const size_t pitch = shots_state_ptr.get_shot_size_bytes_n() + shots_state_ptr.get_shot_pad_bytes_n();
    const void *shot_value_ptr = shots_state_ptr.get_shot_ptr(0).get_work_ptr().get_flt_ptr();
    const Sid shots_n = shots_state_ptr.shots_n;

    auto const shots_value = new Flt[shots_n];
    Cleaner shots_value_cleaner([shots_value] { delete[] shots_value; });

    cuda_check(cudaMemcpy2DAsync(
        shots_value,
        sizeof(Flt),
        shot_value_ptr,
        pitch,
        sizeof(Flt),
        shots_n,
        cudaMemcpyDeviceToHost,
        simulator.stream));

    cuda_check(cudaStreamSynchronize(simulator.stream));

    printf("print_%u:\n", print_i);
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        print_indent(1);
        printf("shot_%u: %f\n", shot_i, shots_value[shot_i]);
    }
}

void perform_print_err(const Simulator &simulator, const unsigned int print_i) {
    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const size_t pitch = shots_state_ptr.get_shot_size_bytes_n() + shots_state_ptr.get_shot_pad_bytes_n();
    const void *shot_value_ptr = shots_state_ptr.get_shot_ptr(0).get_work_ptr().get_err_ptr();
    const Sid shots_n = shots_state_ptr.shots_n;

    auto const shots_value = new Err[shots_n];
    Cleaner shots_value_cleaner([shots_value] { delete[] shots_value; });

    cuda_check(cudaMemcpy2DAsync(
        shots_value,
        sizeof(Err),
        shot_value_ptr,
        pitch,
        sizeof(Err),
        shots_n,
        cudaMemcpyDeviceToHost,
        simulator.stream));

    cuda_check(cudaStreamSynchronize(simulator.stream));

    printf("print_%u:\n", print_i);
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        print_indent(1);
        printf("shot_%u: %d\n", shot_i, shots_value[shot_i]);
    }
}

// execution

static
void execute_op(
    std::istream &,
    const std::function<void()> &op
) {
    op();
}

template<typename Arg0, typename... Args>
static
void execute_op(
    std::istream &istream,
    const std::function<void (Arg0, Args...)> &op
) {
    Arg0 arg0;
    read_value(istream, arg0);
    const std::function wrapped = [op, arg0](Args... args) { return op(arg0, args...); };
    execute_op(istream, wrapped);
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

template<typename... Args>
static
void execute_op(
    std::istream &istream,
    const Simulator &simulator,
    const unsigned int print_i,
    void (*op)(const Simulator &, unsigned int, Args...)
) {
    std::function wrapped = [&simulator,print_i, op](Args... args) { (*op)(simulator, print_i, args...); };
    return execute_op(istream, wrapped);
}

static
void execute_op(
    std::istream &istream,
    const Simulator &simulator,
    const std::string &name
) {
    if (name == "")
        return execute_op(istream, [] {});
    if (name == "X")
        return execute_op(istream, simulator, &Simulator::apply_x);
    if (name == "Y")
        return execute_op(istream, simulator, &Simulator::apply_y);
    if (name == "Z")
        return execute_op(istream, simulator, &Simulator::apply_z);
    if (name == "H")
        return execute_op(istream, simulator, &Simulator::apply_h);
    if (name == "S")
        return execute_op(istream, simulator, &Simulator::apply_s);
    if (name == "SDG")
        return execute_op(istream, simulator, &Simulator::apply_sdg);
    if (name == "T")
        return execute_op(istream, simulator, &Simulator::apply_t);
    if (name == "TDG")
        return execute_op(istream, simulator, &Simulator::apply_tdg);
    if (name == "CX")
        return execute_op(istream, simulator, &Simulator::apply_cx);

    if (name == "MEASURE")
        return execute_op(istream, simulator, &Simulator::apply_measure);
    if (name == "DESIRE")
        return execute_op(istream, simulator, &Simulator::apply_desire);
    if (name == "RESET")
        return execute_op(istream, simulator, &Simulator::apply_reset);

    if (name == "XERR")
        return execute_op(istream, simulator, &Simulator::apply_noise_x);
    if (name == "ZERR")
        return execute_op(istream, simulator, &Simulator::apply_noise_z);
    if (name == "DEP1")
        return execute_op(istream, simulator, &Simulator::apply_noise_depo1);
    if (name == "DEP2")
        return execute_op(istream, simulator, &Simulator::apply_noise_depo2);

    if (name == "INVERT")
        return execute_op(istream, simulator, &Simulator::apply_classical_invert);
    if (name == "CHECK")
        return execute_op(istream, simulator, &Simulator::apply_classical_check);

    if (name == "OR")
        return execute_op(istream, simulator, &Simulator::apply_classical_or);
    if (name == "XOR")
        return execute_op(istream, simulator, &Simulator::apply_classical_xor);
    if (name == "AND")
        return execute_op(istream, simulator, &Simulator::apply_classical_and);
    if (name == "LUT")
        return execute_op(istream, simulator, &Simulator::apply_classical_lut);

    if (name == "LOAD") {
        std::string object;
        read_value(istream, object);

        if (object == "INT")
            return execute_op(istream, simulator, &Simulator::apply_classical_load_int);
        if (object == "FLT")
            return execute_op(istream, simulator, &Simulator::apply_classical_load_flt);

        fprintf(stderr, "Unknown load object: %s\n", object.c_str());
        throw ExecException(ExecError::IllegalOp);
    }

    if (name == "SAVE") {
        std::string object;
        read_value(istream, object);

        if (object == "INT")
            return execute_op(istream, simulator, &Simulator::apply_classical_save_int);
        if (object == "FLT")
            return execute_op(istream, simulator, &Simulator::apply_classical_save_flt);

        fprintf(stderr, "Unknown save object: %s\n", object.c_str());
        throw ExecException(ExecError::IllegalOp);
    }

    if (name == "PRINT") {
        std::string object;
        read_value(istream, object);

        if (object == "STATE")
            return execute_op(istream, simulator, perform_print_state);
        if (object == "INT")
            return execute_op(istream, simulator, perform_print_int);
        if (object == "FLT")
            return execute_op(istream, simulator, perform_print_flt);
        if (object == "ERR")
            return execute_op(istream, simulator, perform_print_err);

        fprintf(stderr, "Unknown print object: %s\n", object.c_str());
        throw ExecException(ExecError::IllegalOp);
    }

    fprintf(stderr, "Unknown op: %s\n", name.c_str());
    throw ExecException(ExecError::IllegalOp);
}

static
void execute_line(
    const Simulator &simulator,
    std::istream &istream
) {
    std::string name;
    read_value(istream, name);
    execute_op(istream, simulator, name);

    skip(istream, is_whitespace);
    ensure(istream, is_linebreak, true);
}

void execute_lines(
    const Simulator &simulator,
    std::istream &istream
) {
    size_t lines_n = 0;
    istream >> std::noskipws;
    while (istream.good()) {
        try {
            execute_line(simulator, istream);
        } catch (...) {
            fprintf(stderr, "Error occurs when parsing line %lu.\n", lines_n + 1);
            throw;
        }
        lines_n += 1;
    }
}

// main

int main(const int argc, const char **argv) {
    CliArgs args{};
    parse_cli_args(argc, argv, args);
    fprintf(stderr, "args:\n");
    fprintf(stderr, "  shots_n=%u\n", args.shots_n);
    fprintf(stderr, "  qubits_n=%u\n", args.qubits_n);
    fprintf(stderr, "  entries_m=%u\n", args.entries_m);
    fprintf(stderr, "  mem_ints_m=%u\n", args.mem_ints_m);
    fprintf(stderr, "  mem_flts_m=%u\n", args.mem_flts_m);
    fprintf(stderr, "  seed=%llu\n", args.seed);
    fprintf(stderr, "\n");

    Simulator simulator;
    Cleaner simulator_cleaner([&simulator] { simulator.destroy(); });

    fprintf(stderr, "Creating\n");
    cudaError cuda_err = cudaSuccess;

    cuda_err = simulator.create(
        args.shots_n, args.qubits_n, args.entries_m,
        args.mem_ints_m, args.mem_flts_m, args.seed);
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

    execute_lines(simulator, std::cin);

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
