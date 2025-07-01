#include <ctime>
#include <cstdio>
#include <functional>

#include "../source/datatype.cuh"
#include "../source/simulator.hpp"

using namespace StnCuda;

// common utils

static
const char *match(const char *str, const char *seg) {
    while (true) {
        if (*seg == '\0') return str;
        if (*str != *seg) return nullptr;
        str++;
        seg++;
    }
}

// cli args

struct CliArgs {
    Sid shots_n = 1;
    Qid qubits_n = 4;
    Kid amps_m = 4;
    Rid results_m = 4;
    bool no_output = false;
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
        if (match(arg_key, "--no_output")) {
            args.no_output = true;
            continue;
        }
        if (match(arg_key, "-")) {
            fprintf(stderr, "Unrecognized option: %s\n", arg_key);
            return ParseCliArgsError::IllegalKey;
        }
        fprintf(stderr, "Unexpected arg: %s\n", arg_key);
        return ParseCliArgsError::IllegalArg;
    }
    return ParseCliArgsError::Success;
}

// circuit ops

void read_word(
    FILE *const file,
    size_t const limit,
    char *const word,
    size_t &count,
    bool &eol,
    bool &eof
) noexcept {
    count = 0;
    if (limit == 0) return;

    eol = false;
    eof = false;

    while (true) {
        const int c = fgetc(file);
        if (c == EOF) {
            eof = true;
            return;
        }
        if (c == '\n') {
            eol = true;
            return;
        }
        if (c == ' ') {
            continue;
        }
        word[0] = static_cast<char>(c);
        count = 1;
        break;
    }

    while (count < limit) {
        const int c = fgetc(file);
        if (c == EOF) {
            eof = true;
            break;
        }
        if (c == '\n') {
            eol = true;
            break;
        }
        if (c == ' ') {
            break;
        }
        word[count] = static_cast<char>(c);
        count++;
    }

    if (count < limit)
        word[count] = '\0';
}

void read_uint(
    FILE *const file,
    size_t const limit,
    unsigned int &value,
    size_t &count,
    bool &eol,
    bool &eof
) noexcept {
    count = 0;
    if (limit == 0) return;

    eol = false;
    eof = false;

    while (true) {
        const int c = fgetc(file);
        if (c == EOF) {
            eof = true;
            return;
        }
        if (c == '\n') {
            eol = true;
            return;
        }
        if (!('0' <= c && c <= '9'))
            continue;
        value = c - '0';
        count = 1;
        break;
    }

    while (count < limit) {
        const int c = fgetc(file);
        if (c == EOF) {
            eof = true;
            break;
        }
        if (c == '\n') {
            eol = true;
            break;
        }
        if (!('0' <= c && c <= '9'))
            break;
        value = value * 10 + (c - '0');
        count++;
    }
}

void next_line(
    FILE *const file,
    size_t &count,
    bool &eof
) noexcept {
    for (count = 0;; count++) {
        const int c = fgetc(file);
        if (c == EOF) {
            eof = true;
            return;
        }
        if (c == '\n') {
            return;
        }
    }
}

enum class ParseCircuitLineError {
    Success,
    FullBuffer,
    IllegalFormat,
    IllegalOp,
    IllegalArg
};

ParseCircuitLineError read_qid(
    FILE *file,
    Qid &qid,
    bool &eol,
    bool &eof
) {
    size_t count;
    constexpr size_t limit = 16;
    read_uint(file, limit, qid, count, eol, eof);
    if (count == 0) return ParseCircuitLineError::IllegalArg;
    if (count > limit) return ParseCircuitLineError::FullBuffer;
    return ParseCircuitLineError::Success;
}

ParseCircuitLineError read_bit(
    FILE *file,
    Bit &bit,
    bool &eol,
    bool &eof
) {
    size_t count;
    constexpr size_t limit = 16;
    unsigned int value;
    read_uint(file, limit, value, count, eol, eof);
    if (count == 0) return ParseCircuitLineError::IllegalArg;
    if (count > limit) return ParseCircuitLineError::FullBuffer;
    if (value != 0 && value != 1) return ParseCircuitLineError::IllegalArg;
    bit = value;
    return ParseCircuitLineError::Success;
}

template<void (Simulator::*op)(Qid) const noexcept, bool isMeasure = false>
ParseCircuitLineError execute_op_q(
    const Simulator &simulator,
    FILE *file,
    bool &measure,
    bool &eol,
    bool &eof
) noexcept {
    measure = isMeasure;

    Qid target;
    if (eol || eof) return ParseCircuitLineError::IllegalFormat;
    if (const ParseCircuitLineError err = read_qid(file, target, eol, eof);
        err != ParseCircuitLineError::Success) { return err; }

    (simulator.*op)(target);

    return ParseCircuitLineError::Success;
}

template<void (Simulator::*op)(Qid, Qid) const noexcept, bool isMeasure = false>
ParseCircuitLineError execute_op_qq(
    const Simulator &simulator,
    FILE *file,
    bool &measure,
    bool &eol,
    bool &eof
) noexcept {
    measure = isMeasure;

    Qid target0;
    if (eol || eof) return ParseCircuitLineError::IllegalFormat;
    if (const ParseCircuitLineError err = read_qid(file, target0, eol, eof);
        err != ParseCircuitLineError::Success) { return err; }

    Qid target1;
    if (eol || eof) return ParseCircuitLineError::IllegalFormat;
    if (const ParseCircuitLineError err = read_qid(file, target1, eol, eof);
        err != ParseCircuitLineError::Success) { return err; }

    (simulator.*op)(target0, target1);

    return ParseCircuitLineError::Success;
}

template<void (Simulator::*op)(Qid, Bit) const noexcept, bool isMeasure = false>
ParseCircuitLineError execute_op_qb(
    const Simulator &simulator,
    FILE *file,
    bool &measure,
    bool &eol,
    bool &eof
) noexcept {
    measure = isMeasure;

    Qid target;
    if (eol || eof) return ParseCircuitLineError::IllegalFormat;
    if (const ParseCircuitLineError err = read_qid(file, target, eol, eof);
        err != ParseCircuitLineError::Success) { return err; }

    Bit value;
    if (eol || eof) return ParseCircuitLineError::IllegalFormat;
    if (const ParseCircuitLineError err = read_bit(file, value, eol, eof);
        err != ParseCircuitLineError::Success) { return err; }

    (simulator.*op)(target, value);

    return ParseCircuitLineError::Success;
}

ParseCircuitLineError execute_op_reset(
    const Simulator &simulator,
    FILE *file,
    bool &measure,
    bool &eol,
    bool &eof
) noexcept {
    measure = true;

    Qid target;
    if (eol || eof) return ParseCircuitLineError::IllegalFormat;
    if (const ParseCircuitLineError err = read_qid(file, target, eol, eof);
        err != ParseCircuitLineError::Success) { return err; }

    simulator.assign(target, false);

    return ParseCircuitLineError::Success;
}

ParseCircuitLineError execute_line(
    const Simulator &simulator,
    FILE *file,
    bool &measure,
    bool &eol,
    bool &eof
) noexcept {
    constexpr size_t name_limit = 16;
    char name[name_limit];
    size_t count;
    read_word(file, name_limit, name, count, eol, eof);
    if (count == 0) return ParseCircuitLineError::Success;
    if (count >= name_limit) return ParseCircuitLineError::FullBuffer;

    if (match(name, "X"))
        return execute_op_q<&Simulator::apply_x>(simulator, file, measure, eol, eof);
    if (match(name, "Y"))
        return execute_op_q<&Simulator::apply_y>(simulator, file, measure, eol, eof);
    if (match(name, "Z"))
        return execute_op_q<&Simulator::apply_z>(simulator, file, measure, eol, eof);
    if (match(name, "H"))
        return execute_op_q<&Simulator::apply_h>(simulator, file, measure, eol, eof);
    if (match(name, "S"))
        return execute_op_q<&Simulator::apply_s>(simulator, file, measure, eol, eof);
    if (match(name, "SDG"))
        return execute_op_q<&Simulator::apply_sdg>(simulator, file, measure, eol, eof);
    if (match(name, "T"))
        return execute_op_q<&Simulator::apply_t>(simulator, file, measure, eol, eof);
    if (match(name, "TDG"))
        return execute_op_q<&Simulator::apply_tdg>(simulator, file, measure, eol, eof);
    if (match(name, "CX"))
        return execute_op_qq<&Simulator::apply_cx>(simulator, file, measure, eol, eof);
    if (match(name, "M"))
        return execute_op_q<&Simulator::measure, true>(simulator, file, measure, eol, eof);
    if (match(name, "D"))
        return execute_op_qb<&Simulator::desire, true>(simulator, file, measure, eol, eof);
    if (match(name, "R"))
        return execute_op_reset(simulator, file, measure, eol, eof);
    return ParseCircuitLineError::IllegalOp;
}

cudaError flush_results(
    const CliArgs &args,
    const Simulator &simulator,
    const size_t line_i,
    const size_t lines_n,
    const Rid result_i,
    const Rid results_n
) {
    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const Kid results_m = shots_state_ptr.results_m;
    const Sid shots_n = shots_state_ptr.shots_n;

    cudaError cuda_err = cudaStreamSynchronize(simulator.stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs execute lines %lu:%lu\n", line_i + 1, lines_n + 1);
        fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
        return cuda_err;
    }

    if (args.no_output)
        return cuda_err;

    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

        Kid amps_n;
        cuda_err = cudaMemcpy(&amps_n, shot_state_ptr.get_amps_ptr().get_amps_n_ptr(),
            sizeof(Kid), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess)break;
        const Bit failed = amps_n == 0;

        Bit results_bit[results_m];
        cuda_err = cudaMemcpy(results_bit, shot_state_ptr.get_results_ptr().get_bits_ptr(),
            results_m * sizeof(Bit), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess)break;

        Flt results_prob[results_m];
        cuda_err = cudaMemcpy(results_prob, shot_state_ptr.get_results_ptr().get_probs_ptr(),
            results_m * sizeof(Flt), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) break;

        for (Rid result_j = result_i; result_j < results_n; ++result_j) {
            const Bit bit = results_bit[result_j % results_m];
            const Flt prob = results_prob[result_j % results_m];
            printf("%u,%u,%u,%f\n", shot_i, failed, bit, prob);
        }
    }

    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "Error occurs when reading results %u:%u\n", result_i, results_n);
        fprintf(stderr, "%s\n%s\n", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));
        return cuda_err;
    }

    return cuda_err;
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

    Simulator simulator;
    cudaError cuda_err = cudaSuccess;
    auto line_err = ParseCircuitLineError::Success;
    do {
        fprintf(stderr, "Creating Simulator\n");

        cuda_err = simulator.create(args.shots_n, args.qubits_n, args.amps_m, args.results_m);
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
        size_t lines_n_flushed = 0;
        Rid results_n = 0;
        Rid results_n_flushed = 0;
        while (true) {
            bool eof = false;
            bool eol = false;
            bool measure = false;
            line_err = execute_line(simulator,stdin, measure, eol, eof);
            if (line_err != ParseCircuitLineError::Success) {
                fprintf(stderr, "Error occurs when parsing line %lu.\n", lines_n + 1);
                break;
            }

            if (!eol) {
                size_t count;
                next_line(stdin, count, eof);
            }
            lines_n += 1;

            if (measure) {
                results_n += 1;
                if (results_n - results_n_flushed >= args.results_m) {
                    cuda_err = flush_results(
                        args, simulator,
                        lines_n_flushed, lines_n,
                        results_n_flushed, results_n);
                    if (cuda_err != cudaSuccess) break;
                    lines_n_flushed = lines_n;
                    results_n_flushed = results_n;
                }
            }

            if (eof) break;
        }

        if (cuda_err != cudaSuccess)
            break;

        cuda_err = flush_results(
            args, simulator,
            lines_n_flushed, lines_n,
            results_n_flushed, results_n);

        if (cuda_err != cudaSuccess)
            break;

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
