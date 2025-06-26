#ifndef STN_CUDA_SCAN_CUH
#define STN_CUDA_SCAN_CUH

#include <cstdio>
#include "../source/datatype.cuh"

using namespace StnCuda;


enum ScanError {
    Success,
    ReadLineFailed,
    ParseFormatFailed
};

static
const char *get_scan_error_name(const ScanError err) {
    switch (err) {
        case Success:
            return "Success";
        case ReadLineFailed:
            return "ReadLineFailed";
        case ParseFormatFailed:
            return "ParseFormatFailed";
        default:
            return "Unknown";
    }
}

static
bool starts_with(const char *str, const char *seg) {
    for (int i = 0;; i++) {
        if (seg[i] == '\0') return true;
        if (str[i] != seg[i]) return false;
    }
    return true;
}


struct CliArgs {
    Sid shots_n = 1;
    Qid qubits_n = 4;
    Kid amps_m = 4;
    Rid results_m = 4;
};

enum class ParseCliArgsError {
    Success = 0,
    IllegalArg = 100001,
    IllegalKey = 100002,
    IllegalValue = 100003,
};

static
ParseCliArgsError parse_cli_args(const int argc, const char **argv, CliArgs &args) {
    for (unsigned int i = 1; i < argc;) {
        const char *arg_key = argv[i++];
        if (starts_with(arg_key, "--shots_n")) {
            const char *arg_value = argv[i++];
            args.shots_n = strtoul(arg_value, nullptr, 10);
            if (args.shots_n == 0) {
                fprintf(stderr, "Illegal value: shots_n=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (starts_with(arg_key, "--qubits_n")) {
            const char *arg_value = argv[i++];
            args.qubits_n = strtoul(arg_value, nullptr, 10);
            if (args.qubits_n == 0) {
                fprintf(stderr, "Illegal value: qubits_n=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (starts_with(arg_key, "--amps_m")) {
            const char *arg_value = argv[i++];
            args.amps_m = strtoul(arg_value, nullptr, 10);
            if (args.amps_m == 0) {
                fprintf(stderr, "Illegal value: amps_m=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (starts_with(arg_key, "--results_m")) {
            const char *arg_value = argv[i++];
            args.results_m = strtoul(arg_value, nullptr, 10);
            if (args.results_m == 0) {
                fprintf(stderr, "Illegal value: results_m=%s\n", arg_value);
                return ParseCliArgsError::IllegalValue;
            }
            continue;
        }
        if (starts_with(arg_key, "-")) {
            fprintf(stderr, "Unrecognized option: %s\n", arg_key);
            return ParseCliArgsError::IllegalKey;
        }
        fprintf(stderr, "Unexpected arg: %s\n", arg_key);
        return ParseCliArgsError::IllegalArg;
    }
    return ParseCliArgsError::Success;
}


enum OpType {
    X,
    Y,
    Z,
    H,
    S,
    SDG,
    T,
    TDG,
    CX,
    M,
    D,
    R
};

struct OperationArgs {
    OpType type;
    Qid target0;
    Qid target1;
};

static
ScanError scan_operation_args(FILE *file, OperationArgs &args) noexcept {
    constexpr size_t buffer_size = 256;
    char buffer[buffer_size];
    const char *line = fgets(buffer, buffer_size, file);
    if (line == nullptr) return ReadLineFailed;

    OpType &type = args.type;
    Qid &target0 = args.target0;
    Qid &target1 = args.target1;

    if (starts_with(line, "X ")) {
        type = X;
        const int err = sscanf(line, "X %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "Y ")) {
        type = Y;
        const int err = sscanf(line, "Y %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "Z ")) {
        type = Z;
        const int err = sscanf(line, "Z %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "H ")) {
        type = H;
        const int err = sscanf(line, "H %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "S ")) {
        type = S;
        const int err = sscanf(line, "S %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "SDG ")) {
        type = SDG;
        const int err = sscanf(line, "SDG %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "T ")) {
        type = T;
        const int err = sscanf(line, "T %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "TDG ")) {
        type = TDG;
        const int err = sscanf(line, "TDG %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "CX ")) {
        type = CX;
        const int err = sscanf(line, "CX %u %u", &target0, &target1);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "M ")) {
        type = M;
        const int err = sscanf(line, "M %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "D ")) {
        type = D;
        const int err = sscanf(line, "D %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    if (starts_with(line, "R ")) {
        type = R;
        const int err = sscanf(line, "R %u", &target0);
        if (err == EOF) return ParseFormatFailed;
        return Success;
    }

    return ParseFormatFailed;
}

#endif
