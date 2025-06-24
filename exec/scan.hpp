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


struct SimulatorArgs {
    Sid shots_n;
    Qid qubits_n;
    Kid amps_m;
    Rid results_m;
};

static
ScanError scan_simulator_args(FILE *file, SimulatorArgs *args) noexcept {
    constexpr size_t buffer_size = 256;
    char buffer[buffer_size];
    const char *line = fgets(buffer, buffer_size, file);
    if (line == nullptr) return ReadLineFailed;

    const int err = sscanf(line,
        "shots_n=%u, qubits_n=%u, amps_m=%u, results_m=%u",
        &args->shots_n, &args->qubits_n, &args->amps_m, &args->results_m);
    if (err == EOF) return ParseFormatFailed;

    return Success;
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
