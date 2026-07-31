#pragma once

#include "core/internal.hpp"
#include "factored/factored.hpp"

#include <algorithm>
#include <optional>
#include <variant>

namespace symft::detail {

inline PauliString project_pauli_body(const PauliString& pauli, int qstart, int qcount) {
    PauliString out(qcount);
    int y_count = 0;
    for (int q = 0; q < qcount; ++q) {
        const int src = qstart + q;
        const bool xb = pauli.xbit(src);
        const bool zb = pauli.zbit(src);
        if (xb) {
            out.set_xbit(q);
        }
        if (zb) {
            out.set_zbit(q);
        }
        if (xb && zb) {
            ++y_count;
        }
    }
    out.set_phase(y_count);
    return out;
}

inline PauliString embed_active_pauli(int n, const PauliString& active_body) {
    PauliString out(n);
    for (int q = 0; q < active_body.nqubits; ++q) {
        if (active_body.xbit(q)) {
            out.set_xbit(q);
        }
        if (active_body.zbit(q)) {
            out.set_zbit(q);
        }
    }
    out.set_phase(active_body.phase_exponent());
    return out;
}

inline int max_condition(const SymbolicPauliString& pauli) {
    return pauli.sign.max_condition();
}

inline int max_condition(const PendingPauliRotation& rotation) {
    return max_condition(rotation.pauli);
}

inline int max_condition(const PendingPauliMeasurement& measurement) {
    int out = max_condition(measurement.pauli);
    if (measurement.record_condition) {
        out = std::max(out, *measurement.record_condition);
    }
    return out;
}

inline int max_condition(const PendingClassicalRecord& record) {
    int out = record.outcome.max_condition();
    if (record.record_condition) {
        out = std::max(out, *record.record_condition);
    }
    return out;
}

template <class... Ts>
struct Overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

inline int max_condition(const PendingOperation& operation) {
    return std::visit(Overloaded{
                          [](const PendingPauliRotation& op) { return max_condition(op); },
                          [](const PendingPauliMeasurement& op) { return max_condition(op); },
                          [](const PendingClassicalRecord& op) { return max_condition(op); },
                      },
                      operation);
}

inline int max_condition(const ApplyPrecomputedActivePauliRotation& instruction) {
    return instruction.sign.max_condition();
}

inline int max_condition(const PromoteDormantRotation& instruction) {
    return instruction.sign.max_condition();
}

inline int max_condition(const RecordMeasurement& instruction) {
    int out = instruction.outcome.max_condition();
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const RecordDetector& instruction) {
    return instruction.outcome.max_condition();
}

inline int max_condition(const MeasurePrecomputedActivePauli& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const IntroduceDormantMeasurementBranch& instruction) {
    int out = std::max(instruction.branch, instruction.outcome.max_condition());
    if (instruction.record_condition) {
        out = std::max(out, *instruction.record_condition);
    }
    return out;
}

inline int max_condition(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return max_condition(inst); }, instruction);
}

} // namespace symft::detail
