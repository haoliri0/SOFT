#include "factored/factored_internal.hpp"

#include <type_traits>
#include <utility>

namespace symft {
using namespace detail;

bool operator==(const PendingPauliRotation& lhs, const PendingPauliRotation& rhs) {
    return lhs.kernel_angle == rhs.kernel_angle && lhs.pauli == rhs.pauli;
}

bool operator==(const PendingPauliMeasurement& lhs, const PendingPauliMeasurement& rhs) {
    return lhs.pauli == rhs.pauli && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const PendingClassicalRecord& lhs, const PendingClassicalRecord& rhs) {
    return lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const PendingOperation& lhs, const PendingOperation& rhs) {
    return std::visit(
        [](const auto& a, const auto& b) -> bool {
            using A = std::decay_t<decltype(a)>;
            using B = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<A, B>) {
                return a == b;
            } else {
                return false;
            }
        },
        lhs,
        rhs);
}

bool operator==(const ApplyPrecomputedActivePauliRotation& lhs, const ApplyPrecomputedActivePauliRotation& rhs) {
    return lhs.pauli == rhs.pauli && lhs.kernel_angle == rhs.kernel_angle && lhs.sign == rhs.sign;
}

bool operator==(const PromoteDormantRotation& lhs, const PromoteDormantRotation& rhs) {
    return lhs.kernel_angle == rhs.kernel_angle && lhs.sign == rhs.sign;
}

bool operator==(const RecordMeasurement& lhs, const RecordMeasurement& rhs) {
    return lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const RecordDetector& lhs, const RecordDetector& rhs) {
    return lhs.outcome == rhs.outcome && lhs.records == rhs.records && lhs.detector == rhs.detector;
}

bool operator==(const MeasurePrecomputedActivePauli& lhs, const MeasurePrecomputedActivePauli& rhs) {
    return lhs.pauli == rhs.pauli && lhs.branch == rhs.branch && lhs.outcome == rhs.outcome &&
           optional_equal(lhs.record, rhs.record) && optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const IntroduceDormantMeasurementBranch& lhs, const IntroduceDormantMeasurementBranch& rhs) {
    return lhs.branch == rhs.branch && lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition);
}

bool operator==(const FactoredInstruction& lhs, const FactoredInstruction& rhs) {
    return std::visit(
        [](const auto& a, const auto& b) -> bool {
            using A = std::decay_t<decltype(a)>;
            using B = std::decay_t<decltype(b)>;
            if constexpr (std::is_same_v<A, B>) {
                return a == b;
            } else {
                return false;
            }
        },
        lhs,
        rhs);
}

FrameFactoredState::FrameFactoredState(int n_, int k_)
    : FrameFactoredState(n_, k_, std::make_shared<SymbolicContext>()) {}

FrameFactoredState::FrameFactoredState(int n_, int k_, std::shared_ptr<SymbolicContext> context_)
    : n(checked_nqubits(n_)),
      k(checked_nqubits(k_)),
      clifford(n),
      context(std::move(context_)) {
    if (k > n) {
        fail("active qubit count exceeds total qubit count");
    }
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
    active_frame = ActivePauliFrame(n, context);
    dormant = DormantState(n - k, context);
}

void left_H(FrameFactoredState& state, int q) {
    left_H(state.clifford, q);
}

void left_H_NXY(FrameFactoredState& state, int q) {
    left_H_NXY(state.clifford, q);
}

void left_H_NXZ(FrameFactoredState& state, int q) {
    left_H_NXZ(state.clifford, q);
}

void left_H_NYZ(FrameFactoredState& state, int q) {
    left_H_NYZ(state.clifford, q);
}

void left_H_XY(FrameFactoredState& state, int q) {
    left_H_XY(state.clifford, q);
}

void left_H_YZ(FrameFactoredState& state, int q) {
    left_H_YZ(state.clifford, q);
}

void left_C_NXYZ(FrameFactoredState& state, int q) {
    left_C_NXYZ(state.clifford, q);
}

void left_C_NZYX(FrameFactoredState& state, int q) {
    left_C_NZYX(state.clifford, q);
}

void left_C_XNYZ(FrameFactoredState& state, int q) {
    left_C_XNYZ(state.clifford, q);
}

void left_C_XYNZ(FrameFactoredState& state, int q) {
    left_C_XYNZ(state.clifford, q);
}

void left_C_XYZ(FrameFactoredState& state, int q) {
    left_C_XYZ(state.clifford, q);
}

void left_C_ZNYX(FrameFactoredState& state, int q) {
    left_C_ZNYX(state.clifford, q);
}

void left_C_ZYNX(FrameFactoredState& state, int q) {
    left_C_ZYNX(state.clifford, q);
}

void left_C_ZYX(FrameFactoredState& state, int q) {
    left_C_ZYX(state.clifford, q);
}

void left_S(FrameFactoredState& state, int q) {
    left_S(state.clifford, q);
}

void left_SDG(FrameFactoredState& state, int q) {
    left_SDG(state.clifford, q);
}

void left_SQRT_X(FrameFactoredState& state, int q) {
    left_SQRT_X(state.clifford, q);
}

void left_SQRT_X_DAG(FrameFactoredState& state, int q) {
    left_SQRT_X_DAG(state.clifford, q);
}

void left_SQRT_Y(FrameFactoredState& state, int q) {
    left_SQRT_Y(state.clifford, q);
}

void left_SQRT_Y_DAG(FrameFactoredState& state, int q) {
    left_SQRT_Y_DAG(state.clifford, q);
}

void left_X(FrameFactoredState& state, int q) {
    left_X(state.clifford, q);
}

void left_Y(FrameFactoredState& state, int q) {
    left_Y(state.clifford, q);
}

void left_Z(FrameFactoredState& state, int q) {
    left_Z(state.clifford, q);
}

void left_CX(FrameFactoredState& state, int control, int target) {
    left_CX(state.clifford, control, target);
}

void left_CY(FrameFactoredState& state, int control, int target) {
    left_CY(state.clifford, control, target);
}

void left_CZ(FrameFactoredState& state, int a, int b) {
    left_CZ(state.clifford, a, b);
}

void left_SWAP(FrameFactoredState& state, int a, int b) {
    left_SWAP(state.clifford, a, b);
}

void left_CXSWAP(FrameFactoredState& state, int a, int b) {
    left_CXSWAP(state.clifford, a, b);
}

void left_CZSWAP(FrameFactoredState& state, int a, int b) {
    left_CZSWAP(state.clifford, a, b);
}

void left_ISWAP(FrameFactoredState& state, int a, int b) {
    left_ISWAP(state.clifford, a, b);
}

void left_ISWAP_DAG(FrameFactoredState& state, int a, int b) {
    left_ISWAP_DAG(state.clifford, a, b);
}

void left_SQRT_XX(FrameFactoredState& state, int a, int b) {
    left_SQRT_XX(state.clifford, a, b);
}

void left_SQRT_XX_DAG(FrameFactoredState& state, int a, int b) {
    left_SQRT_XX_DAG(state.clifford, a, b);
}

void left_SQRT_YY(FrameFactoredState& state, int a, int b) {
    left_SQRT_YY(state.clifford, a, b);
}

void left_SQRT_YY_DAG(FrameFactoredState& state, int a, int b) {
    left_SQRT_YY_DAG(state.clifford, a, b);
}

void left_SQRT_ZZ(FrameFactoredState& state, int a, int b) {
    left_SQRT_ZZ(state.clifford, a, b);
}

void left_SQRT_ZZ_DAG(FrameFactoredState& state, int a, int b) {
    left_SQRT_ZZ_DAG(state.clifford, a, b);
}

void left_SWAPCX(FrameFactoredState& state, int a, int b) {
    left_SWAPCX(state.clifford, a, b);
}

void left_XCX(FrameFactoredState& state, int control, int target) {
    left_XCX(state.clifford, control, target);
}

void left_XCY(FrameFactoredState& state, int control, int target) {
    left_XCY(state.clifford, control, target);
}

void left_XCZ(FrameFactoredState& state, int control, int target) {
    left_XCZ(state.clifford, control, target);
}

void left_YCX(FrameFactoredState& state, int control, int target) {
    left_YCX(state.clifford, control, target);
}

void left_YCY(FrameFactoredState& state, int control, int target) {
    left_YCY(state.clifford, control, target);
}

void left_YCZ(FrameFactoredState& state, int control, int target) {
    left_YCZ(state.clifford, control, target);
}

namespace {

SymbolicPauliString prepare_pending_pauli(FrameFactoredState& state, const PauliString& pauli) {
    if (pauli.nqubits != state.n) {
        fail("Pauli string dimension does not match frame-factored state");
    }
    const PauliString pre = preimage(state.clifford, pauli);
    return conjugate_by(state.active_frame, pre);
}

} // namespace

void apply_pauli(FrameFactoredState& state, const ConditionalPauliString& pauli) {
    if (pauli.pauli.nqubits != state.n) {
        fail("Pauli string dimension does not match frame-factored state");
    }
    state.context->bump_next_condition(pauli.condition);
    const PauliString pre = preimage(state.clifford, pauli.pauli);
    if (pre.has_nonidentity_body()) {
        state.active_frame.add_pauli(pre, pauli.condition);
    }
}

void apply_pauli(FrameFactoredState& state, const PauliString& pauli, int condition) {
    apply_pauli(state, ConditionalPauliString(pauli, condition));
}

void apply_pauli(FrameFactoredState& state, const PauliString& pauli, const SymbolicBool& condition) {
    if (condition.constant) {
        apply_pauli(state, pauli, state.context->fresh_bernoulli_condition(1.0));
    }
    for (int condition_id : condition.conditions) {
        apply_pauli(state, pauli, condition_id);
    }
}

PendingPauliRotation apply_pauli_rotation(FrameFactoredState& state, const PauliString& pauli, double kernel_angle) {
    PendingPauliRotation rotation{kernel_angle, prepare_pending_pauli(state, pauli)};
    state.pending_operations.push_back(rotation);
    return rotation;
}

PendingPauliMeasurement apply_pauli_measurement(FrameFactoredState& state, const PauliString& pauli) {
    PendingPauliMeasurement measurement{prepare_pending_pauli(state, pauli), std::nullopt, std::nullopt};
    state.pending_operations.push_back(measurement);
    return measurement;
}

PendingPauliMeasurement apply_pauli_measurement(
    FrameFactoredState& state,
    const PauliString& pauli,
    const SymbolicBool& sign,
    std::optional<int> record,
    std::optional<int> record_condition) {
    SymbolicPauliString prepared = prepare_pending_pauli(state, pauli);
    PendingPauliMeasurement measurement{
        SymbolicPauliString(prepared.pauli, xor_bool(prepared.sign, sign)),
        record,
        record_condition,
    };
    state.pending_operations.push_back(measurement);
    return measurement;
}

PendingClassicalRecord apply_classical_record(
    FrameFactoredState& state,
    const SymbolicBool& outcome,
    std::optional<int> record,
    std::optional<int> record_condition) {
    state.context->bump_next_condition(outcome);
    PendingClassicalRecord classical_record{outcome, record, record_condition};
    state.pending_operations.push_back(classical_record);
    return classical_record;
}

PendingFactoredState::PendingFactoredState(int n_, int k_)
    : PendingFactoredState(n_, k_, std::make_shared<SymbolicContext>()) {}

PendingFactoredState::PendingFactoredState(int n_, int k_, std::shared_ptr<SymbolicContext> context_)
    : n(checked_nqubits(n_)),
      initial_k(checked_nqubits(k_)),
      k(checked_nqubits(k_)),
      max_k(k),
      context(std::move(context_)) {
    if (k > n) {
        fail("active qubit count exceeds total qubit count");
    }
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
    dormant = DormantState(n - k, context);
}

PendingFactoredState::PendingFactoredState(const FrameFactoredState& state)
    : n(state.n),
      initial_k(state.k),
      k(state.k),
      max_k(state.k),
      dormant(state.dormant.bits, state.context),
      context(state.context),
      pending_operations(state.pending_operations) {
    for (const auto& op : pending_operations) {
        context->bump_next_condition(max_condition(op));
        if (auto measurement = std::get_if<PendingPauliMeasurement>(&op); measurement && measurement->record) {
            next_record = std::max(next_record, *measurement->record + 1);
        }
    }
}


} // namespace symft
