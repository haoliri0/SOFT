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
           optional_equal(lhs.record_condition, rhs.record_condition) &&
           optional_equal(lhs.exp_val, rhs.exp_val);
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
           optional_equal(lhs.record_condition, rhs.record_condition) &&
           optional_equal(lhs.exp_val, rhs.exp_val);
}

bool operator==(const RecordDetector& lhs, const RecordDetector& rhs) {
    return lhs.outcome == rhs.outcome && lhs.records == rhs.records && lhs.detector == rhs.detector;
}

bool operator==(const MeasurePrecomputedActivePauli& lhs, const MeasurePrecomputedActivePauli& rhs) {
    return lhs.pauli == rhs.pauli && lhs.branch == rhs.branch && lhs.outcome == rhs.outcome &&
           optional_equal(lhs.record, rhs.record) && optional_equal(lhs.record_condition, rhs.record_condition) &&
           optional_equal(lhs.exp_val, rhs.exp_val);
}

bool operator==(const IntroduceDormantMeasurementBranch& lhs, const IntroduceDormantMeasurementBranch& rhs) {
    return lhs.branch == rhs.branch && lhs.outcome == rhs.outcome && optional_equal(lhs.record, rhs.record) &&
           optional_equal(lhs.record_condition, rhs.record_condition) &&
           optional_equal(lhs.exp_val, rhs.exp_val);
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

#define SYMFT_DEFINE_SINGLE_FRAME_GATE(function_name, pullback_gate)       \
    void function_name(FrameFactoredState& state, int q) {                \
        if (state.uses_direct_pullback()) {                               \
            state.direct_pullback.append_clifford(                        \
                detail::LocalCliffordGate::pullback_gate, q);             \
        } else {                                                          \
            function_name(state.clifford, q);                             \
        }                                                                 \
    }

#define SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(function_name, pullback_gate)   \
    void function_name(FrameFactoredState& state, int a, int b) {         \
        if (state.uses_direct_pullback()) {                               \
            state.direct_pullback.append_clifford(                        \
                detail::LocalCliffordGate::pullback_gate, a, b);          \
        } else {                                                          \
            function_name(state.clifford, a, b);                          \
        }                                                                 \
    }

FrameFactoredState::FrameFactoredState(int n_, int k_)
    : FrameFactoredState(
          n_,
          k_,
          std::make_shared<SymbolicContext>(),
          FactorizationStrategy::CliffordFrames) {}

FrameFactoredState::FrameFactoredState(int n_, int k_, std::shared_ptr<SymbolicContext> context_)
    : FrameFactoredState(
          n_,
          k_,
          std::move(context_),
          FactorizationStrategy::CliffordFrames) {}

FrameFactoredState::FrameFactoredState(int n_, int k_, FactorizationStrategy strategy_)
    : FrameFactoredState(n_, k_, std::make_shared<SymbolicContext>(), strategy_) {}

FrameFactoredState::FrameFactoredState(
    int n_,
    int k_,
    std::shared_ptr<SymbolicContext> context_,
    FactorizationStrategy strategy_)
    : n(checked_nqubits(n_)),
      k(checked_nqubits(k_)),
      clifford(strategy_ == FactorizationStrategy::DirectPullback ? 0 : n),
      pauli_frame(strategy_ == FactorizationStrategy::DirectPullback ? 0 : n),
      context(std::move(context_)),
      factorization_strategy(strategy_),
      direct_pullback(strategy_ == FactorizationStrategy::DirectPullback ? n : 0) {
    if (strategy_ == FactorizationStrategy::Automatic) {
        fail("frame-factored state requires a resolved factorization strategy");
    }
    if (k > n) {
        fail("active qubit count exceeds total qubit count");
    }
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
}

bool FrameFactoredState::uses_direct_pullback() const {
    return factorization_strategy == FactorizationStrategy::DirectPullback;
}

SYMFT_DEFINE_SINGLE_FRAME_GATE(left_H, H)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_H_NXY, H_NXY)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_H_NXZ, H_NXZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_H_NYZ, H_NYZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_H_XY, H_XY)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_H_YZ, H_YZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_NXYZ, C_NXYZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_NZYX, C_NZYX)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_XNYZ, C_XNYZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_XYNZ, C_XYNZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_XYZ, C_XYZ)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_ZNYX, C_ZNYX)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_ZYNX, C_ZYNX)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_C_ZYX, C_ZYX)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_S, S)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_SDG, SDG)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_SQRT_X, SqrtX)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_SQRT_X_DAG, SqrtXDag)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_SQRT_Y, SqrtY)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_SQRT_Y_DAG, SqrtYDag)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_X, X)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_Y, Y)
SYMFT_DEFINE_SINGLE_FRAME_GATE(left_Z, Z)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_CX, CX)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_CY, CY)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_CZ, CZ)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SWAP, SWAP)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_CXSWAP, CXSWAP)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_CZSWAP, CZSWAP)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_ISWAP, ISWAP)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_ISWAP_DAG, ISWAP_DAG)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SQRT_XX, SQRT_XX)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SQRT_XX_DAG, SQRT_XX_DAG)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SQRT_YY, SQRT_YY)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SQRT_YY_DAG, SQRT_YY_DAG)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SQRT_ZZ, SQRT_ZZ)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SQRT_ZZ_DAG, SQRT_ZZ_DAG)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_SWAPCX, SWAPCX)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_XCX, XCX)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_XCY, XCY)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_XCZ, XCZ)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_YCX, YCX)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_YCY, YCY)
SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE(left_YCZ, YCZ)

#undef SYMFT_DEFINE_SINGLE_FRAME_GATE
#undef SYMFT_DEFINE_TWO_QUBIT_FRAME_GATE

namespace {

SymbolicPauliString prepare_pending_pauli(FrameFactoredState& state, const PauliString& pauli) {
    if (pauli.nqubits != state.n) {
        fail("Pauli string dimension does not match frame-factored state");
    }
    if (state.uses_direct_pullback()) {
        if (state.deferred_pullback.enabled) {
            return SymbolicPauliString(pauli);
        }
        return state.direct_pullback.pull_back(pauli);
    }
    const PauliString pre = preimage(state.clifford, pauli);
    return conjugate_by(state.pauli_frame, pre);
}

void register_deferred_pending_pauli(FrameFactoredState& state) {
    if (!state.deferred_pullback.enabled) {
        return;
    }
    state.deferred_pullback.entries.push_back({
        state.pending_operations.size() - 1,
        state.direct_pullback.event_count(),
    });
}

SymbolicPauliString& pending_pauli(PendingOperation& operation) {
    return std::visit(
        [](auto& typed) -> SymbolicPauliString& {
            if constexpr (requires { typed.pauli.pauli; }) {
                return typed.pauli;
            } else {
                fail("deferred pullback requires a Pauli operation");
            }
        },
        operation);
}

template <typename Operation>
Operation append_pending_pauli(FrameFactoredState& state, Operation operation) {
    state.pending_operations.emplace_back(operation);
    register_deferred_pending_pauli(state);
    return operation;
}

} // namespace

void begin_deferred_direct_pullback(FrameFactoredState& state) {
    if (!state.uses_direct_pullback()) {
        fail("batched direct pullback requires the direct-pullback strategy");
    }
    if (state.deferred_pullback.enabled) {
        fail("deferred direct pullback is already active");
    }
    state.deferred_pullback.enabled = true;
    state.deferred_pullback.entries.clear();
}

void finish_deferred_direct_pullback(FrameFactoredState& state) {
    if (!state.deferred_pullback.enabled) {
        return;
    }
    // A sparse indexed query avoids scanning unrelated history and wins for
    // small batches. Once enough pending Paulis fill multiple packed words,
    // the single reverse sweep amortizes each history event across the batch.
    const auto& entries = state.deferred_pullback.entries;
    if (entries.size() < detail::kMinimumBatchedPullbackCount) {
        for (const auto& entry : entries) {
            auto& pending = pending_pauli(
                state.pending_operations[entry.pending_index]);
            auto pulled = state.direct_pullback.pull_back(
                pending.pauli,
                entry.event_bound);
            pulled.sign = xor_bool(pulled.sign, pending.sign);
            pending = std::move(pulled);
        }
    } else {
        std::vector<PauliString> physical_paulis;
        std::vector<SymbolicBool> initial_signs;
        std::vector<std::size_t> event_bounds;
        physical_paulis.reserve(entries.size());
        initial_signs.reserve(entries.size());
        event_bounds.reserve(entries.size());
        for (const auto& entry : entries) {
            const auto& pending = pending_pauli(
                state.pending_operations[entry.pending_index]);
            physical_paulis.push_back(pending.pauli);
            initial_signs.push_back(pending.sign);
            event_bounds.push_back(entry.event_bound);
        }
        auto pulled_back = state.direct_pullback.pull_back_batch(
            physical_paulis,
            event_bounds,
            initial_signs);
        for (std::size_t index = 0; index < entries.size(); ++index) {
            pending_pauli(state.pending_operations[entries[index].pending_index]) =
                std::move(pulled_back[index]);
        }
    }
    state.deferred_pullback.enabled = false;
    state.deferred_pullback.entries.clear();
}

void apply_pauli(FrameFactoredState& state, const PauliString& pauli, int condition) {
    if (pauli.nqubits != state.n) {
        fail("Pauli string dimension does not match frame-factored state");
    }
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    state.context->bump_next_condition(condition);
    if (state.uses_direct_pullback()) {
        state.direct_pullback.append_pauli(pauli, condition);
        return;
    }
    const PauliString pre = preimage(state.clifford, pauli);
    if (pre.has_nonidentity_body()) {
        state.pauli_frame.add_pauli(pre, condition);
    }
}

void apply_pauli(FrameFactoredState& state, const PauliString& pauli, const SymbolicBool& condition) {
    if (condition.constant) {
        apply_pauli(state, pauli, state.context->fresh_bernoulli_condition(1.0));
    }
    for (int condition_id : condition.conditions) {
        apply_pauli(state, pauli, condition_id);
    }
}

void apply_single_qubit_pauli(
    FrameFactoredState& state,
    int q,
    bool x,
    bool z,
    int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    if (state.uses_direct_pullback()) {
        state.context->bump_next_condition(condition);
        state.direct_pullback.append_single_qubit_pauli(q, x, z, condition);
        return;
    }
    PauliString pauli(state.n);
    pauli.set_xbit(q, x);
    pauli.set_zbit(q, z);
    if (x && z) {
        pauli.set_phase(1);
    }
    apply_pauli(state, pauli, condition);
}

void apply_single_qubit_pauli(
    FrameFactoredState& state,
    int q,
    bool x,
    bool z,
    const SymbolicBool& condition) {
    if (condition.constant) {
        apply_single_qubit_pauli(
            state,
            q,
            x,
            z,
            state.context->fresh_bernoulli_condition(1.0));
    }
    for (int condition_id : condition.conditions) {
        apply_single_qubit_pauli(state, q, x, z, condition_id);
    }
}

PendingPauliRotation apply_pauli_rotation(FrameFactoredState& state, const PauliString& pauli, double kernel_angle) {
    return append_pending_pauli(
        state,
        PendingPauliRotation{kernel_angle, prepare_pending_pauli(state, pauli)});
}

PendingPauliMeasurement apply_pauli_measurement(FrameFactoredState& state, const PauliString& pauli) {
    return append_pending_pauli(
        state,
        PendingPauliMeasurement{
            prepare_pending_pauli(state, pauli),
            std::nullopt,
            std::nullopt,
            std::nullopt});
}

PendingPauliMeasurement apply_pauli_measurement(
    FrameFactoredState& state,
    const PauliString& pauli,
    const SymbolicBool& sign,
    std::optional<int> record,
    std::optional<int> record_condition) {
    SymbolicPauliString prepared = prepare_pending_pauli(state, pauli);
    return append_pending_pauli(
        state,
        PendingPauliMeasurement{
            SymbolicPauliString(prepared.pauli, xor_bool(prepared.sign, sign)),
            record,
            record_condition,
            std::nullopt,
        });
}

PendingPauliMeasurement apply_pauli_expectation(
    FrameFactoredState& state,
    const PauliString& pauli,
    int exp_val) {
    if (exp_val < 0) {
        fail("expectation value index must be nonnegative");
    }
    return append_pending_pauli(
        state,
        PendingPauliMeasurement{
            prepare_pending_pauli(state, pauli),
            std::nullopt,
            std::nullopt,
            exp_val,
        });
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
      context(std::move(context_)),
      tableau(n) {
    if (k > n) {
        fail("active qubit count exceeds total qubit count");
    }
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
}

PendingFactoredState::PendingFactoredState(const FrameFactoredState& state)
    : n(state.n),
      initial_k(state.k),
      k(state.k),
      max_k(state.k),
      context(state.context),
      tableau(state.n),
      pending_operations(state.pending_operations) {
    for (const auto& op : pending_operations) {
        context->bump_next_condition(max_condition(op));
        if (auto measurement = std::get_if<PendingPauliMeasurement>(&op); measurement && measurement->record) {
            next_record = std::max(next_record, *measurement->record + 1);
        }
    }
}


} // namespace symft
