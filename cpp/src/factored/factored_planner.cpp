#include "factored/factored_internal.hpp"
#include "sampler/component_plan.hpp"

#include <algorithm>
#include <optional>
#include <type_traits>
#include <utility>

namespace symft {
using namespace detail;

bool has_pending_operations(const PendingFactoredState& state) {
    return !state.pending_operations.empty();
}

namespace {

void set_planning_active_count(PendingFactoredState& state, int k) {
    const int ki = checked_nqubits(k);
    if (ki > state.n) {
        fail("active qubit count exceeds total qubit count");
    }
    state.k = ki;
    state.max_k = std::max(state.max_k, state.k);
    state.dormant = DormantState(state.n - state.k, state.context);
}

FactoredInstruction push_instruction(PendingFactoredState& state, FactoredInstruction instruction) {
    state.context->bump_next_condition(max_condition(instruction));
    state.instructions.push_back(std::move(instruction));
    return state.instructions.back();
}

std::optional<int> measurement_record(PendingFactoredState& state, const PendingPauliMeasurement& measurement) {
    if (!measurement.record) {
        if (measurement.record_condition) {
            return std::nullopt;
        }
        return state.next_record++;
    }
    state.next_record = std::max(state.next_record, *measurement.record + 1);
    return measurement.record;
}

std::optional<int> measurement_record(PendingFactoredState& state, const PendingClassicalRecord& record) {
    if (!record.record) {
        if (record.record_condition) {
            return std::nullopt;
        }
        return state.next_record++;
    }
    state.next_record = std::max(state.next_record, *record.record + 1);
    return record.record;
}

PauliString transform_by_frame(const CliffordFrame& frame, const PauliString& pauli);
PendingPauliRotation transform_operation_by_frame(const PendingPauliRotation& operation, const CliffordFrame& frame);
PendingPauliMeasurement transform_operation_by_frame(const PendingPauliMeasurement& operation, const CliffordFrame& frame);
PendingClassicalRecord transform_operation_by_frame(const PendingClassicalRecord& operation, const CliffordFrame& frame);
void transform_pending_operations_by_frame(PendingFactoredState& state, const CliffordFrame& frame);

PendingPauliRotation xor_operation_sign_if_anticommutes(
    const PendingPauliRotation& operation,
    const PauliString& pauli,
    const SymbolicBool& sign) {
    if (!pauli_anticommutes(pauli, operation.pauli.pauli)) {
        return operation;
    }
    return PendingPauliRotation{operation.kernel_angle, SymbolicPauliString(operation.pauli.pauli, xor_bool(operation.pauli.sign, sign))};
}

PendingPauliMeasurement xor_operation_sign_if_anticommutes(
    const PendingPauliMeasurement& operation,
    const PauliString& pauli,
    const SymbolicBool& sign) {
    if (!pauli_anticommutes(pauli, operation.pauli.pauli)) {
        return operation;
    }
    return PendingPauliMeasurement{
        SymbolicPauliString(operation.pauli.pauli, xor_bool(operation.pauli.sign, sign)),
        operation.record,
        operation.record_condition,
    };
}

PendingClassicalRecord xor_operation_sign_if_anticommutes(
    const PendingClassicalRecord& operation,
    const PauliString&,
    const SymbolicBool&) {
    return operation;
}

std::size_t symbolic_word_cost(const SymbolicBool& expr) {
    std::size_t count = 0;
    std::size_t previous = 0;
    for (int condition : expr.conditions) {
        const std::size_t word = symbol_word_index(condition);
        if (count == 0 || word != previous) {
            ++count;
            previous = word;
        }
    }
    return count;
}

bool has_lower_sampling_cost(const SymbolicBool& candidate, const SymbolicBool& current) {
    const std::size_t candidate_words = symbolic_word_cost(candidate);
    const std::size_t current_words = symbolic_word_cost(current);
    if (candidate_words != current_words) {
        return candidate_words < current_words;
    }
    return candidate.conditions.size() < current.conditions.size();
}

bool conditions_overlap(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < lhs.size() && j < rhs.size()) {
        if (lhs[i] == rhs[j]) {
            return true;
        }
        if (lhs[i] < rhs[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return false;
}

SymbolicBool reduce_by_relation_once(const SymbolicBool& expr, const SymbolicBool& relation) {
    if (relation.conditions.empty() && !relation.constant) {
        return expr;
    }
    if (!conditions_overlap(expr.conditions, relation.conditions)) {
        return expr;
    }
    const SymbolicBool candidate = xor_bool(expr, relation);
    return has_lower_sampling_cost(candidate, expr) ? candidate : expr;
}

SymbolicBool reduce_by_relations(SymbolicBool expr, const std::vector<SymbolicBool>& relations) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& relation : relations) {
            const SymbolicBool reduced = reduce_by_relation_once(expr, relation);
            if (reduced != expr) {
                expr = reduced;
                changed = true;
            }
        }
    }
    return expr;
}

SymbolicBool measurement_relation(int record_condition, const SymbolicBool& outcome) {
    return xor_bool(symbolic_bool(record_condition), outcome);
}

template <typename Operation>
Operation reduce_pending_operation_sign(const Operation& operation, const SymbolicBool& relation) {
    Operation out = operation;
    out.pauli.sign = reduce_by_relation_once(out.pauli.sign, relation);
    return out;
}

PendingClassicalRecord reduce_pending_operation_sign(
    const PendingClassicalRecord& operation,
    const SymbolicBool& relation) {
    PendingClassicalRecord out = operation;
    out.outcome = reduce_by_relation_once(out.outcome, relation);
    return out;
}

void push_symbolic_pauli_through_pending_from(
    PendingFactoredState& state,
    int first_index_one_based,
    const PauliString& pauli,
    const SymbolicBool& sign) {
    state.context->bump_next_condition(sign);
    const std::size_t start = first_index_one_based <= 1 ? 0 : static_cast<std::size_t>(first_index_one_based - 1);
    for (std::size_t idx = start; idx < state.pending_operations.size(); ++idx) {
        state.pending_operations[idx] = std::visit(
            [&](const auto& op) -> PendingOperation { return xor_operation_sign_if_anticommutes(op, pauli, sign); },
            state.pending_operations[idx]);
    }
}

void reduce_pending_signs_by_measurement_relation(
    PendingFactoredState& state,
    int first_index_one_based,
    std::optional<int> record_condition,
    const SymbolicBool& outcome) {
    if (!record_condition) {
        return;
    }
    const SymbolicBool relation = measurement_relation(*record_condition, outcome);
    state.context->bump_next_condition(relation);
    const std::size_t start = first_index_one_based <= 1 ? 0 : static_cast<std::size_t>(first_index_one_based - 1);
    for (std::size_t idx = start; idx < state.pending_operations.size(); ++idx) {
        state.pending_operations[idx] = std::visit(
            [&](const auto& op) -> PendingOperation {
                return reduce_pending_operation_sign(op, relation);
            },
            state.pending_operations[idx]);
    }
}

std::optional<int> highest_dormant_x_qubit(const PendingFactoredState& state, const PauliString& pauli) {
    for (int q = state.n; q-- > state.k;) {
        if (pauli.xbit(q)) {
            return q - state.k;
        }
    }
    return std::nullopt;
}

SymbolicBool rotation_sign_from_pauli(const SymbolicPauliString& pauli) {
    return xor_bool(pauli.sign, measurement_phase_sign(pauli.pauli));
}

PauliString positive_hermitian_body(PauliString pauli) {
    if (!pauli_squares_to_identity(pauli)) {
        fail("Pauli frame row requires a Hermitian Pauli body");
    }
    pauli.set_phase(pauli_body_y_count(pauli));
    return pauli;
}

PauliString multiply_by_stabilizer_if_anticommutes(
    PauliString row,
    const PauliString& measured_or_rotated,
    const PauliString& stabilizer) {
    if (pauli_anticommutes(row, measured_or_rotated)) {
        row = row * stabilizer;
        row.set_phase(pauli_body_y_count(row));
    }
    return row;
}

// Planner-only stabilizer tableau basis changes. The runtime alpha vector is
// changed only by the emitted promote/measure instructions, never by these frames.
CliffordFrame dormant_rotation_promotion_tableau_frame(
    const PendingFactoredState& state,
    const PauliString& rotation_pauli,
    int picked_dormant) {
    const int old_k = state.k;
    const int picked_q = old_k + picked_dormant;
    const PauliString stabilizer = pauli_z(state.n, picked_q);
    const PauliString promoted_x = positive_hermitian_body(rotation_pauli);
    if (!pauli_anticommutes(promoted_x, stabilizer)) {
        fail("dormant rotation promotion requires an anti-commuting fixed stabilizer");
    }

    CliffordFrame frame(state.n);
    for (int q = 0; q < old_k; ++q) {
        frame.copy_pauli_to_row(
            frame.zrow(q),
            multiply_by_stabilizer_if_anticommutes(pauli_z(state.n, q), promoted_x, stabilizer));
        frame.copy_pauli_to_row(
            frame.xrow(q),
            multiply_by_stabilizer_if_anticommutes(pauli_x(state.n, q), promoted_x, stabilizer));
    }

    frame.copy_pauli_to_row(frame.zrow(old_k), stabilizer);
    frame.copy_pauli_to_row(frame.xrow(old_k), promoted_x);

    int new_q = old_k + 1;
    for (int old_q = old_k; old_q < state.n; ++old_q) {
        if (old_q == picked_q) {
            continue;
        }
        frame.copy_pauli_to_row(
            frame.zrow(new_q),
            multiply_by_stabilizer_if_anticommutes(pauli_z(state.n, old_q), promoted_x, stabilizer));
        frame.copy_pauli_to_row(
            frame.xrow(new_q),
            multiply_by_stabilizer_if_anticommutes(pauli_x(state.n, old_q), promoted_x, stabilizer));
        ++new_q;
    }
    if (new_q != state.n) {
        fail("dormant rotation promotion frame did not repack dormant rows");
    }
    return frame;
}

ApplyPrecomputedActivePauliRotation active_rotation_instruction(
    const PauliString& active_body,
    double kernel_angle,
    const SymbolicBool& sign) {
    ActivePauliAction action(active_body);
    return ApplyPrecomputedActivePauliRotation{
        active_body,
        action,
        PrecomputedActivePauliRotationKernel(action, kernel_angle),
        kernel_angle,
        sign,
        SymbolicBoolEvaluationPlan(sign),
    };
}

std::optional<FactoredInstruction> process_diagonal_dormant_rotation(
    PendingFactoredState& state,
    const PendingPauliRotation& current) {
    const PauliString active_body = project_pauli_body(current.pauli.pauli, 0, state.k);
    const SymbolicBool sign = rotation_sign_from_pauli(current.pauli);
    if (!active_body.has_nonidentity_body()) {
        return std::nullopt;
    }
    if (!pauli_squares_to_identity(active_body)) {
        fail("active rotation Pauli must square to identity");
    }
    return push_instruction(state, active_rotation_instruction(active_body, current.kernel_angle, sign));
}

std::optional<FactoredInstruction> process_nondiagonal_dormant_rotation(
    PendingFactoredState& state,
    PendingPauliRotation current,
    int picked_dormant) {
    const int old_k = state.k;
    const CliffordFrame frame = dormant_rotation_promotion_tableau_frame(state, current.pauli.pauli, picked_dormant);
    current = transform_operation_by_frame(current, frame);
    transform_pending_operations_by_frame(state, frame);
    if (!current.pauli.pauli.same_body(pauli_x(state.n, old_k))) {
        fail("dormant rotation tableau reduction did not expose promoted X");
    }
    const SymbolicBool sign = rotation_sign_from_pauli(current.pauli);
    PromoteDormantRotation instruction{current.kernel_angle, sign, SymbolicBoolEvaluationPlan(sign)};
    FactoredInstruction pushed = push_instruction(state, std::move(instruction));
    set_planning_active_count(state, old_k + 1);
    return pushed;
}

SymbolicBool measurement_base_outcome(const SymbolicPauliString& pauli) {
    return xor_bool(pauli.sign, measurement_phase_sign(pauli.pauli));
}

std::optional<FactoredInstruction> record_deterministic_measurement(
    PendingFactoredState& state,
    const PendingPauliMeasurement& measurement,
    const SymbolicBool& outcome,
    bool queued_first) {
    const auto instruction = push_instruction(
        state,
        RecordMeasurement{
            outcome,
            measurement_record(state, measurement),
            measurement.record_condition,
            SymbolicBoolEvaluationPlan(outcome),
        });
    reduce_pending_signs_by_measurement_relation(
        state,
        queued_first ? 2 : 1,
        measurement.record_condition,
        outcome);
    return instruction;
}

CliffordFrame dormant_measurement_replacement_tableau_frame(
    const PendingFactoredState& state,
    const PauliString& measured_pauli,
    int picked_dormant) {
    const int picked_q = state.k + picked_dormant;
    const PauliString old_stabilizer = pauli_z(state.n, picked_q);
    const PauliString new_stabilizer = positive_hermitian_body(measured_pauli);
    if (!pauli_anticommutes(new_stabilizer, old_stabilizer)) {
        fail("dormant measurement replacement requires an anti-commuting fixed stabilizer");
    }

    CliffordFrame frame(state.n);
    for (int q = 0; q < state.n; ++q) {
        if (q == picked_q) {
            continue;
        }
        frame.copy_pauli_to_row(
            frame.zrow(q),
            multiply_by_stabilizer_if_anticommutes(pauli_z(state.n, q), new_stabilizer, old_stabilizer));
        frame.copy_pauli_to_row(
            frame.xrow(q),
            multiply_by_stabilizer_if_anticommutes(pauli_x(state.n, q), new_stabilizer, old_stabilizer));
    }
    frame.copy_pauli_to_row(frame.zrow(picked_q), new_stabilizer);
    frame.copy_pauli_to_row(frame.xrow(picked_q), old_stabilizer);
    return frame;
}

std::optional<FactoredInstruction> measure_dormant_xy_pauli(
    PendingFactoredState& state,
    PendingPauliMeasurement current,
    int picked_dormant,
    bool queued_first) {
    const int picked_q = state.k + picked_dormant;
    const CliffordFrame frame = dormant_measurement_replacement_tableau_frame(state, current.pauli.pauli, picked_dormant);
    current = transform_operation_by_frame(current, frame);
    transform_pending_operations_by_frame(state, frame);
    if (!current.pauli.pauli.same_body(pauli_z(state.n, picked_q))) {
        fail("dormant measurement tableau reduction did not expose fixed Z");
    }
    const SymbolicBool base_outcome = measurement_base_outcome(current.pauli);
    const int branch = state.context->fresh_condition();
    const SymbolicBool branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending_from(
        state,
        queued_first ? 2 : 1,
        pauli_x(state.n, state.k + picked_dormant),
        branch_bit);
    const SymbolicBool outcome = xor_bool(base_outcome, branch_bit);
    const auto instruction = push_instruction(
        state,
        IntroduceDormantMeasurementBranch{
            branch,
            outcome,
            measurement_record(state, current),
            current.record_condition,
            SymbolicBoolEvaluationPlan(outcome),
        });
    reduce_pending_signs_by_measurement_relation(
        state,
        queued_first ? 2 : 1,
        current.record_condition,
        outcome);
    return instruction;
}

PauliString transform_by_frame(const CliffordFrame& frame, const PauliString& pauli) {
    return coordinates_in_frame(frame, pauli);
}

PendingPauliRotation transform_operation_by_frame(const PendingPauliRotation& operation, const CliffordFrame& frame) {
    return PendingPauliRotation{
        operation.kernel_angle,
        SymbolicPauliString(transform_by_frame(frame, operation.pauli.pauli), operation.pauli.sign),
    };
}

PendingPauliMeasurement transform_operation_by_frame(const PendingPauliMeasurement& operation, const CliffordFrame& frame) {
    return PendingPauliMeasurement{
        SymbolicPauliString(transform_by_frame(frame, operation.pauli.pauli), operation.pauli.sign),
        operation.record,
        operation.record_condition,
    };
}

PendingClassicalRecord transform_operation_by_frame(const PendingClassicalRecord& operation, const CliffordFrame&) {
    return operation;
}

void transform_pending_operations_by_frame(PendingFactoredState& state, const CliffordFrame& frame) {
    for (auto& op : state.pending_operations) {
        op = std::visit([&](const auto& typed) -> PendingOperation { return transform_operation_by_frame(typed, frame); }, op);
    }
}

CliffordFrame active_measurement_coordinate_frame(
    const PendingFactoredState& state,
    const PauliString& active_body,
    const PrecomputedActivePauliMeasurementKernel& kernel) {
    CliffordFrame frame(state.n);
    const int k = state.k;
    const int pivot = kernel.pivot;
    const PauliString measured = embed_active_pauli(state.n, active_body);
    const PauliString fixed_x = kernel.is_diagonal ? pauli_x(state.n, pivot) : pauli_z(state.n, pivot);
    frame.copy_pauli_to_row(frame.xrow(k - 1), fixed_x);
    frame.copy_pauli_to_row(frame.zrow(k - 1), measured);
    int new_q = 0;
    for (int old_q = 0; old_q < k; ++old_q) {
        if (old_q == pivot) {
            continue;
        }
        PauliString zrow = pauli_z(state.n, old_q);
        PauliString xrow = pauli_x(state.n, old_q);
        if (kernel.is_diagonal) {
            if (active_body.zbit(old_q)) {
                xrow = xrow * pauli_x(state.n, pivot);
            }
        } else {
            if (active_body.xbit(old_q)) {
                zrow = zrow * pauli_z(state.n, pivot);
            }
            if (active_body.zbit(old_q)) {
                xrow = xrow * pauli_z(state.n, pivot);
            }
        }
        frame.copy_pauli_to_row(frame.xrow(new_q), xrow);
        frame.copy_pauli_to_row(frame.zrow(new_q), zrow);
        ++new_q;
    }
    if (new_q != k - 1) {
        fail("active measurement tableau dropped the wrong number of qubits");
    }
    return frame;
}

std::optional<FactoredInstruction> measure_active_pauli_branches(
    PendingFactoredState& state,
    const PendingPauliMeasurement& current,
    const PauliString& active_body,
    const SymbolicBool& base_outcome,
    bool queued_first) {
    if (!pauli_squares_to_identity(active_body)) {
        fail("active measurement Pauli must square to identity");
    }
    PrecomputedActivePauliMeasurementKernel kernel(active_body);
    const CliffordFrame frame = active_measurement_coordinate_frame(state, active_body, kernel);
    transform_pending_operations_by_frame(state, frame);
    const int branch = state.context->fresh_condition();
    const SymbolicBool branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending_from(
        state,
        queued_first ? 2 : 1,
        pauli_x(state.n, state.k - 1),
        branch_bit);
    const SymbolicBool outcome = xor_bool(base_outcome, branch_bit);
    MeasurePrecomputedActivePauli instruction{
        active_body,
        std::move(kernel),
        branch,
        outcome,
        measurement_record(state, current),
        current.record_condition,
        SymbolicBoolEvaluationPlan(outcome),
    };
    FactoredInstruction pushed = push_instruction(state, std::move(instruction));
    reduce_pending_signs_by_measurement_relation(
        state,
        queued_first ? 2 : 1,
        current.record_condition,
        outcome);
    set_planning_active_count(state, state.k - 1);
    return pushed;
}

} // namespace

std::optional<FactoredInstruction> process_pending_rotation(PendingFactoredState& state, const PendingPauliRotation& rotation) {
    state.context->bump_next_condition(max_condition(rotation));
    PendingPauliRotation current = rotation;
    const auto picked = highest_dormant_x_qubit(state, current.pauli.pauli);
    if (!picked) {
        return process_diagonal_dormant_rotation(state, current);
    }
    return process_nondiagonal_dormant_rotation(state, current, *picked);
}

std::optional<FactoredInstruction> process_pending_measurement(PendingFactoredState& state, const PendingPauliMeasurement& measurement) {
    state.context->bump_next_condition(max_condition(measurement));
    const bool queued_first = !state.pending_operations.empty() && state.pending_operations.front() == PendingOperation(measurement);
    PendingPauliMeasurement current = measurement;
    const PauliString active_body = project_pauli_body(current.pauli.pauli, 0, state.k);
    const auto picked = highest_dormant_x_qubit(state, current.pauli.pauli);
    if (picked) {
        return measure_dormant_xy_pauli(state, current, *picked, queued_first);
    }
    const SymbolicBool base_outcome = measurement_base_outcome(current.pauli);
    if (!active_body.has_nonidentity_body()) {
        return record_deterministic_measurement(state, current, base_outcome, queued_first);
    }
    return measure_active_pauli_branches(state, current, active_body, base_outcome, queued_first);
}

std::optional<FactoredInstruction> process_pending_classical_record(
    PendingFactoredState& state,
    const PendingClassicalRecord& record) {
    state.context->bump_next_condition(max_condition(record));
    return push_instruction(
        state,
        RecordMeasurement{
            record.outcome,
            measurement_record(state, record),
            record.record_condition,
            SymbolicBoolEvaluationPlan(record.outcome),
        });
}

std::optional<FactoredInstruction> process_next_pending_operation(PendingFactoredState& state) {
    if (state.pending_operations.empty()) {
        return std::nullopt;
    }
    if (state.pending_prefix_instruction_indices.empty()) {
        state.pending_prefix_instruction_indices.push_back(static_cast<int>(state.instructions.size()));
    }
    const PendingOperation operation = state.pending_operations.front();
    const std::size_t start = state.instructions.size();
    std::optional<FactoredInstruction> result = std::visit(
        [&](const auto& op) -> std::optional<FactoredInstruction> {
            if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PendingPauliRotation>) {
                return process_pending_rotation(state, op);
            } else if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PendingPauliMeasurement>) {
                return process_pending_measurement(state, op);
            } else {
                return process_pending_classical_record(state, op);
            }
        },
        operation);
    state.pending_operations.erase(state.pending_operations.begin());
    state.pending_prefix_instruction_indices.push_back(static_cast<int>(state.instructions.size()));
    if (state.instructions.size() == start) {
        return result;
    }
    return state.instructions.back();
}

namespace {

void process_pending_operations_in_place(PendingFactoredState& state) {
    if (!state.pending_operations_optimized && state.instructions.empty() &&
        state.pending_prefix_instruction_indices.empty()) {
        optimize_pending_operations(state);
    }
    while (has_pending_operations(state)) {
        process_next_pending_operation(state);
    }
}

} // namespace

std::vector<FactoredInstruction> process_pending_operations(PendingFactoredState& state) {
    const std::size_t start = state.instructions.size();
    process_pending_operations_in_place(state);
    return std::vector<FactoredInstruction>(
        state.instructions.begin() + static_cast<std::ptrdiff_t>(start),
        state.instructions.end());
}

namespace {

template <typename T>
void reduce_instruction_symbolic_expressions(T& instruction, const std::vector<SymbolicBool>& relations) {
    if constexpr (requires { instruction.sign; }) {
        instruction.sign = reduce_by_relations(std::move(instruction.sign), relations);
    }
    if constexpr (requires { instruction.outcome; }) {
        instruction.outcome = reduce_by_relations(std::move(instruction.outcome), relations);
    }
}

template <typename T>
std::optional<SymbolicBool> measurement_relation_from_instruction(const T& instruction) {
    if constexpr (requires { instruction.record_condition; instruction.outcome; }) {
        if (instruction.record_condition) {
            return measurement_relation(*instruction.record_condition, instruction.outcome);
        }
    }
    return std::nullopt;
}

void reduce_program_symbolic_expressions(std::vector<FactoredInstruction>& instructions) {
    std::vector<SymbolicBool> relations;
    for (auto& instruction : instructions) {
        std::visit(
            [&](auto& inst) {
                reduce_instruction_symbolic_expressions(inst, relations);
            },
            instruction);
        const auto relation = std::visit(
            [](const auto& inst) -> std::optional<SymbolicBool> {
                return measurement_relation_from_instruction(inst);
            },
            instruction);
        if (relation && (!relation->conditions.empty() || relation->constant)) {
            relations.push_back(*relation);
        }
    }
}

void refresh_instruction_plans(ApplyPrecomputedActivePauliRotation& instruction) {
    instruction.sign_plan = SymbolicBoolEvaluationPlan(instruction.sign);
}

void refresh_instruction_plans(PromoteDormantRotation& instruction) {
    instruction.sign_plan = SymbolicBoolEvaluationPlan(instruction.sign);
}

void refresh_instruction_plans(RecordMeasurement& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(RecordDetector& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(MeasurePrecomputedActivePauli& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(IntroduceDormantMeasurementBranch& instruction) {
    instruction.outcome_plan = SymbolicBoolEvaluationPlan(instruction.outcome);
}

void refresh_instruction_plans(FactoredInstruction& instruction) {
    std::visit([](auto& inst) { refresh_instruction_plans(inst); }, instruction);
}

struct RareInfo {
    double event_probability = 0.0;
    std::vector<int> event_rows;
    std::vector<double> event_probabilities;
};

std::optional<RareInfo> rare_categorical_sample_info(const SymbolicCategoricalDistribution& distribution) {
    std::optional<std::size_t> false_row;
    for (std::size_t row = 0; row < distribution.assignments.size(); ++row) {
        bool any_true = false;
        for (int bit = 0; bit < distribution.nbits; ++bit) {
            any_true = any_true || packed_bit(distribution.assignments[row], bit);
        }
        if (!any_true) {
            false_row = row;
            break;
        }
    }
    if (!false_row) {
        return std::nullopt;
    }
    const double event_probability = 1.0 - distribution.probabilities[*false_row];
    if (!(event_probability < kLowProbabilitySampleThreshold)) {
        return std::nullopt;
    }
    RareInfo info;
    info.event_probability = event_probability;
    if (event_probability > 0.0) {
        const double inv_event = 1.0 / event_probability;
        for (std::size_t row = 0; row < distribution.assignments.size(); ++row) {
            if (row == *false_row || distribution.probabilities[row] <= 0.0) {
                continue;
            }
            info.event_rows.push_back(static_cast<int>(row));
            info.event_probabilities.push_back(distribution.probabilities[row] * inv_event);
        }
    }
    return info;
}

void push_bernoulli_sample_group(std::vector<BernoulliSampleGroup>& groups, double probability, int condition) {
    for (auto& group : groups) {
        if (group.probability == probability) {
            group.conditions.push_back(condition);
            return;
        }
    }
    groups.push_back(BernoulliSampleGroup{probability, {condition}});
}

void push_rare_categorical_sample_group(
    std::vector<RareCategoricalSampleGroup>& groups,
    const SymbolicCategoricalDistribution& distribution,
    const RareInfo& info) {
    for (auto& group : groups) {
        if (group.event_probability == info.event_probability &&
            group.nbits == distribution.nbits &&
            group.assignments == distribution.assignments &&
            group.probabilities == distribution.probabilities &&
            group.event_rows == info.event_rows &&
            group.event_probabilities == info.event_probabilities) {
            group.conditions.push_back(distribution.conditions);
            return;
        }
    }
    groups.push_back(RareCategoricalSampleGroup{
        info.event_probability,
        distribution.nbits,
        {distribution.conditions},
        distribution.assignments,
        distribution.probabilities,
        info.event_rows,
        info.event_probabilities,
    });
}

void build_categorical_sample_plan(
    const std::vector<SymbolicCategoricalDistribution>& distributions,
    std::vector<SymbolicCategoricalDistribution>& scalar,
    std::vector<RareCategoricalSampleGroup>& rare_groups) {
    for (const auto& distribution : distributions) {
        const auto info = rare_categorical_sample_info(distribution);
        if (info) {
            push_rare_categorical_sample_group(rare_groups, distribution, *info);
        } else {
            scalar.push_back(distribution);
        }
    }
}

} // namespace

FactoredInstructionProgram::FactoredInstructionProgram(
    int n_,
    int initial_k_,
    std::vector<FactoredInstruction> instructions_,
    int max_k_,
    SymbolicContext context_,
    std::vector<int> pending_prefix_instruction_indices_)
    : n(checked_nqubits(n_)),
      initial_k(checked_nqubits(initial_k_)),
      max_k(checked_nqubits(max_k_)),
      instructions(std::move(instructions_)),
      pending_prefix_instruction_indices(std::move(pending_prefix_instruction_indices_)),
      context(std::move(context_)) {
    if (initial_k > n || max_k > n || initial_k > max_k) {
        fail("invalid factored instruction program dimensions");
    }
    reduce_program_symbolic_expressions(instructions);
    int record_count = 0;
    int detector_count = 0;
    for (auto& instruction : instructions) {
        refresh_instruction_plans(instruction);
        context.bump_next_condition(max_condition(instruction));
        std::visit(
            [&](const auto& inst) {
                if constexpr (requires { inst.record; }) {
                    if (inst.record) {
                        record_count = std::max(record_count, *inst.record);
                    }
                } else if constexpr (requires { inst.detector; }) {
                    detector_count = std::max(detector_count, inst.detector);
                }
            },
            instruction);
    }
    nsymbols = std::max(0, context.next_condition - 1);
    nrecords = record_count;
    ndetectors = detector_count;
    build_categorical_sample_plan(
        context.categorical_distributions,
        sampled_categorical_distributions,
        sampled_rare_categorical_groups);
    for (const auto& [condition, probability] : context.bernoulli_probabilities) {
        if (probability < kLowProbabilitySampleThreshold) {
            push_bernoulli_sample_group(sampled_low_probability_bernoulli_groups, probability, condition);
        } else {
            sampled_bernoulli_conditions.push_back(condition);
            sampled_bernoulli_probabilities.push_back(probability);
        }
    }
    active_component_plan = build_active_component_plan(*this);
    use_active_components =
        active_component_plan != nullptr && active_component_plan->selected;
}

FactoredInstructionProgram factored_instruction_program(const PendingFactoredState& state) {
    return FactoredInstructionProgram(
        state.n,
        state.initial_k,
        state.instructions,
        state.max_k,
        *state.context,
        state.pending_prefix_instruction_indices);
}

FactoredInstructionProgram factored_instruction_program(PendingFactoredState&& state) {
    return FactoredInstructionProgram(
        state.n,
        state.initial_k,
        std::move(state.instructions),
        state.max_k,
        *state.context,
        std::move(state.pending_prefix_instruction_indices));
}

FactoredInstructionProgram plan_factored_updates(PendingFactoredState& state) {
    process_pending_operations_in_place(state);
    return factored_instruction_program(state);
}

FactoredInstructionProgram plan_factored_updates(PendingFactoredState&& state) {
    process_pending_operations_in_place(state);
    return factored_instruction_program(std::move(state));
}

} // namespace symft
