#include "factored/factored_internal.hpp"
#include "sampler/component_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace symft {
using namespace detail;

bool has_pending_operations(const PendingFactoredState& state) {
    return state.pending_operation_cursor < state.pending_operations.size();
}

namespace {

void set_planning_active_count(PendingFactoredState& state, int k) {
    const int ki = checked_nqubits(k);
    if (ki > state.n) {
        fail("active qubit count exceeds total qubit count");
    }
    state.k = ki;
    state.max_k = std::max(state.max_k, state.k);
}

FactoredInstruction push_instruction(PendingFactoredState& state, FactoredInstruction instruction) {
    state.context->bump_next_condition(max_condition(instruction));
    state.instructions.push_back(std::move(instruction));
    return state.instructions.back();
}

std::optional<int> measurement_record(PendingFactoredState& state, const PendingPauliMeasurement& measurement) {
    if (measurement.exp_val) {
        return std::nullopt;
    }
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

SymbolicBool measurement_relation(int record_condition, const SymbolicBool& outcome) {
    return xor_bool(symbolic_bool(record_condition), outcome);
}

void ensure_pending_operation_blocks(PendingFactoredState& state) {
    if (state.pending_operation_blocks_valid) {
        return;
    }
    const std::size_t blocks = (state.pending_operations.size() + 63) >> 6;
    const std::size_t n = static_cast<std::size_t>(state.n);
    state.pending_x_operation_blocks.assign(blocks * n, 0);
    state.pending_z_operation_blocks.assign(blocks * n, 0);
    for (std::size_t operation = 0; operation < state.pending_operations.size(); ++operation) {
        const PauliString* body = std::visit(
            [](const auto& typed) -> const PauliString* {
                if constexpr (requires { typed.pauli.pauli; }) {
                    return &typed.pauli.pauli;
                }
                return nullptr;
            },
            state.pending_operations[operation]);
        if (body == nullptr) {
            continue;
        }
        const std::size_t base = (operation >> 6) * n;
        const std::uint64_t mask = std::uint64_t{1} << (operation & 63);
        for (std::size_t word = 0; word < body->x.size(); ++word) {
            std::uint64_t x_bits = body->x[word];
            std::uint64_t z_bits = body->z[word];
            while (x_bits) {
                const std::size_t q = word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
                if (q < n) {
                    state.pending_x_operation_blocks[base + q] |= mask;
                }
                x_bits &= x_bits - 1;
            }
            while (z_bits) {
                const std::size_t q = word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
                if (q < n) {
                    state.pending_z_operation_blocks[base + q] |= mask;
                }
                z_bits &= z_bits - 1;
            }
        }
    }
    state.pending_operation_blocks_valid = true;
}

void append_pending_branch_sign(
    PendingOperation& operation,
    int branch_condition) {
    std::visit(
        [&](auto& typed) {
            if constexpr (requires { typed.pauli.sign; }) {
                auto& conditions = typed.pauli.sign.conditions;
                if (!conditions.empty() && conditions.back() >= branch_condition) {
                    fail("pending branch conditions must be appended in allocation order");
                }
                conditions.push_back(branch_condition);
            }
        },
        operation);
}

void push_symbolic_pauli_through_indexed_pending(
    PendingFactoredState& state,
    std::size_t start,
    const PauliString& pauli,
    int branch_condition) {
    ensure_pending_operation_blocks(state);
    const std::size_t n = static_cast<std::size_t>(state.n);
    const std::size_t blocks = (state.pending_operations.size() + 63) >> 6;
    for (std::size_t block = start >> 6; block < blocks; ++block) {
        const std::size_t base = block * n;
        std::uint64_t anticommuting = 0;
        for (std::size_t word = 0; word < pauli.x.size(); ++word) {
            std::uint64_t x_bits = pauli.x[word];
            std::uint64_t z_bits = pauli.z[word];
            while (x_bits) {
                const std::size_t q = word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
                if (q < n) {
                    anticommuting ^= state.pending_z_operation_blocks[base + q];
                }
                x_bits &= x_bits - 1;
            }
            while (z_bits) {
                const std::size_t q = word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
                if (q < n) {
                    anticommuting ^= state.pending_x_operation_blocks[base + q];
                }
                z_bits &= z_bits - 1;
            }
        }
        if (block == (start >> 6) && (start & 63) != 0) {
            anticommuting &= ~((std::uint64_t{1} << (start & 63)) - 1);
        }
        while (anticommuting) {
            const int bit = trailing_zeros64(anticommuting);
            const std::size_t operation = block * 64 + static_cast<std::size_t>(bit);
            if (operation < state.pending_operations.size()) {
                append_pending_branch_sign(
                    state.pending_operations[operation],
                    branch_condition);
            }
            anticommuting &= anticommuting - 1;
        }
    }
}

void push_symbolic_pauli_through_pending(
    PendingFactoredState& state,
    const PauliString& pauli,
    int branch_condition) {
    push_symbolic_pauli_through_indexed_pending(
        state,
        state.pending_operation_cursor,
        pauli,
        branch_condition);
}

void substitute_pending_symbols(
    SymbolicBool& expression,
    const std::vector<std::optional<SymbolicBool>>& substitutions);

void reduce_pending_signs_by_measurement_relation(
    PendingFactoredState& state,
    std::optional<int> record_condition,
    const SymbolicBool& outcome) {
    if (!record_condition) {
        return;
    }
    SymbolicBool reduced_outcome = outcome;
    substitute_pending_symbols(reduced_outcome, state.pending_substitutions);
    const SymbolicBool relation = measurement_relation(*record_condition, reduced_outcome);
    state.context->bump_next_condition(relation);
    // The current operation remains at the planning cursor. Deferring the
    // equivalent substitution avoids rescanning the unprocessed suffix.
    const int pivot = *record_condition;
    const bool self_reference = std::binary_search(
        reduced_outcome.conditions.begin(), reduced_outcome.conditions.end(), pivot);
    if (!self_reference &&
        (reduced_outcome.conditions.size() <= 1 ||
         has_lower_sampling_cost(reduced_outcome, symbolic_bool(pivot)))) {
        const std::size_t pivot_index = static_cast<std::size_t>(pivot);
        if (state.pending_substitutions.size() <= pivot_index) {
            state.pending_substitutions.resize(pivot_index + 1);
        }
        state.pending_substitutions[pivot_index] = std::move(reduced_outcome);
        return;
    }
    state.pending_relation_reducer.add(relation);
}

void substitute_pending_symbols(
    SymbolicBool& expression,
    const std::vector<std::optional<SymbolicBool>>& substitutions) {
    if (expression.conditions.empty() || substitutions.empty()) {
        return;
    }
    std::vector<int> expanded = expression.conditions;
    while (true) {
        bool changed = false;
        std::vector<int> next;
        next.reserve(expanded.size());
        for (int condition : expanded) {
            const std::size_t condition_index = static_cast<std::size_t>(condition);
            if (condition_index >= substitutions.size() ||
                !substitutions[condition_index]) {
                next.push_back(condition);
            } else {
                const auto& replacement = *substitutions[condition_index];
                changed = true;
                expression.constant ^= replacement.constant;
                next.insert(
                    next.end(),
                    replacement.conditions.begin(),
                    replacement.conditions.end());
            }
        }
        expanded = normalize_conditions(std::move(next));
        if (!changed) {
            break;
        }
    }
    expression.conditions = std::move(expanded);
}

void reduce_pending_symbolic_bool(
    SymbolicBool& expression,
    PendingFactoredState& state) {
    substitute_pending_symbols(expression, state.pending_substitutions);
    expression = state.pending_relation_reducer.reduce(std::move(expression));
}

void reduce_pending_operation_signs(
    PendingOperation& operation,
    PendingFactoredState& state) {
    std::visit(
        [&](auto& typed) {
            if constexpr (requires { typed.pauli.sign; }) {
                reduce_pending_symbolic_bool(typed.pauli.sign, state);
            } else if constexpr (requires { typed.outcome; }) {
                reduce_pending_symbolic_bool(typed.outcome, state);
            }
        },
        operation);
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
    const SymbolicBool sign = rotation_sign_from_pauli(current.pauli);
    state.tableau.promote_dormant_rotation(current.pauli.pauli, old_k, picked_dormant);
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
    const SymbolicBool& outcome) {
    const auto instruction = push_instruction(
        state,
        RecordMeasurement{
            outcome,
            measurement_record(state, measurement),
            measurement.record_condition,
            SymbolicBoolEvaluationPlan(outcome),
            measurement.exp_val,
        });
    reduce_pending_signs_by_measurement_relation(
        state,
        measurement.record_condition,
        outcome);
    return instruction;
}

std::optional<FactoredInstruction> measure_dormant_xy_pauli(
    PendingFactoredState& state,
    PendingPauliMeasurement current,
    int picked_dormant) {
    if (current.exp_val) {
        return push_instruction(
            state,
            IntroduceDormantMeasurementBranch{
                0,
                SymbolicBool(false),
                std::nullopt,
                std::nullopt,
                SymbolicBoolEvaluationPlan(SymbolicBool(false)),
                current.exp_val,
            });
    }
    const SymbolicBool base_outcome = measurement_base_outcome(current.pauli);
    const PauliString correction =
        state.tableau.replace_dormant_measurement(current.pauli.pauli, state.k, picked_dormant);
    const int branch = state.context->fresh_condition();
    const SymbolicBool branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending(
        state,
        correction,
        branch);
    const SymbolicBool outcome = xor_bool(base_outcome, branch_bit);
    const auto instruction = push_instruction(
        state,
        IntroduceDormantMeasurementBranch{
            branch,
            outcome,
            measurement_record(state, current),
            current.record_condition,
            SymbolicBoolEvaluationPlan(outcome),
            std::nullopt,
        });
    reduce_pending_signs_by_measurement_relation(
        state,
        current.record_condition,
        outcome);
    return instruction;
}

std::optional<FactoredInstruction> evaluate_active_pauli(
    PendingFactoredState& state,
    const PendingPauliMeasurement& current,
    const PauliString& active_body,
    const SymbolicBool& base_outcome) {
    if (!pauli_squares_to_identity(active_body)) {
        fail("active expectation Pauli must square to identity");
    }
    const PrecomputedActivePauliMeasurementKernel kernel(active_body);
    return push_instruction(
        state,
        MeasurePrecomputedActivePauli{
            active_body,
            kernel,
            0,
            base_outcome,
            std::nullopt,
            std::nullopt,
            SymbolicBoolEvaluationPlan(base_outcome),
            current.exp_val,
        });
}

std::optional<FactoredInstruction> measure_active_pauli_branches(
    PendingFactoredState& state,
    const PendingPauliMeasurement& current,
    const PauliString& active_body,
    const SymbolicBool& base_outcome) {
    if (!pauli_squares_to_identity(active_body)) {
        fail("active measurement Pauli must square to identity");
    }
    PrecomputedActivePauliMeasurementKernel kernel(active_body);
    const PauliString correction = state.tableau.remove_active_measurement(
        current.pauli.pauli,
        state.k,
        kernel.pivot,
        kernel.is_diagonal);
    const int branch = state.context->fresh_condition();
    const SymbolicBool branch_bit = symbolic_bool(branch);
    push_symbolic_pauli_through_pending(
        state,
        correction,
        branch);
    const SymbolicBool outcome = xor_bool(base_outcome, branch_bit);
    MeasurePrecomputedActivePauli instruction{
        active_body,
        std::move(kernel),
        branch,
        outcome,
        measurement_record(state, current),
        current.record_condition,
        SymbolicBoolEvaluationPlan(outcome),
        std::nullopt,
    };
    FactoredInstruction pushed = push_instruction(state, std::move(instruction));
    reduce_pending_signs_by_measurement_relation(
        state,
        current.record_condition,
        outcome);
    set_planning_active_count(state, state.k - 1);
    return pushed;
}

std::optional<FactoredInstruction> process_pending_measurement_impl(
    PendingFactoredState& state,
    const PendingPauliMeasurement& measurement) {
    state.context->bump_next_condition(max_condition(measurement));
    PendingPauliMeasurement current = measurement;
    current.pauli.pauli = state.tableau.decompose(measurement.pauli.pauli);
    const PauliString active_body = project_pauli_body(current.pauli.pauli, 0, state.k);
    const auto picked = highest_dormant_x_qubit(state, current.pauli.pauli);
    if (picked) {
        return measure_dormant_xy_pauli(state, current, *picked);
    }
    const SymbolicBool base_outcome = measurement_base_outcome(current.pauli);
    if (!active_body.has_nonidentity_body()) {
        return record_deterministic_measurement(state, current, base_outcome);
    }
    if (current.exp_val) {
        return evaluate_active_pauli(state, current, active_body, base_outcome);
    }
    return measure_active_pauli_branches(state, current, active_body, base_outcome);
}

} // namespace

std::optional<FactoredInstruction> process_pending_rotation(
    PendingFactoredState& state,
    const PendingPauliRotation& rotation) {
    state.context->bump_next_condition(max_condition(rotation));
    PendingPauliRotation current = rotation;
    current.pauli.pauli = state.tableau.decompose(rotation.pauli.pauli);
    const auto picked = highest_dormant_x_qubit(state, current.pauli.pauli);
    if (!picked) {
        return process_diagonal_dormant_rotation(state, current);
    }
    return process_nondiagonal_dormant_rotation(state, current, *picked);
}

std::optional<FactoredInstruction> process_pending_measurement(
    PendingFactoredState& state,
    const PendingPauliMeasurement& measurement) {
    return process_pending_measurement_impl(state, measurement);
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
            std::nullopt,
        });
}

std::optional<FactoredInstruction> process_next_pending_operation(PendingFactoredState& state) {
    if (!has_pending_operations(state)) {
        return std::nullopt;
    }
    if (state.pending_prefix_instruction_indices.empty()) {
        state.pending_prefix_instruction_indices.push_back(static_cast<int>(state.instructions.size()));
    }
    PendingOperation operation = state.pending_operations[
        state.pending_operation_cursor];
    reduce_pending_operation_signs(operation, state);
    // From this point on, the cursor denotes the first later operation. Any
    // measurement correction is propagated only through that unprocessed suffix.
    ++state.pending_operation_cursor;
    const std::size_t start = state.instructions.size();
    std::optional<FactoredInstruction> result = std::visit(
        [&](const auto& op) -> std::optional<FactoredInstruction> {
            if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PendingPauliRotation>) {
                return process_pending_rotation(state, op);
            } else if constexpr (std::is_same_v<std::decay_t<decltype(op)>, PendingPauliMeasurement>) {
                return process_pending_measurement_impl(state, op);
            } else {
                return process_pending_classical_record(state, op);
            }
        },
        operation);
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
void reduce_instruction_symbolic_expressions(T& instruction, const SymbolicRelationReducer& reducer) {
    if constexpr (requires { instruction.sign; }) {
        instruction.sign = reducer.reduce(std::move(instruction.sign));
    }
    if constexpr (requires { instruction.outcome; }) {
        instruction.outcome = reducer.reduce(std::move(instruction.outcome));
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
    SymbolicRelationReducer reducer;
    for (auto& instruction : instructions) {
        std::visit(
            [&](auto& inst) {
                reduce_instruction_symbolic_expressions(inst, reducer);
            },
            instruction);
        const auto relation = std::visit(
            [](const auto& inst) -> std::optional<SymbolicBool> {
                return measurement_relation_from_instruction(inst);
            },
            instruction);
        if (relation && (!relation->conditions.empty() || relation->constant)) {
            reducer.add(*relation);
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
    int exp_val_count = 0;
    for (auto& instruction : instructions) {
        refresh_instruction_plans(instruction);
        context.bump_next_condition(max_condition(instruction));
        std::visit(
            [&](const auto& inst) {
                if constexpr (requires { inst.record; }) {
                    if (inst.record) {
                        record_count = std::max(record_count, *inst.record);
                    }
                }
                if constexpr (requires { inst.detector; }) {
                    detector_count = std::max(detector_count, inst.detector);
                }
                if constexpr (requires { inst.exp_val; }) {
                    if (inst.exp_val) {
                        exp_val_count = std::max(exp_val_count, *inst.exp_val + 1);
                    }
                }
            },
            instruction);
    }
    nsymbols = std::max(0, context.next_condition - 1);
    nrecords = record_count;
    ndetectors = detector_count;
    nexpvals = exp_val_count;
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
