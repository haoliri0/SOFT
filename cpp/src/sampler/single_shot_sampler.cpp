#include "active_kernels.hpp"

#include "sampler/contiguous_active.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/component_plan.hpp"
#include "sampler/random.hpp"
#include "sampler/single_shot.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace symft {
using namespace detail;

namespace {

constexpr int kDefaultSingleShotSampleChunkShots = 2048;
constexpr std::uint64_t kSingleShotBranchSeedXor = 0x5eed1234ULL;

void check_symbol_slot(const FactoredExecutorState& runtime, int condition) {
    if (condition <= 0 || condition > runtime.nsymbols) {
        fail("symbolic condition exceeds executor symbol table");
    }
}

bool is_assigned(const FactoredExecutorState& runtime, int condition) {
    check_symbol_slot(runtime, condition);
    const std::size_t word = symbol_word_index(condition);
    return word < runtime.assigned_words.size() && (runtime.assigned_words[word] & symbol_bit_mask(condition)) != 0;
}

void set_symbol_assignment_unchecked(FactoredExecutorState& runtime, int condition, bool value) {
    const std::size_t word = symbol_word_index(condition);
    const std::uint64_t mask = symbol_bit_mask(condition);
    runtime.assigned_words[word] |= mask;
    if (value) {
        runtime.value_words[word] |= mask;
    } else {
        runtime.value_words[word] &= ~mask;
    }
}

void set_assigned_symbol_true_unchecked(FactoredExecutorState& runtime, int condition) {
    const std::size_t word = symbol_word_index(condition);
    const std::uint64_t mask = symbol_bit_mask(condition);
    runtime.value_words[word] |= mask;
}

void assign_symbol(FactoredExecutorState& runtime, std::optional<int> condition, bool value) {
    if (!condition) {
        return;
    }
    check_symbol_slot(runtime, *condition);
    const std::size_t word = symbol_word_index(*condition);
    const std::uint64_t mask = symbol_bit_mask(*condition);
    if ((runtime.assigned_words[word] & mask) != 0) {
        if (((runtime.value_words[word] & mask) != 0) != value) {
            fail("symbolic condition was assigned inconsistent concrete values");
        }
        return;
    }
    set_symbol_assignment_unchecked(runtime, *condition, value);
}

void assign_symbol(FactoredExecutorState& runtime, int condition, bool value) {
    assign_symbol(runtime, std::optional<int>(condition), value);
}

bool eval_symbolic_bool_packed(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    const std::size_t max_word = static_cast<std::size_t>(plan.word_indices.back());
    if (max_word >= runtime.assigned_words.size()) {
        fail("symbolic condition expression has no concrete value");
    }
    const auto* word_indices = plan.word_indices.data();
    const auto* word_masks = plan.word_masks.data();
    const auto* assigned_words = runtime.assigned_words.data();
    const auto* value_words = runtime.value_words.data();
    std::uint64_t parity_bits = 0;
    std::uint64_t missing = 0;
    for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
        const std::size_t word = static_cast<std::size_t>(word_indices[i]);
        const std::uint64_t mask = word_masks[i];
        missing |= mask & ~assigned_words[word];
        parity_bits ^= value_words[word] & mask;
    }
    if (missing != 0) {
        fail("symbolic condition expression has no concrete value");
    }
    return plan.constant != is_odd_popcount(parity_bits);
}

bool eval_symbolic_bool_packed_unchecked(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    const auto* word_indices = plan.word_indices.data();
    const auto* word_masks = plan.word_masks.data();
    const auto* value_words = runtime.value_words.data();
    std::uint64_t parity_bits = 0;
    for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
        const std::size_t word = static_cast<std::size_t>(word_indices[i]);
        parity_bits ^= value_words[word] & word_masks[i];
    }
    return plan.constant != is_odd_popcount(parity_bits);
}

bool eval_symbolic_bool_scalar(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    bool out = plan.constant;
    for (int condition : plan.conditions) {
        check_symbol_slot(runtime, condition);
        const std::size_t word = symbol_word_index(condition);
        const std::uint64_t mask = symbol_bit_mask(condition);
        if ((runtime.assigned_words[word] & mask) == 0) {
            fail("symbolic condition expression has no concrete value");
        }
        out = out != ((runtime.value_words[word] & mask) != 0);
    }
    return out;
}

[[maybe_unused]] bool eval_symbolic_bool(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    if (plan.word_indices.empty()) {
        return eval_symbolic_bool_scalar(plan, runtime);
    }
    return eval_symbolic_bool_packed(plan, runtime);
}

bool eval_symbolic_bool_unchecked(const SymbolicBoolEvaluationPlan& plan, const FactoredExecutorState& runtime) {
    // Hot planned execution relies on the planner's assignment-before-use invariant.
    // Keep eval_symbolic_bool above as the checked counterpart for validation/debugging.
    if (plan.word_indices.empty()) {
        return plan.constant;
    }
    return eval_symbolic_bool_packed_unchecked(plan, runtime);
}

bool record_parity_from_measurements(
    const std::vector<std::uint64_t>& measurement_words,
    int nrecords,
    const std::vector<int>& records) {
    bool parity = false;
    for (int record : records) {
        if (record <= 0 || record > nrecords) {
            fail("detector references an out-of-range measurement record");
        }
        parity = parity != packed_bit(measurement_words, record - 1);
    }
    return parity;
}

bool detector_outcome_from_runtime(
    const FactoredExecutorState& runtime,
    const RecordDetector& instruction) {
    if (!instruction.records.empty()) {
        return record_parity_from_measurements(
            runtime.measurement_words,
            runtime.nrecords,
            instruction.records);
    }
    return eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
}

int normalize_single_shot_sample_chunk_shots(int sample_chunk_shots) {
    if (sample_chunk_shots < 0) {
        fail("single-shot sample chunk count must be nonnegative");
    }
    return sample_chunk_shots > 0 ? sample_chunk_shots : kDefaultSingleShotSampleChunkShots;
}

struct SingleShotExpressionEvaluator {
    const PresampledExpressionPlan& expression_plan;
    const PresampledExpressionBlock& expression_block;
    int shot_index = 0;

    bool eval(std::size_t instruction_index, const FactoredExecutorState& runtime) const {
        if (instruction_index >= expression_plan.instruction_expressions.size()) {
            fail("single-shot presampled expression plan does not match program");
        }
        const auto& expression = expression_plan.instruction_expressions[instruction_index];
        if (shot_index < 0 || shot_index >= expression_block.nshots) {
            fail("single-shot presampled expression shot index is out of range");
        }
        if (expression.block_expression_index < 0 ||
            expression.block_expression_index >= static_cast<int>(expression_plan.block_expressions.size())) {
            fail("single-shot presampled expression references an out-of-range block expression");
        }
        bool out = presampled_expression_block_bit(
            expression_block,
            expression.block_expression_index,
            shot_index);
        if (!expression.residual_plan.conditions.empty()) {
            out = out != eval_symbolic_bool_unchecked(expression.residual_plan, runtime);
        }
        return out;
    }
};

void write_measurement_record(
    FactoredExecutorState& runtime,
    std::optional<int> record,
    bool outcome,
    std::optional<int> record_condition) {
    if (record) {
        if (*record <= 0) {
            fail("measurement record id must be positive");
        }
        if (*record > runtime.nrecords) {
            runtime.nrecords = *record;
        }
        const std::size_t nwords = symbol_word_count(runtime.nrecords);
        if (runtime.measurement_words.size() < nwords) {
            runtime.measurement_words.resize(nwords, 0);
        }
        const std::size_t word = symbol_word_index(*record);
        const std::uint64_t mask = symbol_bit_mask(*record);
        if (outcome) {
            runtime.measurement_words[word] |= mask;
        } else {
            runtime.measurement_words[word] &= ~mask;
        }
    }
    assign_symbol(runtime, record_condition, outcome);
}

void write_detector_record(FactoredExecutorState& runtime, int detector, bool outcome) {
    if (detector <= 0) {
        fail("detector id must be positive");
    }
    if (detector > runtime.ndetectors) {
        runtime.ndetectors = detector;
    }
    const std::size_t nwords = symbol_word_count(runtime.ndetectors);
    if (runtime.detector_words.size() < nwords) {
        runtime.detector_words.resize(nwords, 0);
    }
    const std::size_t word = symbol_word_index(detector);
    const std::uint64_t mask = symbol_bit_mask(detector);
    if (outcome) {
        runtime.detector_words[word] |= mask;
    } else {
        runtime.detector_words[word] &= ~mask;
    }
}

void sample_categorical_distribution(
    FactoredExecutorState& runtime,
    const std::vector<int>& conditions,
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    if (static_cast<int>(conditions.size()) != nbits) {
        fail("categorical condition count does not match assignment bit count");
    }
    bool any_assigned = false;
    bool all_assigned = true;
    for (int condition : conditions) {
        const bool assigned = is_assigned(runtime, condition);
        any_assigned = any_assigned || assigned;
        all_assigned = all_assigned && assigned;
    }
    if (all_assigned) {
        return;
    }
    if (any_assigned) {
        fail("categorical symbolic distribution was only partially preassigned");
    }
    const int row = sample_categorical_row(runtime.rng_state, probabilities);
    for (std::size_t i = 0; i < conditions.size(); ++i) {
        assign_symbol(runtime, conditions[i], packed_bit(assignments[static_cast<std::size_t>(row)], static_cast<int>(i)));
    }
}

bool any_assigned(const FactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        if (is_assigned(runtime, condition)) {
            return true;
        }
    }
    return false;
}

bool any_categorical_group_assigned(
    const FactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        if (any_assigned(runtime, conditions)) {
            return true;
        }
    }
    return false;
}

void assign_conditions_false(FactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        check_symbol_slot(runtime, condition);
        set_symbol_assignment_unchecked(runtime, condition, false);
    }
}

void assign_categorical_group_false(
    FactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        assign_conditions_false(runtime, conditions);
    }
}

void sample_rare_categorical_group(FactoredExecutorState& runtime, const RareCategoricalSampleGroup& group) {
    if (any_categorical_group_assigned(runtime, group.conditions)) {
        for (const auto& conditions : group.conditions) {
            sample_categorical_distribution(runtime, conditions, group.nbits, group.assignments, group.probabilities);
        }
        return;
    }

    assign_categorical_group_false(runtime, group.conditions);
    if (group.event_probability <= 0.0) {
        return;
    }
    int idx = 0;
    const int nsets = static_cast<int>(group.conditions.size());
    while (true) {
        const int gap = static_cast<int>(sample_geometric_gap(runtime.rng_state, group.event_probability));
        if (gap >= nsets - idx) {
            return;
        }
        idx += gap;
        const int row = group.event_rows[static_cast<std::size_t>(sample_categorical_row(runtime.rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                set_assigned_symbol_true_unchecked(runtime, conditions[bit_idx]);
            }
        }
        ++idx;
    }
}

void sample_low_probability_bernoulli_group(FactoredExecutorState& runtime, const BernoulliSampleGroup& group) {
    if (any_assigned(runtime, group.conditions)) {
        for (int condition : group.conditions) {
            if (!is_assigned(runtime, condition)) {
                assign_symbol(runtime, condition, sample_bernoulli(runtime.rng_state, group.probability));
            }
        }
        return;
    }

    assign_conditions_false(runtime, group.conditions);
    if (group.probability <= 0.0) {
        return;
    }
    int idx = 0;
    const int nconditions = static_cast<int>(group.conditions.size());
    while (true) {
        const int gap = static_cast<int>(sample_geometric_gap(runtime.rng_state, group.probability));
        if (gap >= nconditions - idx) {
            return;
        }
        idx += gap;
        set_assigned_symbol_true_unchecked(runtime, group.conditions[static_cast<std::size_t>(idx)]);
        ++idx;
    }
}

void sample_exogenous_symbols(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    for (const auto& distribution : program.sampled_categorical_distributions) {
        sample_categorical_distribution(runtime, distribution.conditions, distribution.nbits, distribution.assignments, distribution.probabilities);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        sample_rare_categorical_group(runtime, group);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        const int condition = program.sampled_bernoulli_conditions[i];
        if (!is_assigned(runtime, condition)) {
            assign_symbol(runtime, condition, sample_bernoulli(runtime.rng_state, program.sampled_bernoulli_probabilities[i]));
        }
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        sample_low_probability_bernoulli_group(runtime, group);
    }
}

void assign_presampled_exogenous(FactoredExecutorState& runtime, const PresampledExogenous& samples, int shot_index) {
    if (shot_index < 0 || shot_index >= samples.nshots) {
        fail("presampled shot index is out of range");
    }
    if (runtime.nsymbols != samples.nsymbols || runtime.value_words.size() < samples.nwords ||
        runtime.assigned_words.size() < samples.nwords ||
        samples.exogenous_assigned_words.size() != samples.nwords ||
        samples.value_words.size() != static_cast<std::size_t>(samples.nshots) * samples.nwords) {
        fail("presampled exogenous table does not match executor symbol table");
    }
    const std::size_t base = static_cast<std::size_t>(shot_index) * samples.nwords;
    for (std::size_t word = 0; word < samples.nwords; ++word) {
        const std::uint64_t exogenous = samples.exogenous_assigned_words[word];
        const std::uint64_t row = samples.value_words[base + word] & exogenous;
        const std::uint64_t overlap = runtime.assigned_words[word] & exogenous;
        if ((overlap & (runtime.value_words[word] ^ row)) != 0) {
            fail("presampled exogenous condition conflicts with an existing assignment");
        }
        runtime.assigned_words[word] |= exogenous;
        runtime.value_words[word] |= row;
    }
}

std::size_t runtime_active_dim(const FactoredExecutorState& runtime) {
    return active_length(runtime.k);
}

void ensure_runtime_active_capacity(FactoredExecutorState& runtime, int max_k) {
    const std::size_t max_dim = active_length(max_k);
    if (runtime.active_re.size() < max_dim) {
        runtime.active_re.resize(max_dim, 0.0);
    }
    if (runtime.active_im.size() < max_dim) {
        runtime.active_im.resize(max_dim, 0.0);
    }
    if (runtime.active_scratch_re.size() < max_dim) {
        runtime.active_scratch_re.resize(max_dim, 0.0);
    }
    if (runtime.active_scratch_im.size() < max_dim) {
        runtime.active_scratch_im.resize(max_dim, 0.0);
    }
}

void promote_first_dormant_rotation(FactoredExecutorState& runtime, double kernel_angle) {
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = runtime_active_dim(runtime);
    const std::size_t promoted_dim = 2 * dim;
    if (runtime.active_re.size() < promoted_dim) {
        runtime.active_re.resize(promoted_dim, 0.0);
    }
    if (runtime.active_im.size() < promoted_dim) {
        runtime.active_im.resize(promoted_dim, 0.0);
    }
    if (runtime.active_scratch_re.size() < promoted_dim) {
        runtime.active_scratch_re.resize(promoted_dim, 0.0);
    }
    if (runtime.active_scratch_im.size() < promoted_dim) {
        runtime.active_scratch_im.resize(promoted_dim, 0.0);
    }
    const double c = std::cos(kernel_angle);
    const double s = std::sin(kernel_angle);
    promote_contiguous_active(
        runtime.active_re.data(),
        runtime.active_im.data(),
        dim,
        c,
        -s);
    ++runtime.k;
    --runtime.ndormant;
}

void rotate_pauli(FactoredExecutorState& runtime, const PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match active state");
    }
    rotate_contiguous_active(
        runtime.active_re.data(),
        runtime.active_im.data(),
        runtime_active_dim(runtime),
        kernel,
        sign);
}

double active_diagonal_measurement_branch_probability(
    const FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    return diagonal_probability_contiguous(
        runtime.active_re.data(),
        runtime.active_im.data(),
        kernel,
        branch);
}

double active_measurement_branch_probability(
    const FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch) {
    return nondiagonal_probability_contiguous(
        runtime.active_re.data(),
        runtime.active_im.data(),
        kernel,
        branch);
}

void project_diagonal_active_pauli_measurement(
    FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const double invnorm = 1.0 / std::sqrt(probability);
    project_diagonal_contiguous(
        runtime.active_re.data(),
        runtime.active_im.data(),
        kernel,
        branch,
        invnorm);
    --runtime.k;
}

void project_active_pauli_measurement(
    FactoredExecutorState& runtime,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double probability) {
    if (probability <= 0.0) {
        fail("sampled an impossible active measurement branch");
    }
    const std::size_t out_dim = kernel.out_dim;
    const double invnorm = 1.0 / std::sqrt(probability);
    if (runtime.active_scratch_re.size() < out_dim) {
        runtime.active_scratch_re.resize(out_dim, 0.0);
    }
    if (runtime.active_scratch_im.size() < out_dim) {
        runtime.active_scratch_im.resize(out_dim, 0.0);
    }
    project_nondiagonal_contiguous(
        runtime.active_re.data(),
        runtime.active_im.data(),
        runtime.active_scratch_re.data(),
        runtime.active_scratch_im.data(),
        kernel,
        branch,
        invnorm);
    --runtime.k;
}

void execute_instruction(
    FactoredExecutorState& runtime,
    const ApplyPrecomputedActivePauliRotation& instruction);
void execute_instruction(
    FactoredExecutorState& runtime,
    const PromoteDormantRotation& instruction);
void execute_instruction(
    FactoredExecutorState& runtime,
    const RecordMeasurement& instruction);
void execute_instruction(
    FactoredExecutorState& runtime,
    const RecordDetector& instruction);
void execute_instruction(
    FactoredExecutorState& runtime,
    const MeasurePrecomputedActivePauli& instruction);
void execute_instruction(
    FactoredExecutorState& runtime,
    const IntroduceDormantMeasurementBranch& instruction);

void configure_single_shot_components(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program) {
    if (program.use_active_components &&
        (program.active_component_plan == nullptr ||
         program.active_component_plan->instruction_steps.size() !=
             program.instructions.size())) {
        fail("program does not contain an executable active component plan");
    }
    const bool enabled =
        program.use_active_components &&
        program.active_component_plan != nullptr;
    runtime.active_components_enabled = enabled;
    if (!enabled) {
        runtime.active_components.clear();
        ensure_runtime_active_capacity(runtime, program.max_k);
        return;
    }

    const auto& plan = *program.active_component_plan;
    if (plan.component_count !=
        static_cast<int>(plan.component_max_k.size())) {
        fail("active component plan has inconsistent component capacities");
    }
    runtime.active_components.resize(
        static_cast<std::size_t>(plan.component_count));
    for (int component = 0; component < plan.component_count; ++component) {
        auto& buffer =
            runtime.active_components[static_cast<std::size_t>(component)];
        const std::size_t capacity = active_length(
            plan.component_max_k[static_cast<std::size_t>(component)]);
        buffer.re.resize(capacity, 0.0);
        buffer.im.resize(capacity, 0.0);
        buffer.scratch_re.resize(capacity, 0.0);
        buffer.scratch_im.resize(capacity, 0.0);
    }
    std::vector<double>().swap(runtime.active_re);
    std::vector<double>().swap(runtime.active_im);
    std::vector<double>().swap(runtime.active_scratch_re);
    std::vector<double>().swap(runtime.active_scratch_im);
}

void reset_single_shot_components(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program) {
    if (program.use_active_components &&
        (program.active_component_plan == nullptr ||
         program.active_component_plan->instruction_steps.size() !=
             program.instructions.size())) {
        fail("program does not contain an executable active component plan");
    }
    const bool should_enable =
        program.use_active_components &&
        program.active_component_plan != nullptr;
    if (runtime.active_components_enabled != should_enable ||
        (should_enable &&
         runtime.active_components.size() !=
             program.active_component_plan->component_max_k.size())) {
        configure_single_shot_components(runtime, program);
    }
    if (!runtime.active_components_enabled) {
        return;
    }
    const auto& plan = *program.active_component_plan;
    for (auto& component : runtime.active_components) {
        component.k = 0;
        component.active = false;
    }
    if (plan.initial_components != program.initial_k) {
        fail("active component plan has the wrong initial component count");
    }
    for (int component = 0; component < plan.initial_components; ++component) {
        auto& buffer =
            runtime.active_components[static_cast<std::size_t>(component)];
        if (buffer.re.size() < 2 || buffer.im.size() < 2) {
            fail("initial active component has insufficient storage");
        }
        buffer.k = 1;
        buffer.active = true;
        buffer.re[0] = 1.0;
        buffer.im[0] = 0.0;
        buffer.re[1] = 0.0;
        buffer.im[1] = 0.0;
    }
}

void merge_single_shot_components(
    FactoredExecutorState& runtime,
    const ActiveComponentPlan& plan,
    std::uint32_t merge_offset,
    std::uint16_t merge_count,
    int expected_target) {
    if (merge_count == 0 ||
        static_cast<std::size_t>(merge_offset) + merge_count >
            plan.merge_components.size()) {
        fail("active component instruction has an invalid merge range");
    }
    const int target_id =
        plan.merge_components[static_cast<std::size_t>(merge_offset)];
    if (target_id != expected_target ||
        target_id < 0 ||
        target_id >= static_cast<int>(runtime.active_components.size())) {
        fail("active component merge has an invalid target");
    }
    auto& target =
        runtime.active_components[static_cast<std::size_t>(target_id)];
    if (!target.active) {
        fail("active component merge target is inactive");
    }
    for (std::uint16_t source_index = 1;
         source_index < merge_count;
         ++source_index) {
        const int source_id = plan.merge_components[
            static_cast<std::size_t>(merge_offset) + source_index];
        if (source_id < 0 ||
            source_id >= static_cast<int>(runtime.active_components.size()) ||
            source_id == target_id) {
            fail("active component merge has an invalid source");
        }
        auto& source =
            runtime.active_components[static_cast<std::size_t>(source_id)];
        if (!source.active) {
            fail("active component merge source is inactive");
        }
        const std::size_t target_dim = active_length(target.k);
        const std::size_t source_dim = active_length(source.k);
        const int merged_k = target.k + source.k;
        const std::size_t merged_dim = active_length(merged_k);
        if (target.re.size() < merged_dim ||
            target.im.size() < merged_dim) {
            fail("active component merge target has insufficient storage");
        }
        for (std::size_t source_basis = source_dim;
             source_basis-- > 0;) {
            const double source_re = source.re[source_basis];
            const double source_im = source.im[source_basis];
            const std::size_t output_base = source_basis << target.k;
            SYMFT_SINGLE_SIMD_LOOP
            for (std::size_t target_basis = 0;
                 target_basis < target_dim;
                 ++target_basis) {
                const double target_re = target.re[target_basis];
                const double target_im = target.im[target_basis];
                const std::size_t output = output_base + target_basis;
                target.re[output] =
                    target_re * source_re - target_im * source_im;
                target.im[output] =
                    target_re * source_im + target_im * source_re;
            }
        }
        target.k = merged_k;
        source.k = 0;
        source.active = false;
    }
}

class ScopedSingleShotComponent {
  public:
    ScopedSingleShotComponent(
        FactoredExecutorState& runtime_,
        SingleShotActiveComponent& component_)
        : runtime(runtime_),
          component(component_),
          global_k(runtime_.k),
          global_ndormant(runtime_.ndormant) {
        if (!component.active) {
            fail("cannot execute an instruction on an inactive component");
        }
        std::swap(runtime.active_re, component.re);
        std::swap(runtime.active_im, component.im);
        std::swap(runtime.active_scratch_re, component.scratch_re);
        std::swap(runtime.active_scratch_im, component.scratch_im);
        runtime.k = component.k;
    }

    ~ScopedSingleShotComponent() {
        component.k = runtime.k;
        std::swap(runtime.active_re, component.re);
        std::swap(runtime.active_im, component.im);
        std::swap(runtime.active_scratch_re, component.scratch_re);
        std::swap(runtime.active_scratch_im, component.scratch_im);
        runtime.k = global_k;
        runtime.ndormant = global_ndormant;
    }

  private:
    FactoredExecutorState& runtime;
    SingleShotActiveComponent& component;
    int global_k;
    int global_ndormant;
};

void execute_component_rotation(
    FactoredExecutorState& runtime,
    const ActiveComponentPlan& plan,
    const ActiveComponentRotationStep& step,
    bool sign) {
    merge_single_shot_components(
        runtime,
        plan,
        step.merge_offset,
        step.merge_count,
        step.component);
    auto& component =
        runtime.active_components[static_cast<std::size_t>(step.component)];
    ScopedSingleShotComponent scope(runtime, component);
    rotate_pauli(runtime, step.kernel, sign);
}

void execute_component_promotion(
    FactoredExecutorState& runtime,
    const ActiveComponentPromotionStep& step,
    double kernel_angle,
    bool sign) {
    if (runtime.ndormant <= 0 ||
        step.component < 0 ||
        step.component >= static_cast<int>(runtime.active_components.size())) {
        fail("active component promotion is invalid");
    }
    auto& component =
        runtime.active_components[static_cast<std::size_t>(step.component)];
    if (component.active || component.re.size() < 2 ||
        component.im.size() < 2) {
        fail("active component promotion target is unavailable");
    }
    const double c = std::cos(kernel_angle);
    const double s = std::sin(kernel_angle);
    component.k = 1;
    component.active = true;
    component.re[0] = c;
    component.im[0] = 0.0;
    component.re[1] = 0.0;
    component.im[1] = sign ? s : -s;
    ++runtime.k;
    --runtime.ndormant;
}

void execute_component_measurement(
    FactoredExecutorState& runtime,
    const ActiveComponentPlan& plan,
    const ActiveComponentMeasurementStep& step,
    int branch_condition) {
    merge_single_shot_components(
        runtime,
        plan,
        step.merge_offset,
        step.merge_count,
        step.component);
    auto& component =
        runtime.active_components[static_cast<std::size_t>(step.component)];
    {
        ScopedSingleShotComponent scope(runtime, component);
        double prob_true =
            step.kernel.is_diagonal
                ? active_diagonal_measurement_branch_probability(
                      runtime,
                      step.kernel,
                      true)
                : active_measurement_branch_probability(
                      runtime,
                      step.kernel,
                      true);
        prob_true = std::clamp(prob_true, 0.0, 1.0);
        const bool branch =
            sample_bernoulli(runtime.rng_state, prob_true);
        const double probability =
            branch ? prob_true : 1.0 - prob_true;
        assign_symbol(runtime, branch_condition, branch);
        if (step.kernel.is_diagonal) {
            project_diagonal_active_pauli_measurement(
                runtime,
                step.kernel,
                branch,
                probability);
        } else {
            project_active_pauli_measurement(
                runtime,
                step.kernel,
                branch,
                probability);
        }
    }
    --runtime.k;
    ++runtime.ndormant;
    if (step.deactivate_after) {
        if (component.k != 0) {
            fail("active component measurement did not remove its last coordinate");
        }
        component.active = false;
    }
}

void execute_component_instruction(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    std::size_t instruction_index) {
    const auto& plan = *program.active_component_plan;
    const auto ref = plan.instruction_steps[instruction_index];
    switch (ref.kind) {
    case ActiveComponentStepKind::IgnoredGlobalPhase:
        return;
    case ActiveComponentStepKind::Rotation: {
        const auto& instruction =
            std::get<ApplyPrecomputedActivePauliRotation>(
                program.instructions[instruction_index]);
        const bool sign =
            eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
        execute_component_rotation(
            runtime,
            plan,
            plan.rotations[ref.payload],
            sign);
        return;
    }
    case ActiveComponentStepKind::Promotion: {
        const auto& instruction =
            std::get<PromoteDormantRotation>(
                program.instructions[instruction_index]);
        const bool sign =
            eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
        execute_component_promotion(
            runtime,
            plan.promotions[ref.payload],
            instruction.kernel_angle,
            sign);
        return;
    }
    case ActiveComponentStepKind::Measurement: {
        const auto& instruction =
            std::get<MeasurePrecomputedActivePauli>(
                program.instructions[instruction_index]);
        execute_component_measurement(
            runtime,
            plan,
            plan.measurements[ref.payload],
            instruction.branch);
        const bool outcome =
            eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
        write_measurement_record(
            runtime,
            instruction.record,
            outcome,
            instruction.record_condition);
        return;
    }
    case ActiveComponentStepKind::None:
        std::visit(
            [&](const auto& inst) { execute_instruction(runtime, inst); },
            program.instructions[instruction_index]);
        return;
    }
    fail("unknown active component instruction kind");
}

void execute_instruction(FactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    const bool sign = eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
    rotate_pauli(runtime, instruction.rotation_kernel, sign);
}

void execute_instruction(FactoredExecutorState& runtime, const PromoteDormantRotation& instruction) {
    const bool sign = eval_symbolic_bool_unchecked(instruction.sign_plan, runtime);
    promote_first_dormant_rotation(runtime, sign ? -instruction.kernel_angle : instruction.kernel_angle);
}

void execute_instruction(FactoredExecutorState& runtime, const RecordMeasurement& instruction) {
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const RecordDetector& instruction) {
    const bool outcome = detector_outcome_from_runtime(runtime, instruction);
    write_detector_record(runtime, instruction.detector, outcome);
}

void execute_instruction(FactoredExecutorState& runtime, const MeasurePrecomputedActivePauli& instruction) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    double prob_true = instruction.kernel.is_diagonal
                           ? active_diagonal_measurement_branch_probability(runtime, instruction.kernel, true)
                           : active_measurement_branch_probability(runtime, instruction.kernel, true);
    prob_true = std::clamp(prob_true, 0.0, 1.0);
    const double prob_false = 1.0 - prob_true;
    const bool branch = sample_bernoulli(runtime.rng_state, prob_true);
    const double probability = branch ? prob_true : prob_false;
    assign_symbol(runtime, instruction.branch, branch);
    if (instruction.kernel.is_diagonal) {
        project_diagonal_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    } else {
        project_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    }
    ++runtime.ndormant;
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction(FactoredExecutorState& runtime, const IntroduceDormantMeasurementBranch& instruction) {
    const bool branch = sample_bernoulli(runtime.rng_state, 0.5);
    assign_symbol(runtime, instruction.branch, branch);
    const bool outcome = eval_symbolic_bool_unchecked(instruction.outcome_plan, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const ApplyPrecomputedActivePauliRotation& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool sign = evaluator.eval(instruction_index, runtime);
    rotate_pauli(runtime, instruction.rotation_kernel, sign);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const PromoteDormantRotation& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool sign = evaluator.eval(instruction_index, runtime);
    promote_first_dormant_rotation(runtime, sign ? -instruction.kernel_angle : instruction.kernel_angle);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const RecordMeasurement& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool outcome = evaluator.eval(instruction_index, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const RecordDetector& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool outcome = !instruction.records.empty() || instruction.outcome.conditions.empty()
                             ? detector_outcome_from_runtime(runtime, instruction)
                             : evaluator.eval(instruction_index, runtime);
    write_detector_record(runtime, instruction.detector, outcome);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const MeasurePrecomputedActivePauli& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    if (runtime.k <= 0) {
        fail("cannot measure an active Pauli when k == 0");
    }
    double prob_true = instruction.kernel.is_diagonal
                           ? active_diagonal_measurement_branch_probability(runtime, instruction.kernel, true)
                           : active_measurement_branch_probability(runtime, instruction.kernel, true);
    prob_true = std::clamp(prob_true, 0.0, 1.0);
    const double prob_false = 1.0 - prob_true;
    const bool branch = sample_bernoulli(runtime.rng_state, prob_true);
    const double probability = branch ? prob_true : prob_false;
    assign_symbol(runtime, instruction.branch, branch);
    if (instruction.kernel.is_diagonal) {
        project_diagonal_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    } else {
        project_active_pauli_measurement(runtime, instruction.kernel, branch, probability);
    }
    ++runtime.ndormant;
    const bool outcome = evaluator.eval(instruction_index, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const IntroduceDormantMeasurementBranch& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool branch = sample_bernoulli(runtime.rng_state, 0.5);
    assign_symbol(runtime, instruction.branch, branch);
    const bool outcome = evaluator.eval(instruction_index, runtime);
    write_measurement_record(runtime, instruction.record, outcome, instruction.record_condition);
}

void execute_instruction_presampled(
    FactoredExecutorState& runtime,
    const FactoredInstruction& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    std::visit(
        [&](const auto& inst) {
            execute_instruction_presampled(
                runtime,
                inst,
                evaluator,
                instruction_index);
        },
        instruction);
}

void execute_component_instruction_presampled(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& plan = *program.active_component_plan;
    const auto ref = plan.instruction_steps[instruction_index];
    switch (ref.kind) {
    case ActiveComponentStepKind::IgnoredGlobalPhase:
        return;
    case ActiveComponentStepKind::Rotation: {
        execute_component_rotation(
            runtime,
            plan,
            plan.rotations[ref.payload],
            evaluator.eval(instruction_index, runtime));
        return;
    }
    case ActiveComponentStepKind::Promotion: {
        const auto& instruction =
            std::get<PromoteDormantRotation>(
                program.instructions[instruction_index]);
        execute_component_promotion(
            runtime,
            plan.promotions[ref.payload],
            instruction.kernel_angle,
            evaluator.eval(instruction_index, runtime));
        return;
    }
    case ActiveComponentStepKind::Measurement: {
        const auto& instruction =
            std::get<MeasurePrecomputedActivePauli>(
                program.instructions[instruction_index]);
        execute_component_measurement(
            runtime,
            plan,
            plan.measurements[ref.payload],
            instruction.branch);
        const bool outcome =
            evaluator.eval(instruction_index, runtime);
        write_measurement_record(
            runtime,
            instruction.record,
            outcome,
            instruction.record_condition);
        return;
    }
    case ActiveComponentStepKind::None:
        execute_instruction_presampled(
            runtime,
            program.instructions[instruction_index],
            evaluator,
            instruction_index);
        return;
    }
    fail("unknown active component instruction kind");
}

template <typename Instruction>
bool execute_instruction_postselected(FactoredExecutorState& runtime, const Instruction& instruction) {
    execute_instruction(runtime, instruction);
    return true;
}

bool execute_instruction_postselected(FactoredExecutorState& runtime, const RecordDetector& instruction) {
    return !detector_outcome_from_runtime(runtime, instruction);
}

template <typename Instruction>
bool execute_instruction_postselected(
    FactoredExecutorState& runtime,
    const Instruction& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    execute_instruction_presampled(runtime, instruction, evaluator, instruction_index);
    return true;
}

bool execute_instruction_postselected(
    FactoredExecutorState& runtime,
    const RecordDetector& instruction,
    const SingleShotExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const bool outcome = !instruction.records.empty() || instruction.outcome.conditions.empty()
                             ? detector_outcome_from_runtime(runtime, instruction)
                             : evaluator.eval(instruction_index, runtime);
    return !outcome;
}

} // namespace

FactoredExecutorState::FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed)
    : n(program.n),
      k(program.initial_k),
      ndormant(program.n - program.initial_k),
      nsymbols(program.nsymbols),
      nrecords(program.nrecords),
      ndetectors(program.ndetectors),
      value_words(symbol_word_count(program.nsymbols), 0),
      assigned_words(symbol_word_count(program.nsymbols), 0),
      measurement_words(symbol_word_count(program.nrecords), 0),
      detector_words(symbol_word_count(program.ndetectors), 0),
      rng_state(seed) {
    configure_single_shot_components(*this, program);
    reset_single_shot_components(*this, program);
    if (!active_components_enabled) {
        const std::size_t dim = active_length(program.initial_k);
        std::fill_n(active_re.data(), dim, 0.0);
        std::fill_n(active_im.data(), dim, 0.0);
        active_re[0] = 1.0;
    }
}

void reset_executor(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    bool clear_detector_records) {
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
    runtime.ndetectors = program.ndetectors;
    reset_single_shot_components(runtime, program);
    if (!runtime.active_components_enabled) {
        ensure_runtime_active_capacity(runtime, program.max_k);
        const std::size_t dim = active_length(program.initial_k);
        std::fill_n(runtime.active_re.data(), dim, 0.0);
        std::fill_n(runtime.active_im.data(), dim, 0.0);
        runtime.active_re[0] = 1.0;
    }
    const std::size_t nwords = symbol_word_count(program.nsymbols);
    if (runtime.value_words.size() != nwords) {
        runtime.value_words.resize(nwords);
    }
    std::fill(runtime.value_words.begin(), runtime.value_words.end(), 0);
    if (runtime.assigned_words.size() != nwords) {
        runtime.assigned_words.resize(nwords);
    }
    std::fill(runtime.assigned_words.begin(), runtime.assigned_words.end(), 0);
    const std::size_t record_words = symbol_word_count(program.nrecords);
    if (runtime.measurement_words.size() != record_words) {
        runtime.measurement_words.resize(record_words);
    }
    std::fill(runtime.measurement_words.begin(), runtime.measurement_words.end(), 0);
    const std::size_t detector_words = symbol_word_count(program.ndetectors);
    if (runtime.detector_words.size() != detector_words) {
        runtime.detector_words.resize(detector_words);
    }
    if (clear_detector_records) {
        std::fill(runtime.detector_words.begin(), runtime.detector_words.end(), 0);
    }
}

void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    sample_exogenous_symbols(runtime, program);
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_component_instruction(runtime, program, idx);
        }
        return;
    }
    for (const auto& instruction : program.instructions) {
        std::visit(
            [&](const auto& inst) { execute_instruction(runtime, inst); },
            instruction);
    }
}

void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    if (program.nsymbols != samples.nsymbols) {
        fail("presampled exogenous table does not match program");
    }
    assign_presampled_exogenous(runtime, samples, shot_index);
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_component_instruction(runtime, program, idx);
        }
        return;
    }
    for (const auto& instruction : program.instructions) {
        std::visit(
            [&](const auto& inst) { execute_instruction(runtime, inst); },
            instruction);
    }
}

void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("single-shot presampled expression plan does not match program");
    }
    if (shot_index < 0 || shot_index >= expression_block.nshots) {
        fail("single-shot presampled expression shot index is out of range");
    }
    SingleShotExpressionEvaluator evaluator{expression_plan, expression_block, shot_index};
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_component_instruction_presampled(
                runtime,
                program,
                evaluator,
                idx);
        }
        return;
    }
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        execute_instruction_presampled(
            runtime,
            program.instructions[idx],
            evaluator,
            idx);
    }
}

void assign_presampled_exogenous_in_place(
    FactoredExecutorState& runtime,
    const PresampledExogenous& samples,
    int shot_index) {
    assign_presampled_exogenous(runtime, samples, shot_index);
}

void execute_instruction_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstruction& instruction) {
    if (runtime.active_components_enabled) {
        fail("direct instruction execution is unavailable for a component executor");
    }
    std::visit([&](const auto& inst) { execute_instruction(runtime, inst); }, instruction);
}

bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    assign_presampled_exogenous(runtime, samples, shot_index);
    if (!runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            const bool survived = std::visit(
                [&](const auto& inst) {
                    return execute_instruction_postselected(runtime, inst);
                },
                program.instructions[idx]);
            if (!survived) {
                return false;
            }
        }
        return true;
    }
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        bool survived = true;
        if (program.active_component_plan->instruction_steps[idx].kind !=
                ActiveComponentStepKind::None) {
            execute_component_instruction(runtime, program, idx);
        } else {
            survived = std::visit(
                [&](const auto& inst) {
                    return execute_instruction_postselected(runtime, inst);
                },
                program.instructions[idx]);
        }
        if (!survived) {
            return false;
        }
    }
    return true;
}

bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int shot_index) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("single-shot presampled expression plan does not match program");
    }
    if (shot_index < 0 || shot_index >= expression_block.nshots) {
        fail("single-shot presampled expression shot index is out of range");
    }
    SingleShotExpressionEvaluator evaluator{expression_plan, expression_block, shot_index};
    if (!runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            const bool survived = std::visit(
                [&](const auto& inst) {
                    return execute_instruction_postselected(
                        runtime,
                        inst,
                        evaluator,
                        idx);
                },
                program.instructions[idx]);
            if (!survived) {
                return false;
            }
        }
        return true;
    }
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        bool survived = true;
        if (program.active_component_plan->instruction_steps[idx].kind !=
                ActiveComponentStepKind::None) {
            execute_component_instruction_presampled(
                runtime,
                program,
                evaluator,
                idx);
        } else {
            survived = std::visit(
                [&](const auto& inst) {
                    return execute_instruction_postselected(
                        runtime,
                        inst,
                        evaluator,
                        idx);
                },
                program.instructions[idx]);
        }
        if (!survived) {
            return false;
        }
    }
    return true;
}

std::vector<std::uint64_t> execute(FactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    execute_in_place(runtime, program);
    return runtime.measurement_words;
}

std::vector<std::uint64_t> sample_measurements(const FactoredInstructionProgram& program, std::uint64_t seed) {
    FactoredExecutorState runtime(program, seed);
    return execute(runtime, program);
}

int default_single_shot_sample_chunk_shots() {
    return kDefaultSingleShotSampleChunkShots;
}

std::vector<std::vector<std::uint64_t>> sample_measurements(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed,
    int sample_chunk_shots) {
    if (shots < 0) {
        fail("shot count must be nonnegative");
    }

    const int chunk_shots = normalize_single_shot_sample_chunk_shots(sample_chunk_shots);
    PackedPresampledExogenous packed_samples;
    prepare_presampled_exogenous_packed(packed_samples, program);
    PresampledExpressionPlan expression_plan;
    prepare_presampled_expression_plan(expression_plan, program, packed_samples);
    PresampledExpressionBlock expression_block;
    FactoredExecutorState runtime(program, seed ^ kSingleShotBranchSeedXor);

    std::vector<std::vector<std::uint64_t>> out;
    out.reserve(static_cast<std::size_t>(shots));

    std::uint64_t exogenous_rng_state = seed;
    for (int offset = 0; offset < shots; offset += chunk_shots) {
        const int chunk = std::min(chunk_shots, shots - offset);
        resample_prepared_exogenous_packed_in_place(
            packed_samples,
            program,
            chunk,
            exogenous_rng_state);
        exogenous_rng_state = packed_samples.next_rng_state;
        evaluate_presampled_expression_block(
            expression_block,
            expression_plan,
            packed_samples);
        for (int shot = 0; shot < chunk; ++shot) {
            reset_executor(runtime, program);
            execute_in_place(runtime, program, expression_plan, expression_block, shot);
            out.push_back(runtime.measurement_words);
        }
    }
    return out;
}

std::vector<std::vector<std::uint64_t>> sample_measurements(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    return sample_measurements(program, shots, seed, 0);
}

} // namespace symft
