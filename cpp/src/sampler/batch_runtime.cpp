#include "batch_internal.hpp"
#include "sampler/component_plan.hpp"
#include "simd/simd.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#endif

namespace symft {

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const ApplyPrecomputedActivePauliRotation& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.sign_plan, runtime);
    rotate_pauli_batch(runtime, instruction.rotation_kernel, runtime.eval_scratch);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const PromoteDormantRotation& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.sign_plan, runtime);
    promote_first_dormant_rotation_batch(runtime, instruction.kernel_angle, runtime.eval_scratch);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const RecordMeasurement& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.outcome_plan, runtime);
    write_batch_measurement_record(runtime, instruction.record, runtime.eval_scratch, instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const RecordDetector& instruction) {
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.outcome_plan, runtime);
    write_batch_detector_record(runtime, instruction.detector, runtime.eval_scratch);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const MeasurePrecomputedActivePauli& instruction) {
    measure_precomputed_active_pauli_batch(
        runtime,
        instruction.kernel,
        instruction.branch,
        instruction.outcome_plan,
        instruction.record,
        instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const IntroduceDormantMeasurementBranch& instruction) {
    fill_batch_random_half_bits(runtime.eval_scratch, runtime);
    const auto& branch_bits = runtime.eval_scratch;
    assign_batch_symbol(runtime, instruction.branch, branch_bits);
    if (write_direct_branch_measurement_record(
            runtime,
            instruction.branch,
            instruction.outcome_plan,
            instruction.record,
            instruction.record_condition)) {
        return;
    }
    eval_symbolic_bool_batch(runtime.eval_scratch, instruction.outcome_plan, runtime);
    write_batch_measurement_record(runtime, instruction.record, runtime.eval_scratch, instruction.record_condition);
}

void execute_batch_instruction(BatchFactoredExecutorState& runtime, const FactoredInstruction& instruction) {
    std::visit([&](const auto& inst) { execute_batch_instruction(runtime, inst); }, instruction);
}

namespace {

inline constexpr int kExpensivePureCompactionDenominator = 64;

bool component_batch_bit_at(
    const std::vector<std::uint64_t>& bits,
    int shot) {
    const std::size_t word = batch_shot_word(shot);
    return word < bits.size() &&
           (bits[word] & batch_shot_mask(shot)) != 0;
}

void configure_batch_components(
    BatchFactoredExecutorState& runtime,
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
        return;
    }
    const auto& plan = *program.active_component_plan;
    if (plan.component_count !=
        static_cast<int>(plan.component_max_k.size())) {
        fail("active component plan has inconsistent component capacities");
    }
    runtime.active_components.resize(
        static_cast<std::size_t>(plan.component_count));
    const std::size_t pitch =
        static_cast<std::size_t>(runtime.active_pitch);
    for (int component = 0; component < plan.component_count; ++component) {
        auto& buffer =
            runtime.active_components[static_cast<std::size_t>(component)];
        const std::size_t capacity = active_length(
            plan.component_max_k[static_cast<std::size_t>(component)]);
        buffer.stride = capacity;
        const std::size_t storage = capacity * pitch;
        buffer.re.resize(storage, 0.0);
        buffer.im.resize(storage, 0.0);
        buffer.scratch_re.resize(storage, 0.0);
        buffer.scratch_im.resize(storage, 0.0);
    }
    std::vector<double>().swap(runtime.active_re);
    std::vector<double>().swap(runtime.active_im);
    std::vector<double>().swap(runtime.scratch_re);
    std::vector<double>().swap(runtime.scratch_im);
}

void reset_batch_components(
    BatchFactoredExecutorState& runtime,
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
        configure_batch_components(runtime, program);
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
    const std::size_t pitch =
        static_cast<std::size_t>(runtime.active_pitch);
    for (int component_id = 0;
         component_id < plan.initial_components;
         ++component_id) {
        auto& component =
            runtime.active_components[
                static_cast<std::size_t>(component_id)];
        if (component.stride < 2) {
            fail("initial active component has insufficient storage");
        }
        component.k = 1;
        component.active = true;
        if (runtime.dense_shot_major_active) {
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const std::size_t base =
                    static_cast<std::size_t>(shot) * component.stride;
                component.re[base] = 1.0;
                component.im[base] = 0.0;
                component.re[base + 1] = 0.0;
                component.im[base + 1] = 0.0;
            }
        } else {
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const std::size_t lane = static_cast<std::size_t>(shot);
                component.re[lane] = 1.0;
                component.im[lane] = 0.0;
                component.re[pitch + lane] = 0.0;
                component.im[pitch + lane] = 0.0;
            }
        }
    }
}

void ensure_dense_batch_active_storage(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program) {
    runtime.active_components_enabled = false;
    runtime.active_components.clear();
    const std::size_t max_dim = active_length(program.max_k);
    runtime.active_stride = max_dim;
    const std::size_t active_size =
        max_dim * static_cast<std::size_t>(runtime.active_pitch);
    runtime.active_re.resize(active_size, 0.0);
    runtime.active_im.resize(active_size, 0.0);
    runtime.scratch_re.resize(active_size, 0.0);
    runtime.scratch_im.resize(active_size, 0.0);
}

void initialize_dense_batch_active(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program) {
    const std::size_t dim = active_length(program.initial_k);
    if (runtime.dense_shot_major_active) {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t base =
                static_cast<std::size_t>(shot) * runtime.active_stride;
            std::fill_n(runtime.active_re.data() + base, dim, 0.0);
            std::fill_n(runtime.active_im.data() + base, dim, 0.0);
            runtime.active_re[base] = 1.0;
        }
        return;
    }
    const std::size_t pitch =
        static_cast<std::size_t>(runtime.active_pitch);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * pitch;
        std::fill_n(runtime.active_re.data() + base, pitch, 0.0);
        std::fill_n(runtime.active_im.data() + base, pitch, 0.0);
    }
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        runtime.active_re[static_cast<std::size_t>(shot)] = 1.0;
    }
}

void merge_batch_components(
    BatchFactoredExecutorState& runtime,
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
    const std::size_t pitch =
        static_cast<std::size_t>(runtime.active_pitch);
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
        if (target.stride < merged_dim) {
            fail("active component merge target has insufficient storage");
        }
        if (runtime.dense_shot_major_active) {
            for (int shot = 0; shot < runtime.active_shots; ++shot) {
                const std::size_t target_shot =
                    static_cast<std::size_t>(shot) * target.stride;
                const std::size_t source_shot =
                    static_cast<std::size_t>(shot) * source.stride;
                for (std::size_t source_basis = source_dim;
                     source_basis-- > 0;) {
                    const double source_re =
                        source.re[source_shot + source_basis];
                    const double source_im =
                        source.im[source_shot + source_basis];
                    const std::size_t output_base =
                        target_shot + (source_basis << target.k);
                    SYMFT_SINGLE_SIMD_LOOP
                    for (std::size_t target_basis = 0;
                         target_basis < target_dim;
                         ++target_basis) {
                        const double target_re =
                            target.re[target_shot + target_basis];
                        const double target_im =
                            target.im[target_shot + target_basis];
                        const std::size_t output =
                            output_base + target_basis;
                        target.re[output] =
                            target_re * source_re -
                            target_im * source_im;
                        target.im[output] =
                            target_re * source_im +
                            target_im * source_re;
                    }
                }
            }
        } else {
            for (std::size_t source_basis = source_dim;
                 source_basis-- > 0;) {
                const std::size_t source_base = source_basis * pitch;
                const std::size_t output_basis_base =
                    source_basis << target.k;
                for (std::size_t target_basis = 0;
                     target_basis < target_dim;
                     ++target_basis) {
                    const std::size_t target_base =
                        target_basis * pitch;
                    const std::size_t output_base =
                        (output_basis_base + target_basis) * pitch;
                    SYMFT_BATCH_SIMD_LOOP
                    for (int shot = 0;
                         shot < runtime.active_shots;
                         ++shot) {
                        const std::size_t lane =
                            static_cast<std::size_t>(shot);
                        const double target_re =
                            target.re[target_base + lane];
                        const double target_im =
                            target.im[target_base + lane];
                        const double source_re =
                            source.re[source_base + lane];
                        const double source_im =
                            source.im[source_base + lane];
                        target.re[output_base + lane] =
                            target_re * source_re -
                            target_im * source_im;
                        target.im[output_base + lane] =
                            target_re * source_im +
                            target_im * source_re;
                    }
                }
            }
        }
        target.k = merged_k;
        source.k = 0;
        source.active = false;
    }
}

class ScopedBatchComponent {
  public:
    ScopedBatchComponent(
        BatchFactoredExecutorState& runtime_,
        BatchActiveComponent& component_)
        : runtime(runtime_),
          component(component_),
          global_k(runtime_.k),
          global_ndormant(runtime_.ndormant),
          global_stride(runtime_.active_stride) {
        if (!component.active) {
            fail("cannot execute an instruction on an inactive component");
        }
        std::swap(runtime.active_re, component.re);
        std::swap(runtime.active_im, component.im);
        std::swap(runtime.scratch_re, component.scratch_re);
        std::swap(runtime.scratch_im, component.scratch_im);
        runtime.k = component.k;
        runtime.active_stride = component.stride;
    }

    ~ScopedBatchComponent() {
        component.k = runtime.k;
        component.stride = runtime.active_stride;
        std::swap(runtime.active_re, component.re);
        std::swap(runtime.active_im, component.im);
        std::swap(runtime.scratch_re, component.scratch_re);
        std::swap(runtime.scratch_im, component.scratch_im);
        runtime.k = global_k;
        runtime.ndormant = global_ndormant;
        runtime.active_stride = global_stride;
    }

  private:
    BatchFactoredExecutorState& runtime;
    BatchActiveComponent& component;
    int global_k;
    int global_ndormant;
    std::size_t global_stride;
};

void execute_batch_component_rotation(
    BatchFactoredExecutorState& runtime,
    const ActiveComponentPlan& plan,
    const ActiveComponentRotationStep& step,
    const std::vector<std::uint64_t>& sign_bits) {
    merge_batch_components(
        runtime,
        plan,
        step.merge_offset,
        step.merge_count,
        step.component);
    auto& component =
        runtime.active_components[static_cast<std::size_t>(step.component)];
    ScopedBatchComponent scope(runtime, component);
    rotate_pauli_batch(runtime, step.kernel, sign_bits);
}

void execute_batch_component_promotion(
    BatchFactoredExecutorState& runtime,
    const ActiveComponentPromotionStep& step,
    double kernel_angle,
    const std::vector<std::uint64_t>& sign_bits) {
    if (runtime.ndormant <= 0 ||
        step.component < 0 ||
        step.component >= static_cast<int>(runtime.active_components.size())) {
        fail("active component promotion is invalid");
    }
    auto& component =
        runtime.active_components[static_cast<std::size_t>(step.component)];
    if (component.active || component.stride < 2) {
        fail("active component promotion target is unavailable");
    }
    const double c = std::cos(kernel_angle);
    const double s = std::sin(kernel_angle);
    component.k = 1;
    component.active = true;
    const std::size_t pitch =
        static_cast<std::size_t>(runtime.active_pitch);
    if (runtime.dense_shot_major_active) {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t base =
                static_cast<std::size_t>(shot) * component.stride;
            component.re[base] = c;
            component.im[base] = 0.0;
            component.re[base + 1] = 0.0;
            component.im[base + 1] =
                component_batch_bit_at(sign_bits, shot) ? s : -s;
        }
    } else {
        for (int shot = 0; shot < runtime.active_shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            component.re[lane] = c;
            component.im[lane] = 0.0;
            component.re[pitch + lane] = 0.0;
            component.im[pitch + lane] =
                component_batch_bit_at(sign_bits, shot) ? s : -s;
        }
    }
    ++runtime.k;
    --runtime.ndormant;
}

void execute_batch_component_measurement(
    BatchFactoredExecutorState& runtime,
    const ActiveComponentPlan& plan,
    const ActiveComponentMeasurementStep& step,
    int branch_condition) {
    merge_batch_components(
        runtime,
        plan,
        step.merge_offset,
        step.merge_count,
        step.component);
    auto& component =
        runtime.active_components[static_cast<std::size_t>(step.component)];
    {
        ScopedBatchComponent scope(runtime, component);
        measure_precomputed_active_pauli_branch_batch(
            runtime,
            step.kernel,
            branch_condition);
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

void execute_batch_component_instruction(
    BatchFactoredExecutorState& runtime,
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
        eval_symbolic_bool_batch(
            runtime.eval_scratch,
            instruction.sign_plan,
            runtime);
        execute_batch_component_rotation(
            runtime,
            plan,
            plan.rotations[ref.payload],
            runtime.eval_scratch);
        return;
    }
    case ActiveComponentStepKind::Promotion: {
        const auto& instruction =
            std::get<PromoteDormantRotation>(
                program.instructions[instruction_index]);
        eval_symbolic_bool_batch(
            runtime.eval_scratch,
            instruction.sign_plan,
            runtime);
        execute_batch_component_promotion(
            runtime,
            plan.promotions[ref.payload],
            instruction.kernel_angle,
            runtime.eval_scratch);
        return;
    }
    case ActiveComponentStepKind::Measurement: {
        const auto& instruction =
            std::get<MeasurePrecomputedActivePauli>(
                program.instructions[instruction_index]);
        execute_batch_component_measurement(
            runtime,
            plan,
            plan.measurements[ref.payload],
            instruction.branch);
        if (write_direct_branch_measurement_record(
                runtime,
                instruction.branch,
                instruction.outcome_plan,
                instruction.record,
                instruction.record_condition)) {
            return;
        }
        eval_symbolic_bool_batch(
            runtime.eval_scratch,
            instruction.outcome_plan,
            runtime);
        write_batch_measurement_record(
            runtime,
            instruction.record,
            runtime.eval_scratch,
            instruction.record_condition);
        return;
    }
    case ActiveComponentStepKind::None:
        execute_batch_instruction(
            runtime,
            program.instructions[instruction_index]);
        return;
    }
    fail("unknown active component instruction kind");
}

std::uint64_t live_word_mask_for_shots(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

void ensure_word_scratch(std::vector<std::uint64_t>& words, std::size_t nwords) {
    if (words.size() < nwords) {
        words.resize(nwords, 0);
    }
}

void mark_condition_uses(
    std::vector<int>& last_use,
    const SymbolicBoolEvaluationPlan& plan,
    int instruction_index) {
    for (int condition : plan.conditions) {
        if (condition > 0 && condition < static_cast<int>(last_use.size())) {
            last_use[static_cast<std::size_t>(condition)] =
                std::max(last_use[static_cast<std::size_t>(condition)], instruction_index);
        }
    }
}

std::vector<int> condition_last_uses(const FactoredInstructionProgram& program) {
    std::vector<int> last_use(static_cast<std::size_t>(program.nsymbols + 1), -1);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const int instruction_index = static_cast<int>(idx);
        std::visit(
            [&](const auto& inst) {
                using T = std::decay_t<decltype(inst)>;
                if constexpr (
                    std::is_same_v<T, ApplyPrecomputedActivePauliRotation> ||
                    std::is_same_v<T, PromoteDormantRotation>) {
                    mark_condition_uses(last_use, inst.sign_plan, instruction_index);
                } else if constexpr (std::is_same_v<T, RecordDetector>) {
                    if (inst.records.empty() && !inst.outcome_plan.conditions.empty()) {
                        mark_condition_uses(last_use, inst.outcome_plan, instruction_index);
                    }
                } else if constexpr (
                    std::is_same_v<T, RecordMeasurement> ||
                    std::is_same_v<T, MeasurePrecomputedActivePauli> ||
                    std::is_same_v<T, IntroduceDormantMeasurementBranch>) {
                    mark_condition_uses(last_use, inst.outcome_plan, instruction_index);
                }
            },
            program.instructions[idx]);
    }
    return last_use;
}

std::vector<int> measurement_record_last_uses(
    const FactoredInstructionProgram& program,
    const std::vector<std::vector<int>>& retained_record_uses,
    int final_instruction_index) {
    const int nrecords = program.nrecords;
    std::vector<int> last_use(static_cast<std::size_t>(nrecords + 1), -1);
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        const auto* detector = std::get_if<RecordDetector>(&program.instructions[idx]);
        if (detector == nullptr) {
            continue;
        }
        for (int record : detector->records) {
            if (record <= 0 || record > nrecords) {
                fail("detector references an out-of-range measurement record");
            }
            last_use[static_cast<std::size_t>(record)] =
                std::max(last_use[static_cast<std::size_t>(record)], static_cast<int>(idx));
        }
    }
    for (const auto& records : retained_record_uses) {
        for (int record : records) {
            if (record <= 0 || record > nrecords) {
                fail("retained record use references an out-of-range measurement record");
            }
            last_use[static_cast<std::size_t>(record)] =
                std::max(last_use[static_cast<std::size_t>(record)], final_instruction_index);
        }
    }
    return last_use;
}

const std::vector<std::vector<int>>& empty_retained_record_uses() {
    static const std::vector<std::vector<int>> empty;
    return empty;
}

void refresh_batch_postselection_metadata(
    BatchDetectorPostselectionScratch& scratch,
    const FactoredInstructionProgram& program,
    const BatchDetectorPostselectionOptions& options) {
    if (scratch.postselection_metadata_program == &program &&
        scratch.postselection_metadata_retained_record_uses == options.retained_record_uses) {
        return;
    }
    const auto& retained_record_uses = options.retained_record_uses == nullptr
                                           ? empty_retained_record_uses()
                                           : *options.retained_record_uses;
    scratch.condition_last_use_by_index = condition_last_uses(program);
    scratch.record_last_use_by_index = measurement_record_last_uses(
        program,
        retained_record_uses,
        static_cast<int>(program.instructions.size()));
    scratch.postselection_metadata_program = &program;
    scratch.postselection_metadata_retained_record_uses = options.retained_record_uses;
}

int count_live_bits(const std::vector<std::uint64_t>& bits, int shots) {
    int count = 0;
    const std::size_t nwords = batch_word_count(shots);
    for (std::size_t word = 0; word < nwords; ++word) {
        count += detail::popcount64(bits[word] & live_word_mask_for_shots(shots, word));
    }
    return count;
}

template <typename Fn>
void for_each_live_assigned_condition(
    const std::vector<std::uint64_t>& assigned_words,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    Fn&& fn) {
    for (std::size_t word = 0; word < assigned_words.size(); ++word) {
        std::uint64_t bits = assigned_words[word];
        while (bits != 0) {
            const int bit = trailing_zeros64(bits);
            const int condition = static_cast<int>(word << 6) + bit + 1;
            if (condition < static_cast<int>(condition_last_use.size()) &&
                condition_last_use[static_cast<std::size_t>(condition)] > instruction_index) {
                fn(condition);
            }
            bits &= bits - 1;
        }
    }
}

bool instruction_is_pure_over_dead(const ApplyPrecomputedActivePauliRotation&) {
    return true;
}

bool instruction_is_pure_over_dead(const PromoteDormantRotation&) {
    return true;
}

bool instruction_is_pure_over_dead(const RecordMeasurement&) {
    return true;
}

bool instruction_is_pure_over_dead(const RecordDetector&) {
    return true;
}

bool instruction_is_pure_over_dead(const MeasurePrecomputedActivePauli&) {
    return false;
}

bool instruction_is_pure_over_dead(const IntroduceDormantMeasurementBranch&) {
    return false;
}

bool instruction_is_pure_over_dead(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return instruction_is_pure_over_dead(inst); }, instruction);
}

bool instruction_is_expensive_over_dead(const ApplyPrecomputedActivePauliRotation&) {
    return true;
}

bool instruction_is_expensive_over_dead(const PromoteDormantRotation&) {
    return true;
}

bool instruction_is_expensive_over_dead(const RecordMeasurement&) {
    return false;
}

bool instruction_is_expensive_over_dead(const RecordDetector&) {
    return false;
}

bool instruction_is_expensive_over_dead(const MeasurePrecomputedActivePauli&) {
    return true;
}

bool instruction_is_expensive_over_dead(const IntroduceDormantMeasurementBranch&) {
    return true;
}

bool instruction_is_expensive_over_dead(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return instruction_is_expensive_over_dead(inst); }, instruction);
}

bool should_compact_dead_before_instruction(
    const BatchFactoredExecutorState& runtime,
    const BatchDetectorPostselectionScratch& scratch,
    const BatchDetectorPostselectionOptions& options,
    const FactoredInstruction& instruction) {
    if (runtime.active_shots == 0) {
        return false;
    }
    if (scratch.dead_count == 0) {
        return false;
    }
    if (!instruction_is_pure_over_dead(instruction)) {
        return true;
    }
    if (runtime.dense_shot_major_active &&
        !runtime.active_components_enabled) {
        return false;
    }
    int denominator = std::max(1, options.mask_dead_shots_min_fraction_denominator);
    if (instruction_is_expensive_over_dead(instruction)) {
        denominator = std::max(denominator, kExpensivePureCompactionDenominator);
    }
    return scratch.dead_count * denominator >= runtime.active_shots;
}

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386))
__attribute__((target("bmi2")))
std::uint64_t compress_bits_bmi2(std::uint64_t bits, std::uint64_t keep_mask) {
    return _pext_u64(bits, keep_mask);
}
#endif

std::uint64_t compress_bits_portable(std::uint64_t bits, std::uint64_t keep_mask) {
    std::uint64_t out = 0;
    std::uint64_t dest = 1;
    while (keep_mask != 0) {
        const std::uint64_t bit = keep_mask & (~keep_mask + 1);
        if ((bits & bit) != 0) {
            out |= dest;
        }
        keep_mask &= keep_mask - 1;
        dest <<= 1;
    }
    return out;
}

std::uint64_t compress_bits(std::uint64_t bits, std::uint64_t keep_mask) {
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386))
    static const bool has_bmi2 = __builtin_cpu_supports("bmi2");
    if (has_bmi2) {
        return compress_bits_bmi2(bits, keep_mask);
    }
#endif
    return compress_bits_portable(bits, keep_mask);
}

void append_compressed_bits(
    std::vector<std::uint64_t>& out,
    int& dest_bit,
    std::uint64_t compressed,
    int nbits) {
    if (nbits == 0) {
        return;
    }
    const std::size_t word = static_cast<std::size_t>(dest_bit >> 6);
    const int shift = dest_bit & 63;
    out[word] |= compressed << shift;
    if (shift != 0 && nbits > 64 - shift) {
        out[word + 1] |= compressed >> (64 - shift);
    }
    dest_bit += nbits;
}

void compact_live_measurement_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_use,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    if (stride_words == 0 || old_shots == 0 || record_last_use.size() <= 1) {
        return;
    }
    const auto record_is_live = [&](int record) {
        const int last_use = record_last_use[static_cast<std::size_t>(record)];
        return include_current_use ? last_use >= instruction_index : last_use > instruction_index;
    };
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (int record = 1; record < static_cast<int>(record_last_use.size()); ++record) {
            if (!record_is_live(record)) {
                continue;
            }
            const std::size_t base = static_cast<std::size_t>(record - 1) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for (int record = 1; record < static_cast<int>(record_last_use.size()); ++record) {
        if (!record_is_live(record)) {
            continue;
        }
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(record - 1) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal measurement compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void compact_live_detector_columns(
    std::vector<std::uint64_t>& columns,
    int ndetectors,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    if (stride_words == 0 || old_shots == 0 || ndetectors <= 0) {
        return;
    }
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (int detector = 1; detector <= ndetectors; ++detector) {
            const std::size_t base = static_cast<std::size_t>(detector - 1) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for (int detector = 1; detector <= ndetectors; ++detector) {
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(detector - 1) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal detector compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void compact_live_symbol_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<std::uint64_t>& assigned_words,
    const std::vector<int>& condition_last_use,
    int instruction_index,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    if (stride_words == 0 || old_shots == 0 || condition_last_use.size() <= 1) {
        return;
    }
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for_each_live_assigned_condition(assigned_words, condition_last_use, instruction_index, [&](int condition) {
            const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        });
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for_each_live_assigned_condition(assigned_words, condition_last_use, instruction_index, [&](int condition) {
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = static_cast<std::size_t>(condition - 1) * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal symbol compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    });
}

void compact_live_expression_columns(
    std::vector<std::uint64_t>& columns,
    const std::vector<int>& expression_last_use,
    int instruction_index,
    std::size_t stride_words,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits,
    std::vector<std::uint64_t>& scratch) {
    const std::size_t expression_count = expression_last_use.size();
    if (stride_words == 0 || old_shots == 0 || expression_count == 0) {
        return;
    }
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (std::size_t expression = 0; expression < expression_count; ++expression) {
            if (expression_last_use[expression] <= instruction_index) {
                continue;
            }
            const std::size_t base = expression * stride_words;
            columns[base] = compress_bits(columns[base], keep_mask);
            for (std::size_t word = 1; word < stride_words; ++word) {
                columns[base + word] = 0;
            }
        }
        return;
    }
    ensure_word_scratch(scratch, stride_words);
    const std::size_t nwords = batch_word_count(old_shots);
    for (std::size_t expression = 0; expression < expression_count; ++expression) {
        if (expression_last_use[expression] <= instruction_index) {
            continue;
        }
        std::fill(scratch.begin(), scratch.begin() + static_cast<std::ptrdiff_t>(stride_words), 0);
        const std::size_t base = expression * stride_words;
        int dest_bit = 0;
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t keep_mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            if (keep_mask == 0) {
                continue;
            }
            const int kept = detail::popcount64(keep_mask);
            append_compressed_bits(
                scratch,
                dest_bit,
                compress_bits(columns[base + word], keep_mask),
                kept);
        }
        if (dest_bit != survivor_count) {
            fail("internal expression compaction count mismatch");
        }
        std::copy_n(scratch.data(), stride_words, columns.data() + base);
    }
}

void collect_live_sources(
    std::vector<int>& live_sources,
    int old_shots,
    int survivor_count,
    const std::vector<std::uint64_t>& keep_bits) {
    live_sources.resize(static_cast<std::size_t>(survivor_count));
    int dest = 0;
    if (old_shots <= 64) {
        const std::uint64_t keep_mask = keep_bits.empty() ? 0 : keep_bits[0] & live_word_mask_for_shots(old_shots, 0);
        for (int src = 0; src < old_shots; ++src) {
            if ((keep_mask & (std::uint64_t{1} << src)) != 0) {
                live_sources[static_cast<std::size_t>(dest++)] = src;
            }
        }
    } else {
        const std::size_t nwords = batch_word_count(old_shots);
        for (std::size_t word = 0; word < nwords; ++word) {
            std::uint64_t mask = keep_bits[word] & live_word_mask_for_shots(old_shots, word);
            while (mask != 0) {
                const int bit = trailing_zeros64(mask);
                live_sources[static_cast<std::size_t>(dest++)] = static_cast<int>(word << 6) + bit;
                mask &= mask - 1;
            }
        }
    }
    if (dest != survivor_count) {
        fail("internal live source compaction count mismatch");
    }
}

void compact_active_columns(
    BatchFactoredExecutorState& runtime,
    int survivor_count,
    const std::vector<int>& live_sources) {
    int first_moved = 0;
    while (first_moved < survivor_count &&
           live_sources[static_cast<std::size_t>(first_moved)] == first_moved) {
        ++first_moved;
    }
    if (first_moved == survivor_count) {
        return;
    }

    if (runtime.active_components_enabled) {
        const std::size_t pitch =
            static_cast<std::size_t>(runtime.active_pitch);
        for (auto& component : runtime.active_components) {
            if (!component.active) {
                continue;
            }
            const std::size_t dim = active_length(component.k);
            if (runtime.dense_shot_major_active) {
                for (int dst = first_moved; dst < survivor_count; ++dst) {
                    const int src =
                        live_sources[static_cast<std::size_t>(dst)];
                    const std::size_t src_base =
                        static_cast<std::size_t>(src) * component.stride;
                    const std::size_t dst_base =
                        static_cast<std::size_t>(dst) * component.stride;
                    std::copy_n(
                        component.re.data() + src_base,
                        dim,
                        component.re.data() + dst_base);
                    std::copy_n(
                        component.im.data() + src_base,
                        dim,
                        component.im.data() + dst_base);
                }
            } else {
                for (std::size_t basis = 0; basis < dim; ++basis) {
                    double* row_re =
                        component.re.data() + basis * pitch;
                    double* row_im =
                        component.im.data() + basis * pitch;
                    for (int dst = first_moved;
                         dst < survivor_count;
                         ++dst) {
                        const int src =
                            live_sources[static_cast<std::size_t>(dst)];
                        row_re[dst] = row_re[src];
                        row_im[dst] = row_im[src];
                    }
                }
            }
        }
        return;
    }

    const std::size_t dim = active_length(runtime.k);
    if (runtime.dense_shot_major_active) {
        const std::size_t stride = runtime.active_stride;
        for (int dst = first_moved; dst < survivor_count; ++dst) {
            const int src = live_sources[static_cast<std::size_t>(dst)];
            const std::size_t src_base = static_cast<std::size_t>(src) * stride;
            const std::size_t dst_base = static_cast<std::size_t>(dst) * stride;
            std::copy_n(runtime.active_re.data() + src_base, dim, runtime.active_re.data() + dst_base);
            std::copy_n(runtime.active_im.data() + src_base, dim, runtime.active_im.data() + dst_base);
        }
        return;
    }

    const std::size_t pitch = static_cast<std::size_t>(runtime.active_pitch);
    for (std::size_t basis = 0; basis < dim; ++basis) {
        double* row_re = runtime.active_re.data() + basis * pitch;
        double* row_im = runtime.active_im.data() + basis * pitch;
        for (int dst = first_moved; dst < survivor_count; ++dst) {
            const int src = live_sources[static_cast<std::size_t>(dst)];
            row_re[dst] = row_re[src];
            row_im[dst] = row_im[src];
        }
    }
}

void compact_surviving_shots(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& keep_bits,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    std::vector<std::uint64_t>& scratch,
    std::vector<int>& live_sources,
    std::vector<std::uint64_t>* expression_words,
    const std::vector<int>* expression_last_use,
    bool compact_detector_records) {
    const int old_shots = runtime.active_shots;
    const int survivor_count = count_live_bits(keep_bits, old_shots);
    if (survivor_count == old_shots) {
        return;
    }
    if (survivor_count == 0) {
        runtime.active_shots = 0;
        return;
    }
    collect_live_sources(live_sources, old_shots, survivor_count, keep_bits);
    compact_active_columns(runtime, survivor_count, live_sources);
    if (expression_words != nullptr) {
        if (expression_last_use == nullptr) {
            fail("internal expression compaction last-use table is missing");
        }
        compact_live_expression_columns(
            *expression_words,
            *expression_last_use,
            instruction_index,
            runtime.batch_words,
            old_shots,
            survivor_count,
            keep_bits,
            scratch);
    }
    compact_live_symbol_columns(
        runtime.value_words,
        runtime.assigned_words,
        condition_last_use,
        instruction_index,
        runtime.batch_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch);
    compact_live_measurement_columns(
        runtime.measurement_words,
        record_last_use,
        instruction_index,
        include_current_record_use,
        runtime.batch_words,
        old_shots,
        survivor_count,
        keep_bits,
        scratch);
    if (compact_detector_records) {
        compact_live_detector_columns(
            runtime.detector_words,
            runtime.ndetectors,
            runtime.batch_words,
            old_shots,
            survivor_count,
            keep_bits,
            scratch);
    }
    runtime.active_shots = survivor_count;
}

void compact_dead_shots_if_needed(
    BatchFactoredExecutorState& runtime,
    BatchDetectorPostselectionScratch& scratch,
    const std::vector<int>& condition_last_use,
    const std::vector<int>& record_last_use,
    int instruction_index,
    bool include_current_record_use,
    std::vector<std::uint64_t>* expression_words = nullptr,
    const std::vector<int>* expression_last_use = nullptr,
    bool compact_detector_records = false) {
    if (runtime.active_shots == 0) {
        return;
    }
    const int dead_count = scratch.dead_count;
    if (dead_count == 0) {
        return;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    for (std::size_t word = 0; word < nwords; ++word) {
        scratch.keep_bits[word] = (~scratch.dead_bits[word]) & live_word_mask_for_shots(runtime.active_shots, word);
    }
    std::fill(
        scratch.keep_bits.begin() + static_cast<std::ptrdiff_t>(nwords),
        scratch.keep_bits.end(),
        0);
    compact_surviving_shots(
        runtime,
        scratch.keep_bits,
        condition_last_use,
        record_last_use,
        instruction_index,
        include_current_record_use,
        scratch.compact_scratch,
        scratch.live_sources,
        expression_words,
        expression_last_use,
        compact_detector_records);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    scratch.dead_count = 0;
}

int mark_dead_from_detector_bits(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::uint64_t>& detector_bits,
    BatchDetectorPostselectionScratch& scratch) {
    if (runtime.active_shots == 0) {
        return 0;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    int discarded_now = 0;
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask_for_shots(runtime.active_shots, word);
        const std::uint64_t fired = detector_bits[word] & live;
        const std::uint64_t newly_dead = fired & ~scratch.dead_bits[word];
        scratch.dead_bits[word] |= fired;
        discarded_now += detail::popcount64(newly_dead);
    }
    if (discarded_now != 0) {
        scratch.dead_count += discarded_now;
    }
    return discarded_now;
}

int mark_dead_from_constant_detector(
    BatchFactoredExecutorState& runtime,
    bool fired,
    BatchDetectorPostselectionScratch& scratch) {
    if (!fired || runtime.active_shots == 0) {
        return 0;
    }
    const std::size_t nwords = batch_word_count(runtime.active_shots);
    int discarded_now = 0;
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask_for_shots(runtime.active_shots, word);
        const std::uint64_t newly_dead = live & ~scratch.dead_bits[word];
        scratch.dead_bits[word] |= live;
        discarded_now += detail::popcount64(newly_dead);
    }
    if (discarded_now != 0) {
        scratch.dead_count += discarded_now;
    }
    return discarded_now;
}

int mark_dead_from_detector_records(
    BatchFactoredExecutorState& runtime,
    const RecordDetector& instruction,
    BatchDetectorPostselectionScratch& scratch) {
    if (runtime.active_shots == 0 || instruction.records.empty()) {
        return 0;
    }
    for (int record : instruction.records) {
        if (record <= 0 || record > runtime.nrecords) {
            fail("detector references an out-of-range measurement record");
        }
    }

    const std::size_t nwords = batch_word_count(runtime.active_shots);
    int discarded_now = 0;
    if (instruction.records.size() == 1) {
        const std::size_t base = batch_record_offset(runtime, instruction.records.front(), 0);
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t live = live_word_mask_for_shots(runtime.active_shots, word);
            const std::uint64_t fired = runtime.measurement_words[base + word] & live;
            const std::uint64_t newly_dead = fired & ~scratch.dead_bits[word];
            scratch.dead_bits[word] |= fired;
            discarded_now += detail::popcount64(newly_dead);
        }
    } else {
        for (std::size_t word = 0; word < nwords; ++word) {
            std::uint64_t fired = 0;
            for (int record : instruction.records) {
                fired ^= runtime.measurement_words[batch_record_offset(runtime, record, word)];
            }
            fired &= live_word_mask_for_shots(runtime.active_shots, word);
            const std::uint64_t newly_dead = fired & ~scratch.dead_bits[word];
            scratch.dead_bits[word] |= fired;
            discarded_now += detail::popcount64(newly_dead);
        }
    }
    if (discarded_now != 0) {
        scratch.dead_count += discarded_now;
    }
    return discarded_now;
}

const std::vector<std::uint64_t>& detector_record_outcome_bits(
    BatchFactoredExecutorState& runtime,
    const RecordDetector& instruction,
    std::vector<std::uint64_t>& out) {
    if (instruction.records.empty()) {
        eval_symbolic_bool_batch(out, instruction.outcome_plan, runtime);
        return out;
    }
    ensure_word_scratch(out, runtime.batch_words);
    const std::size_t nwords = runtime_batch_word_count(runtime);
    std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(nwords), 0);
    for (int record : instruction.records) {
        if (record <= 0 || record > runtime.nrecords) {
            fail("detector references an out-of-range measurement record");
        }
        const std::size_t base = batch_record_offset(runtime, record, 0);
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= runtime.measurement_words[base + word];
        }
    }
    std::fill(out.begin() + static_cast<std::ptrdiff_t>(nwords), out.end(), 0);
    return out;
}

std::uint64_t expression_slice_word(
    const PresampledExpressionBlock& block,
    int block_expression_index,
    int first_sample_shot,
    int active_shots,
    std::size_t dest_word) {
    const int bit_offset = first_sample_shot & 63;
    const std::size_t src_word = static_cast<std::size_t>(first_sample_shot >> 6) + dest_word;
    const std::size_t expression_index = static_cast<std::size_t>(block_expression_index);
    std::uint64_t out = 0;
    if (src_word < block.shot_words) {
        out = block.expression_words[presampled_expression_block_offset(block, expression_index, src_word)] >> bit_offset;
    }
    if (bit_offset != 0 && src_word + 1 < block.shot_words) {
        out |= block.expression_words[presampled_expression_block_offset(block, expression_index, src_word + 1)]
               << (64 - bit_offset);
    }
    return out & live_word_mask_for_shots(active_shots, dest_word);
}

struct BatchExpressionEvaluator {
    const PresampledExpressionPlan& expression_plan;
    const std::vector<std::uint64_t>* expression_words = nullptr;
    const PresampledExpressionBlock* expression_block = nullptr;
    std::size_t expression_stride_words = 0;
    int first_sample_shot = 0;
    std::vector<std::uint64_t>& out;

    const std::vector<std::uint64_t>& eval(
        std::size_t instruction_index,
        const BatchFactoredExecutorState& runtime) const {
        if (instruction_index >= expression_plan.instruction_expressions.size()) {
            fail("batch presampled expression plan does not match program");
        }
        const auto& expression = expression_plan.instruction_expressions[instruction_index];
        if (expression.block_expression_index < 0 ||
            expression.block_expression_index >= static_cast<int>(expression_plan.block_expressions.size())) {
            fail("batch presampled expression references an out-of-range block expression");
        }
        const int block_expression_index = expression.block_expression_index;
        const std::size_t nwords = runtime_batch_word_count(runtime);
        if (out.size() < runtime.batch_words) {
            out.resize(runtime.batch_words, 0);
        }
        if (expression_block != nullptr) {
            if (nwords == 1 && (first_sample_shot & 63) == 0) {
                const std::size_t src_word = static_cast<std::size_t>(first_sample_shot >> 6);
                const std::size_t expression_index = static_cast<std::size_t>(block_expression_index);
                out[0] = src_word < expression_block->shot_words
                             ? expression_block->expression_words[presampled_expression_block_offset(
                                   *expression_block,
                                   expression_index,
                                   src_word)] &
                                   live_word_mask_for_shots(runtime.active_shots, 0)
                             : 0;
            } else {
                for (std::size_t word = 0; word < nwords; ++word) {
                    out[word] = expression_slice_word(
                        *expression_block,
                        block_expression_index,
                        first_sample_shot,
                        runtime.active_shots,
                        word);
                }
            }
        } else {
            const std::size_t base = static_cast<std::size_t>(block_expression_index) * expression_stride_words;
            if (expression_words == nullptr || expression_words->size() < base + expression_stride_words) {
                fail("batch presampled expression workspace is too short");
            }
            if (nwords == 1) {
                out[0] = (*expression_words)[base] & live_word_mask_for_shots(runtime.active_shots, 0);
            } else {
                for (std::size_t word = 0; word < nwords; ++word) {
                    out[word] = (*expression_words)[base + word] & live_word_mask_for_shots(runtime.active_shots, word);
                }
            }
        }
        if (nwords < out.size()) {
            std::fill(out.begin() + static_cast<std::ptrdiff_t>(nwords), out.end(), 0);
        }
        if (!expression.residual_plan.conditions.empty()) {
            xor_symbolic_bool_batch_into(out, expression.residual_plan, runtime);
        }
        return out;
    }
};

constexpr std::size_t kShotMajorRotationRunLimit = 32;

bool is_active_rotation_instruction(const FactoredInstruction& instruction) {
    return std::get_if<ApplyPrecomputedActivePauliRotation>(&instruction) != nullptr;
}

const ApplyPrecomputedActivePauliRotation& active_rotation_instruction_at(
    const FactoredInstructionProgram& program,
    std::size_t instruction_index) {
    const auto* rotation = std::get_if<ApplyPrecomputedActivePauliRotation>(
        &program.instructions[instruction_index]);
    if (rotation == nullptr) {
        fail("internal shot-major rotation run contains a non-rotation instruction");
    }
    return *rotation;
}

std::size_t shot_major_rotation_run_length(
    const FactoredInstructionProgram& program,
    std::size_t first_index) {
    std::size_t run_len = 0;
    while (first_index + run_len < program.instructions.size() &&
           run_len < kShotMajorRotationRunLimit &&
           is_active_rotation_instruction(program.instructions[first_index + run_len])) {
        ++run_len;
    }
    return run_len;
}

bool rotation_run_sign_at(
    const std::vector<std::uint64_t>& sign_words,
    std::size_t stride_words,
    std::size_t run_offset,
    int shot) {
    const std::size_t word = batch_shot_word(shot);
    return (sign_words[run_offset * stride_words + word] & batch_shot_mask(shot)) != 0;
}

bool packed_bit_at(const std::vector<std::uint64_t>& bits, int shot) {
    const std::size_t word = batch_shot_word(shot);
    return word < bits.size() && ((bits[word] & batch_shot_mask(shot)) != 0);
}

bool postselected_shot_is_dead(const BatchDetectorPostselectionScratch& scratch, int shot) {
    return scratch.dead_count != 0 && packed_bit_at(scratch.dead_bits, shot);
}

std::size_t execute_shot_major_rotation_run(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const BatchExpressionEvaluator& evaluator,
    std::size_t first_index,
    const BatchDetectorPostselectionScratch* postselection_scratch = nullptr) {
    const std::size_t run_len = shot_major_rotation_run_length(program, first_index);
    if (run_len <= 1 || runtime.active_shots == 0) {
        return 0;
    }

    const std::size_t active_words = runtime_batch_word_count(runtime);
    const std::size_t total_words = run_len * active_words;
    if (runtime.rotation_run_sign_words.size() < total_words) {
        runtime.rotation_run_sign_words.resize(total_words, 0);
    }
    std::array<const ApplyPrecomputedActivePauliRotation*, kShotMajorRotationRunLimit> rotations{};
    for (std::size_t run_offset = 0; run_offset < run_len; ++run_offset) {
        const auto& rotation = active_rotation_instruction_at(program, first_index + run_offset);
        rotations[run_offset] = &rotation;
        if (rotation.rotation_kernel.action.nqubits != runtime.k) {
            fail("rotation kernel dimension does not match batch active state");
        }
        const auto& sign_bits = evaluator.eval(first_index + run_offset, runtime);
        std::copy_n(
            sign_bits.data(),
            active_words,
            runtime.rotation_run_sign_words.data() + run_offset * active_words);
    }

    const std::size_t dim = active_length(runtime.k);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        if (postselection_scratch != nullptr && postselected_shot_is_dead(*postselection_scratch, shot)) {
            continue;
        }
        double* re = batch_active_re_for_shot(runtime, shot);
        double* im = batch_active_im_for_shot(runtime, shot);
        for (std::size_t run_offset = 0; run_offset < run_len; ++run_offset) {
            const auto& rotation = *rotations[run_offset];
            detail::rotate_contiguous_active(
                re,
                im,
                dim,
                rotation.rotation_kernel,
                rotation_run_sign_at(runtime.rotation_run_sign_words, active_words, run_offset, shot));
        }
    }
    return run_len;
}

void rotate_shot_major_postselected(
    BatchFactoredExecutorState& runtime,
    const PrecomputedActivePauliRotationKernel& kernel,
    const std::vector<std::uint64_t>& sign_bits,
    const BatchDetectorPostselectionScratch& scratch) {
    if (!runtime.dense_shot_major_active || scratch.dead_count == 0) {
        rotate_pauli_batch(runtime, kernel, sign_bits);
        return;
    }
    if (kernel.action.nqubits != runtime.k) {
        fail("rotation kernel dimension does not match batch active state");
    }
    const std::size_t dim = active_length(runtime.k);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        if (postselected_shot_is_dead(scratch, shot)) {
            continue;
        }
        detail::rotate_contiguous_active(
            batch_active_re_for_shot(runtime, shot),
            batch_active_im_for_shot(runtime, shot),
            dim,
            kernel,
            packed_bit_at(sign_bits, shot));
    }
}

void promote_first_dormant_rotation_shot_major_postselected(
    BatchFactoredExecutorState& runtime,
    double kernel_angle,
    const std::vector<std::uint64_t>& sign_bits,
    const BatchDetectorPostselectionScratch& scratch) {
    if (!runtime.dense_shot_major_active || scratch.dead_count == 0) {
        promote_first_dormant_rotation_batch(runtime, kernel_angle, sign_bits);
        return;
    }
    if (runtime.ndormant <= 0) {
        fail("cannot promote a dormant qubit when none remain");
    }
    const std::size_t dim = active_length(runtime.k);
    const std::size_t promoted_dim = 2 * dim;
    if (runtime.active_stride < promoted_dim) {
        fail("batch active shot-major stride is too short for dormant promotion");
    }
    const double c = std::cos(kernel_angle);
    const double s = std::sin(kernel_angle);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        if (postselected_shot_is_dead(scratch, shot)) {
            continue;
        }
        const double q = packed_bit_at(sign_bits, shot) ? s : -s;
        detail::promote_contiguous_active(
            batch_active_re_for_shot(runtime, shot),
            batch_active_im_for_shot(runtime, shot),
            dim,
            c,
            q);
    }
    ++runtime.k;
    --runtime.ndormant;
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const ApplyPrecomputedActivePauliRotation& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& sign_bits = evaluator.eval(instruction_index, runtime);
    rotate_pauli_batch(runtime, instruction.rotation_kernel, sign_bits);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const PromoteDormantRotation& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& sign_bits = evaluator.eval(instruction_index, runtime);
    promote_first_dormant_rotation_batch(runtime, instruction.kernel_angle, sign_bits);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const RecordMeasurement& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const RecordDetector& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& outcome_bits = !instruction.records.empty() || instruction.outcome.conditions.empty()
                                   ? detector_record_outcome_bits(runtime, instruction, runtime.eval_scratch)
                                   : evaluator.eval(instruction_index, runtime);
    if (!runtime.store_detector_records) {
        const std::size_t nwords = runtime_batch_word_count(runtime);
        for (std::size_t word = 0; word < nwords; ++word) {
            runtime.detector_any_words[word] |= outcome_bits[word] & batch_live_word_mask(runtime, word);
        }
        return;
    }
    write_batch_detector_record(runtime, instruction.detector, outcome_bits);
}

int execute_batch_instruction_postselected(
    BatchFactoredExecutorState& runtime,
    const ApplyPrecomputedActivePauliRotation& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index,
    BatchDetectorPostselectionScratch& scratch) {
    const auto& sign_bits = evaluator.eval(instruction_index, runtime);
    rotate_shot_major_postselected(runtime, instruction.rotation_kernel, sign_bits, scratch);
    return 0;
}

int execute_batch_instruction_postselected(
    BatchFactoredExecutorState& runtime,
    const PromoteDormantRotation& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index,
    BatchDetectorPostselectionScratch& scratch) {
    const auto& sign_bits = evaluator.eval(instruction_index, runtime);
    promote_first_dormant_rotation_shot_major_postselected(runtime, instruction.kernel_angle, sign_bits, scratch);
    return 0;
}

int execute_batch_instruction_postselected(
    BatchFactoredExecutorState& runtime,
    const RecordDetector& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index,
    BatchDetectorPostselectionScratch& scratch) {
    if (!instruction.records.empty()) {
        return mark_dead_from_detector_records(runtime, instruction, scratch);
    }
    if (instruction.outcome.conditions.empty()) {
        return mark_dead_from_constant_detector(runtime, instruction.outcome.constant, scratch);
    }
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    return mark_dead_from_detector_bits(runtime, outcome_bits, scratch);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const MeasurePrecomputedActivePauli& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    measure_precomputed_active_pauli_branch_batch(runtime, instruction.kernel, instruction.branch);
    if (write_direct_branch_measurement_record(
            runtime,
            instruction.branch,
            instruction.outcome_plan,
            instruction.record,
            instruction.record_condition)) {
        return;
    }
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const IntroduceDormantMeasurementBranch& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    fill_batch_random_half_bits(runtime.eval_scratch, runtime);
    const auto& branch_bits = runtime.eval_scratch;
    assign_batch_symbol(runtime, instruction.branch, branch_bits);
    if (write_direct_branch_measurement_record(
            runtime,
            instruction.branch,
            instruction.outcome_plan,
            instruction.record,
            instruction.record_condition)) {
        return;
    }
    const auto& outcome_bits = evaluator.eval(instruction_index, runtime);
    write_batch_measurement_record(runtime, instruction.record, outcome_bits, instruction.record_condition);
}

void execute_batch_component_instruction_presampled(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index) {
    const auto& plan = *program.active_component_plan;
    const auto ref = plan.instruction_steps[instruction_index];
    switch (ref.kind) {
    case ActiveComponentStepKind::IgnoredGlobalPhase:
        return;
    case ActiveComponentStepKind::Rotation: {
        const auto& sign_bits =
            evaluator.eval(instruction_index, runtime);
        execute_batch_component_rotation(
            runtime,
            plan,
            plan.rotations[ref.payload],
            sign_bits);
        return;
    }
    case ActiveComponentStepKind::Promotion: {
        const auto& instruction =
            std::get<PromoteDormantRotation>(
                program.instructions[instruction_index]);
        const auto& sign_bits =
            evaluator.eval(instruction_index, runtime);
        execute_batch_component_promotion(
            runtime,
            plan.promotions[ref.payload],
            instruction.kernel_angle,
            sign_bits);
        return;
    }
    case ActiveComponentStepKind::Measurement: {
        const auto& instruction =
            std::get<MeasurePrecomputedActivePauli>(
                program.instructions[instruction_index]);
        execute_batch_component_measurement(
            runtime,
            plan,
            plan.measurements[ref.payload],
            instruction.branch);
        if (write_direct_branch_measurement_record(
                runtime,
                instruction.branch,
                instruction.outcome_plan,
                instruction.record,
                instruction.record_condition)) {
            return;
        }
        const auto& outcome_bits =
            evaluator.eval(instruction_index, runtime);
        write_batch_measurement_record(
            runtime,
            instruction.record,
            outcome_bits,
            instruction.record_condition);
        return;
    }
    case ActiveComponentStepKind::None:
        std::visit(
            [&](const auto& inst) {
                execute_batch_instruction_presampled(
                    runtime,
                    inst,
                    evaluator,
                    instruction_index);
            },
            program.instructions[instruction_index]);
        return;
    }
    fail("unknown active component instruction kind");
}

template <typename Instruction>
int execute_batch_instruction_postselected(
    BatchFactoredExecutorState& runtime,
    const Instruction& instruction,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index,
    BatchDetectorPostselectionScratch& scratch) {
    execute_batch_instruction_presampled(runtime, instruction, evaluator, instruction_index);
    if constexpr (std::is_same_v<std::decay_t<Instruction>, RecordDetector>) {
        return mark_dead_from_detector_bits(runtime, runtime.eval_scratch, scratch);
    }
    return 0;
}

int execute_batch_component_instruction_postselected(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const BatchExpressionEvaluator& evaluator,
    std::size_t instruction_index,
    BatchDetectorPostselectionScratch& scratch) {
    const auto& plan = *program.active_component_plan;
    const auto ref = plan.instruction_steps[instruction_index];
    if (ref.kind == ActiveComponentStepKind::IgnoredGlobalPhase) {
        return 0;
    }
    if (ref.kind == ActiveComponentStepKind::Rotation) {
        const auto& sign_bits =
            evaluator.eval(instruction_index, runtime);
        const auto& step = plan.rotations[ref.payload];
        merge_batch_components(
            runtime,
            plan,
            step.merge_offset,
            step.merge_count,
            step.component);
        auto& component =
            runtime.active_components[
                static_cast<std::size_t>(step.component)];
        ScopedBatchComponent scope(runtime, component);
        rotate_shot_major_postselected(
            runtime,
            step.kernel,
            sign_bits,
            scratch);
        return 0;
    }
    if (ref.kind != ActiveComponentStepKind::None) {
        execute_batch_component_instruction_presampled(
            runtime,
            program,
            evaluator,
            instruction_index);
        return 0;
    }
    return std::visit(
        [&](const auto& inst) {
            return execute_batch_instruction_postselected(
                runtime,
                inst,
                evaluator,
                instruction_index,
                scratch);
        },
        program.instructions[instruction_index]);
}

void initialize_expression_workspace(
    std::vector<std::uint64_t>& expression_words,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    std::size_t stride_words,
    int active_shots,
    int first_sample_shot) {
    if (first_sample_shot < 0 || active_shots > expression_block.nshots - first_sample_shot) {
        fail("batch presampled expression source range is out of bounds");
    }
    const std::size_t expression_count = expression_plan.block_expressions.size();
    const std::size_t total_words = expression_count * stride_words;
    if (expression_words.size() < total_words) {
        expression_words.resize(total_words, 0);
    }
    for (std::size_t expression = 0; expression < expression_count; ++expression) {
        const std::size_t base = expression * stride_words;
        for (std::size_t word = 0; word < stride_words; ++word) {
            expression_words[base + word] = expression_slice_word(
                expression_block,
                static_cast<int>(expression),
                first_sample_shot,
                active_shots,
                word);
        }
    }
}

BatchDetectorPostselectionResult execute_batch_postselected_with_expressions(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot,
    BatchDetectorPostselectionScratch& scratch,
    const BatchDetectorPostselectionOptions& options) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("batch presampled expression plan does not match program");
    }
    if (expression_plan.block_expression_last_use_by_index.size() != expression_plan.block_expressions.size()) {
        fail("batch presampled expression last-use table does not match expression plan");
    }
    prepare_batch_detector_postselection_scratch(scratch, runtime, program, options);
    std::fill(scratch.dead_bits.begin(), scratch.dead_bits.end(), 0);
    scratch.dead_count = 0;
    if (runtime.active_shots == 0) {
        return {};
    }
    int discarded = 0;
    BatchExpressionEvaluator evaluator{
        expression_plan,
        nullptr,
        &expression_block,
        0,
        first_sample_shot,
        runtime.eval_scratch};
    bool expression_workspace_materialized = false;
    auto materialize_expression_workspace = [&]() {
        if (expression_workspace_materialized) {
            return;
        }
        initialize_expression_workspace(
            scratch.expression_words,
            expression_plan,
            expression_block,
            runtime.batch_words,
            runtime.active_shots,
            first_sample_shot);
        evaluator.expression_words = &scratch.expression_words;
        evaluator.expression_block = nullptr;
        evaluator.expression_stride_words = runtime.batch_words;
        evaluator.first_sample_shot = 0;
        expression_workspace_materialized = true;
    };

    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        if (runtime.active_shots == 0) {
            break;
        }
        if (should_compact_dead_before_instruction(runtime, scratch, options, program.instructions[idx])) {
            materialize_expression_workspace();
            compact_dead_shots_if_needed(
                runtime,
                scratch,
                scratch.condition_last_use_by_index,
                scratch.record_last_use_by_index,
                static_cast<int>(idx) - 1,
                false,
                &scratch.expression_words,
                &expression_plan.block_expression_last_use_by_index);
            if (runtime.active_shots == 0) {
                break;
            }
        }
        if (runtime.dense_shot_major_active &&
            !runtime.active_components_enabled) {
            const std::size_t consumed = execute_shot_major_rotation_run(
                runtime,
                program,
                evaluator,
                idx,
                &scratch);
            if (consumed != 0) {
                idx += consumed - 1;
                continue;
            }
        }
        const auto& instruction = program.instructions[idx];
        if (const auto* detector = std::get_if<RecordDetector>(&instruction)) {
            discarded += execute_batch_instruction_postselected(
                runtime,
                *detector,
                evaluator,
                idx,
                scratch);
            if (scratch.dead_count >= runtime.active_shots) {
                runtime.active_shots = 0;
                break;
            }
            continue;
        }
        if (runtime.active_components_enabled) {
            discarded += execute_batch_component_instruction_postselected(
                runtime,
                program,
                evaluator,
                idx,
                scratch);
        } else {
            discarded += std::visit(
                [&](const auto& inst) {
                    return execute_batch_instruction_postselected(
                        runtime,
                        inst,
                        evaluator,
                        idx,
                        scratch);
                },
                instruction);
        }
    }
    compact_dead_shots_if_needed(
        runtime,
        scratch,
        scratch.condition_last_use_by_index,
        scratch.record_last_use_by_index,
        static_cast<int>(program.instructions.size()),
        true,
        expression_workspace_materialized ? &scratch.expression_words : nullptr,
        expression_workspace_materialized ? &expression_plan.block_expression_last_use_by_index : nullptr);
    return {discarded, runtime.active_shots};
}

} // namespace


int default_batch_count(int max_k) {
    const std::size_t dim = active_length(max_k);
    const std::size_t count = std::min<std::size_t>(
        kDefaultBatchShots,
        std::max<std::size_t>(1, kDefaultBatchActiveAmplitudes / dim));
    return static_cast<int>(count);
}

const char* active_batch_backend() {
    return simd::dispatch_name();
}

void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples) {
    assign_presampled_exogenous_batch(runtime, samples);
}

void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int first_sample_shot) {
    assign_presampled_exogenous_batch(runtime, samples, first_sample_shot);
}

void execute_batch_instruction_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction) {
    if (runtime.active_components_enabled) {
        fail("direct batch instruction execution is unavailable for a component executor");
    }
    execute_batch_instruction(runtime, instruction);
}

BatchFactoredExecutorState::BatchFactoredExecutorState(
    const FactoredInstructionProgram& program,
    int batches_,
    std::uint64_t seed)
    : batches(batches_ > 0 ? batches_ : default_batch_count(program.max_k)),
      rng_state(seed) {
    reset_batch_executor(*this, program, batches);
}

void reset_batch_executor(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    int shots,
    bool clear_symbol_values) {
    if (shots < 0 || shots > runtime.batches) {
        fail("active batch shot count is out of range");
    }
    runtime.n = program.n;
    runtime.k = program.initial_k;
    runtime.ndormant = program.n - program.initial_k;
    runtime.active_shots = shots;
    runtime.active_pitch = padded_batch_active_pitch(runtime.batches);
    runtime.nsymbols = program.nsymbols;
    runtime.nrecords = program.nrecords;
    runtime.ndetectors = program.ndetectors;
    runtime.max_k = program.max_k;
    runtime.batch_words = batch_word_count(runtime.batches);

    reset_batch_components(runtime, program);
    if (!runtime.active_components_enabled) {
        ensure_dense_batch_active_storage(runtime, program);
    }

    const std::size_t symbol_size = static_cast<std::size_t>(program.nsymbols) * runtime.batch_words;
    if (runtime.value_words.size() != symbol_size) {
        runtime.value_words.resize(symbol_size, 0);
    }
    if (clear_symbol_values) {
        std::fill(runtime.value_words.begin(), runtime.value_words.end(), 0);
    }
    const std::size_t assigned_size = symbol_word_count(program.nsymbols);
    if (runtime.assigned_words.size() != assigned_size) {
        runtime.assigned_words.resize(assigned_size, 0);
    }
    std::fill(runtime.assigned_words.begin(), runtime.assigned_words.end(), 0);
    const std::size_t measurement_size = static_cast<std::size_t>(program.nrecords) * runtime.batch_words;
    if (runtime.measurement_words.size() != measurement_size) {
        runtime.measurement_words.resize(measurement_size, 0);
    }
    std::fill(runtime.measurement_words.begin(), runtime.measurement_words.end(), 0);
    if (runtime.store_detector_records) {
        const std::size_t detector_size = static_cast<std::size_t>(program.ndetectors) * runtime.batch_words;
        if (runtime.detector_words.size() != detector_size) {
            runtime.detector_words.resize(detector_size, 0);
        }
        std::fill(runtime.detector_words.begin(), runtime.detector_words.end(), 0);
    }
    if (runtime.detector_any_words.size() != runtime.batch_words) {
        runtime.detector_any_words.resize(runtime.batch_words, 0);
    }
    std::fill(runtime.detector_any_words.begin(), runtime.detector_any_words.end(), 0);
    if (runtime.eval_scratch.size() != runtime.batch_words) {
        runtime.eval_scratch.resize(runtime.batch_words, 0);
    }
    std::fill(runtime.eval_scratch.begin(), runtime.eval_scratch.end(), 0);
    if (runtime.shot_coefficient_scalars.size() < static_cast<std::size_t>(runtime.active_pitch)) {
        runtime.shot_coefficient_scalars.resize(static_cast<std::size_t>(runtime.active_pitch), 0.0);
    }
    if (runtime.branch_prob_true.size() < static_cast<std::size_t>(runtime.active_pitch)) {
        runtime.branch_prob_true.resize(static_cast<std::size_t>(runtime.active_pitch), 0.0);
    }
    if (runtime.branch_invnorms.size() < static_cast<std::size_t>(runtime.active_pitch)) {
        runtime.branch_invnorms.resize(static_cast<std::size_t>(runtime.active_pitch), 0.0);
    }

    if (!runtime.active_components_enabled) {
        initialize_dense_batch_active(runtime, program);
    }
}

void execute_batch_in_place(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    sample_exogenous_symbols_batch(runtime, program);
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_batch_component_instruction(runtime, program, idx);
        }
        return;
    }
    for (const auto& instruction : program.instructions) {
        execute_batch_instruction(runtime, instruction);
    }
}

void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    assign_presampled_exogenous_batch(runtime, samples);
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_batch_component_instruction(runtime, program, idx);
        }
        return;
    }
    for (const auto& instruction : program.instructions) {
        execute_batch_instruction(runtime, instruction);
    }
}

void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    assign_presampled_exogenous_batch(runtime, samples, first_sample_shot);
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_batch_component_instruction(runtime, program, idx);
        }
        return;
    }
    for (const auto& instruction : program.instructions) {
        execute_batch_instruction(runtime, instruction);
    }
}

void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot) {
    if (runtime.n != program.n || runtime.k + runtime.ndormant != runtime.n) {
        fail("batch executor state does not match program");
    }
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        fail("batch presampled expression plan does not match program");
    }
    if (runtime.active_shots == 0) {
        return;
    }
    BatchExpressionEvaluator evaluator{
        expression_plan,
        nullptr,
        &expression_block,
        0,
        first_sample_shot,
        runtime.eval_scratch};
    if (runtime.active_components_enabled) {
        for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
            execute_batch_component_instruction_presampled(
                runtime,
                program,
                evaluator,
                idx);
        }
        return;
    }
    for (std::size_t idx = 0; idx < program.instructions.size();) {
        if (runtime.dense_shot_major_active) {
            const std::size_t consumed = execute_shot_major_rotation_run(
                runtime,
                program,
                evaluator,
                idx);
            if (consumed != 0) {
                idx += consumed;
                continue;
            }
        }
        const auto& instruction = program.instructions[idx];
        std::visit(
            [&](const auto& inst) {
                execute_batch_instruction_presampled(
                    runtime,
                    inst,
                    evaluator,
                    idx);
            },
            instruction);
        ++idx;
    }
}

void prepare_batch_detector_postselection_scratch(
    BatchDetectorPostselectionScratch& scratch,
    const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime.batch_words;
    ensure_word_scratch(scratch.dead_bits, nwords);
    ensure_word_scratch(scratch.keep_bits, nwords);
    ensure_word_scratch(scratch.scratch, nwords);
    ensure_word_scratch(scratch.compact_scratch, nwords);
    if (scratch.live_sources.capacity() < static_cast<std::size_t>(runtime.batches)) {
        scratch.live_sources.reserve(static_cast<std::size_t>(runtime.batches));
    }
}

void prepare_batch_detector_postselection_scratch(
    BatchDetectorPostselectionScratch& scratch,
    const BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const BatchDetectorPostselectionOptions& options) {
    prepare_batch_detector_postselection_scratch(scratch, runtime);
    refresh_batch_postselection_metadata(scratch, program, options);
}

BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot,
    BatchDetectorPostselectionScratch& scratch,
    const BatchDetectorPostselectionOptions& options) {
    return execute_batch_postselected_with_expressions(
        runtime,
        program,
        expression_plan,
        expression_block,
        first_sample_shot,
        scratch,
        options);
}

std::vector<std::vector<std::uint64_t>> sample_measurements_batch(
    const FactoredInstructionProgram& program,
    int shots,
    int batches,
    std::uint64_t seed) {
    if (shots < 0) {
        fail("shot count must be nonnegative");
    }
    BatchFactoredExecutorState runtime(program, batches, seed);
    std::vector<std::vector<std::uint64_t>> out(
        static_cast<std::size_t>(shots),
        std::vector<std::uint64_t>(symbol_word_count(program.nrecords), 0));

    int offset = 0;
    while (offset < shots) {
        const int block = std::min(runtime.batches, shots - offset);
        reset_batch_executor(runtime, program, block);
        execute_batch_in_place(runtime, program);
        for (int shot = 0; shot < block; ++shot) {
            auto& row = out[static_cast<std::size_t>(offset + shot)];
            for (int record = 1; record <= runtime.nrecords; ++record) {
                const std::size_t word = batch_shot_word(shot);
                const bool bit = (runtime.measurement_words[batch_record_offset(runtime, record, word)] &
                                  batch_shot_mask(shot)) != 0;
                if (bit) {
                    const int record_bit = record - 1;
                    row[static_cast<std::size_t>(record_bit >> 6)] |=
                        std::uint64_t{1} << (record_bit & 63);
                }
            }
        }
        offset += block;
    }
    return out;
}

} // namespace symft
