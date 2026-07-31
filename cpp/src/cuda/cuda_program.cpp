#include "cuda/cuda_program.hpp"

#include "core/internal.hpp"
#include "sampler/active_internal.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <optional>
#include <variant>

namespace symft::cuda {
namespace {

using detail::symbol_word_count;

std::int32_t checked_i32(std::size_t value, const char* what) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw Error(std::string(what) + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

std::int32_t append_records(CudaProgramData& out, const std::vector<int>& records) {
    const std::int32_t offset = checked_i32(out.record_table.size(), "record table");
    out.record_table.reserve(out.record_table.size() + records.size());
    for (int record : records) {
        out.record_table.push_back(record);
    }
    return offset;
}

std::int32_t append_conditions(CudaProgramData& out, const std::vector<int>& conditions) {
    const std::int32_t offset = checked_i32(out.sample_condition_table.size(), "sample condition table");
    out.sample_condition_table.reserve(out.sample_condition_table.size() + conditions.size());
    for (int condition : conditions) {
        out.sample_condition_table.push_back(condition);
    }
    return offset;
}

std::int32_t append_block_expression_conditions(CudaProgramData& out, const std::vector<int>& conditions) {
    const std::int32_t offset = checked_i32(
        out.block_expression_condition_table.size(),
        "block expression condition table");
    out.block_expression_condition_table.reserve(out.block_expression_condition_table.size() + conditions.size());
    for (int condition : conditions) {
        out.block_expression_condition_table.push_back(condition);
    }
    return offset;
}

void append_block_expression_plans(CudaProgramData& out, const PresampledExpressionPlan& plan) {
    out.block_expression_plans.reserve(plan.block_expressions.size());
    for (const auto& expression : plan.block_expressions) {
        CudaBlockExpression gpu;
        gpu.parent = expression.parent_block_expression_index;
        gpu.constant = expression.constant ? 1 : 0;
        gpu.parent_delta_constant = expression.parent_delta_constant ? 1 : 0;
        gpu.condition_offset = append_block_expression_conditions(out, expression.exogenous_conditions);
        gpu.condition_count = checked_i32(expression.exogenous_conditions.size(), "block expression condition count");
        gpu.delta_condition_offset = append_block_expression_conditions(out, expression.parent_delta_exogenous_conditions);
        gpu.delta_condition_count = checked_i32(
            expression.parent_delta_exogenous_conditions.size(),
            "block expression delta condition count");
        const auto mask_plan = SymbolicBoolEvaluationPlan(SymbolicBool(false, expression.exogenous_conditions));
        gpu.mask_offset = checked_i32(out.block_expression_masks.size(), "block expression mask table");
        gpu.mask_count = checked_i32(mask_plan.word_indices.size(), "block expression mask count");
        for (std::size_t idx = 0; idx < mask_plan.word_indices.size(); ++idx) {
            out.block_expression_masks.push_back(CudaWordMask{
                mask_plan.word_indices[idx],
                mask_plan.word_masks[idx],
            });
        }
        out.block_expression_plans.push_back(gpu);
    }
}

std::int32_t append_probabilities(CudaProgramData& out, const std::vector<double>& probabilities) {
    const std::int32_t offset = checked_i32(out.sample_probability_table.size(), "sample probability table");
    out.sample_probability_table.reserve(out.sample_probability_table.size() + probabilities.size());
    for (double probability : probabilities) {
        out.sample_probability_table.push_back(probability);
    }
    return offset;
}

std::uint64_t packed_assignment_word(const std::vector<std::uint64_t>& assignment, int nbits) {
    if (nbits > 64) {
        throw Error("CUDA categorical sampler only supports up to 64 assignment bits per row");
    }
    std::uint64_t out = 0;
    for (int bit = 0; bit < nbits; ++bit) {
        if (packed_bit(assignment, bit)) {
            out |= std::uint64_t{1} << bit;
        }
    }
    return out;
}

std::int32_t append_assignments(
    CudaProgramData& out,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    int nbits) {
    const std::int32_t offset = checked_i32(out.sample_assignment_table.size(), "sample assignment table");
    out.sample_assignment_table.reserve(out.sample_assignment_table.size() + assignments.size());
    for (const auto& assignment : assignments) {
        out.sample_assignment_table.push_back(packed_assignment_word(assignment, nbits));
    }
    return offset;
}

std::int32_t append_expression(CudaProgramData& out, const PresampledExpression& expression) {
    CudaExpression gpu;
    gpu.block_expression = expression.block_expression_index;
    gpu.residual_offset = checked_i32(out.residual_masks.size(), "residual mask table");
    gpu.residual_count = checked_i32(expression.residual_plan.word_indices.size(), "residual mask count");
    gpu.residual_constant = expression.residual_plan.constant ? 1 : 0;
    for (std::size_t idx = 0; idx < expression.residual_plan.word_indices.size(); ++idx) {
        out.residual_masks.push_back(CudaWordMask{
            expression.residual_plan.word_indices[idx],
            expression.residual_plan.word_masks[idx],
        });
    }
    out.expressions.push_back(gpu);
    return checked_i32(out.expressions.size() - 1, "expression index");
}

std::int32_t append_full_expression(CudaProgramData& out, const SymbolicBoolEvaluationPlan& plan) {
    CudaExpression gpu;
    gpu.block_expression = -1;
    gpu.residual_offset = checked_i32(out.residual_masks.size(), "residual mask table");
    gpu.residual_count = checked_i32(plan.word_indices.size(), "residual mask count");
    gpu.residual_constant = plan.constant ? 1 : 0;
    for (std::size_t idx = 0; idx < plan.word_indices.size(); ++idx) {
        out.residual_masks.push_back(CudaWordMask{
            plan.word_indices[idx],
            plan.word_masks[idx],
        });
    }
    out.expressions.push_back(gpu);
    return checked_i32(out.expressions.size() - 1, "expression index");
}

std::int32_t append_rotation(CudaProgramData& out, const PrecomputedActivePauliRotationKernel& kernel) {
    CudaRotationKernel gpu;
    if (kernel.is_diagonal) {
        gpu.kind = CudaRotationKernelKind::Diagonal;
    } else if (kernel.uniform_imag_pairs) {
        gpu.kind = CudaRotationKernelKind::UniformImagPairs;
    } else if (kernel.real_pair_flip) {
        gpu.kind = CudaRotationKernelKind::RealPairFlip;
    } else {
        gpu.kind = CudaRotationKernelKind::GeneralPair;
    }
    gpu.is_diagonal = kernel.is_diagonal ? 1 : 0;
    gpu.uniform_imag_pairs = kernel.uniform_imag_pairs ? 1 : 0;
    gpu.real_pair_flip = kernel.real_pair_flip ? 1 : 0;
    gpu.xz_overlap_odd = kernel.action.xz_overlap_odd ? 1 : 0;
    gpu.pair_bit = static_cast<std::int32_t>(kernel.pair_bit);
    gpu.pair_count = checked_i32(kernel.pair_count, "rotation pair count");
    gpu.xmask = kernel.action.xmask;
    gpu.zmask = kernel.action.zmask;
    gpu.minus_even_re = static_cast<CudaReal>(kernel.minus_even_coefficient.real());
    gpu.minus_even_im = static_cast<CudaReal>(kernel.minus_even_coefficient.imag());
    gpu.cos_angle = kernel.cos_kernel_angle;
    out.rotations.push_back(gpu);
    return checked_i32(out.rotations.size() - 1, "rotation index");
}

std::int32_t append_measurement(CudaProgramData& out, const PrecomputedActivePauliMeasurementKernel& kernel) {
    CudaMeasurementKernel gpu;
    gpu.is_diagonal = kernel.is_diagonal ? 1 : 0;
    gpu.pivot = kernel.pivot;
    gpu.xmask = kernel.action.xmask;
    gpu.zmask = kernel.action.zmask;
    gpu.even_phase_re = static_cast<CudaReal>(kernel.action.even_phase.real());
    gpu.even_phase_im = static_cast<CudaReal>(kernel.action.even_phase.imag());
    gpu.diagonal_phase_bit = kernel.diagonal_phase_bit;
    gpu.out_dim = checked_i32(kernel.out_dim, "measurement output dimension");
    out.measurements.push_back(gpu);
    return checked_i32(out.measurements.size() - 1, "measurement index");
}

std::int32_t optional_id(std::optional<int> value) {
    return value ? *value : 0;
}

CudaInstruction base_instruction(
    CudaInstructionKind kind,
    std::size_t instruction_index) {
    CudaInstruction inst;
    inst.kind = kind;
    inst.expression = checked_i32(instruction_index, "instruction expression index");
    return inst;
}

void append_instruction(
    CudaProgramData& out,
    const ApplyPrecomputedActivePauliRotation& source,
    const PresampledExpressionPlan& expression_plan,
    std::size_t instruction_index) {
    auto inst = base_instruction(
        CudaInstructionKind::ActiveRotation,
        instruction_index);
    (void)expression_plan;
    inst.rotation = append_rotation(out, source.rotation_kernel);
    out.instructions.push_back(inst);
}

void append_instruction(
    CudaProgramData& out,
    const PromoteDormantRotation& source,
    const PresampledExpressionPlan& expression_plan,
    std::size_t instruction_index) {
    auto inst = base_instruction(
        CudaInstructionKind::PromoteDormantRotation,
        instruction_index);
    (void)expression_plan;
    inst.kernel_angle = source.kernel_angle;
    inst.kernel_cos_angle = static_cast<CudaReal>(std::cos(source.kernel_angle));
    inst.kernel_sin_angle = static_cast<CudaReal>(std::sin(source.kernel_angle));
    out.instructions.push_back(inst);
}

void append_instruction(
    CudaProgramData& out,
    const RecordMeasurement& source,
    const PresampledExpressionPlan& expression_plan,
    std::size_t instruction_index) {
    auto inst = base_instruction(
        CudaInstructionKind::RecordMeasurement,
        instruction_index);
    (void)expression_plan;
    inst.record = optional_id(source.record);
    inst.record_condition = optional_id(source.record_condition);
    out.instructions.push_back(inst);
}

void append_instruction(
    CudaProgramData& out,
    const RecordDetector& source,
    const PresampledExpressionPlan& expression_plan,
    std::size_t instruction_index) {
    auto inst = base_instruction(
        CudaInstructionKind::RecordDetector,
        instruction_index);
    (void)expression_plan;
    inst.detector = source.detector;
    inst.record_list_offset = append_records(out, source.records);
    inst.record_list_count = checked_i32(source.records.size(), "detector record count");
    out.instructions.push_back(inst);
}

void append_instruction(
    CudaProgramData& out,
    const MeasurePrecomputedActivePauli& source,
    const PresampledExpressionPlan& expression_plan,
    std::size_t instruction_index) {
    auto inst = base_instruction(
        CudaInstructionKind::ActiveMeasurement,
        instruction_index);
    (void)expression_plan;
    inst.measurement = append_measurement(out, source.kernel);
    inst.branch_condition = source.branch;
    inst.record = optional_id(source.record);
    inst.record_condition = optional_id(source.record_condition);
    out.instructions.push_back(inst);
}

void append_instruction(
    CudaProgramData& out,
    const IntroduceDormantMeasurementBranch& source,
    const PresampledExpressionPlan& expression_plan,
    std::size_t instruction_index) {
    auto inst = base_instruction(
        CudaInstructionKind::IntroduceDormantBranch,
        instruction_index);
    (void)expression_plan;
    inst.branch_condition = source.branch;
    inst.record = optional_id(source.record);
    inst.record_condition = optional_id(source.record_condition);
    out.instructions.push_back(inst);
}

const ApplyPrecomputedActivePauliRotation* active_rotation_at(const FactoredInstruction& instruction) {
    return std::get_if<ApplyPrecomputedActivePauliRotation>(&instruction);
}

void append_rotation_run(
    CudaProgramData& out,
    const FactoredInstructionProgram& program,
    std::size_t begin,
    std::size_t count) {
    auto inst = base_instruction(CudaInstructionKind::ActiveRotationRun, begin);
    inst.rotation_run_offset = checked_i32(out.rotation_run_items.size(), "rotation run table");
    inst.rotation_run_count = checked_i32(count, "rotation run length");
    if (inst.rotation_run_count > out.max_rotation_run_length) {
        out.max_rotation_run_length = inst.rotation_run_count;
    }

    out.rotation_run_items.reserve(out.rotation_run_items.size() + count);
    for (std::size_t idx = 0; idx < count; ++idx) {
        const std::size_t instruction_index = begin + idx;
        const auto* rotation = active_rotation_at(program.instructions[instruction_index]);
        if (rotation == nullptr) {
            throw Error("internal CUDA rotation run contains a non-rotation instruction");
        }
        out.rotation_run_items.push_back(CudaRotationRunItem{
            checked_i32(instruction_index, "instruction expression index"),
            append_rotation(out, rotation->rotation_kernel),
        });
    }
    out.instructions.push_back(inst);
}

void append_instruction_stream(
    CudaProgramData& out,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan) {
    for (std::size_t idx = 0; idx < program.instructions.size();) {
        if (active_rotation_at(program.instructions[idx]) != nullptr) {
            std::size_t end = idx + 1;
            while (end < program.instructions.size() &&
                   active_rotation_at(program.instructions[end]) != nullptr) {
                ++end;
            }
            const std::size_t count = end - idx;
            if (count >= 2) {
                append_rotation_run(out, program, idx, count);
                idx = end;
                continue;
            }
        }

        std::visit(
            [&](const auto& instruction) {
                append_instruction(out, instruction, expression_plan, idx);
            },
            program.instructions[idx]);
        ++idx;
    }
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(
    const ApplyPrecomputedActivePauliRotation& instruction) {
    return instruction.sign_plan;
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(
    const PromoteDormantRotation& instruction) {
    return instruction.sign_plan;
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(
    const RecordMeasurement& instruction) {
    return instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(
    const RecordDetector& instruction) {
    return instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(
    const MeasurePrecomputedActivePauli& instruction) {
    return instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(
    const IntroduceDormantMeasurementBranch& instruction) {
    return instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan& full_instruction_expression_plan(const FactoredInstruction& instruction) {
    return std::visit(
        [](const auto& inst) -> const SymbolicBoolEvaluationPlan& {
            return full_instruction_expression_plan(inst);
        },
        instruction);
}

void prepare_condition_sampler_refs(CudaProgramData& out, int nsymbols) {
    out.condition_sampler_refs.assign(static_cast<std::size_t>(nsymbols), CudaConditionSamplerRef{});
    out.sampler_count = 0;
}

void assign_condition_sampler(
    CudaProgramData& out,
    int condition,
    CudaSamplerKind kind,
    std::int32_t index,
    std::int32_t sampler_id,
    std::int32_t condition_offset) {
    if (condition <= 0 || condition > out.symbol_count) {
        throw Error("CUDA exogenous sampler references an out-of-range condition");
    }
    auto& ref = out.condition_sampler_refs[static_cast<std::size_t>(condition - 1)];
    if (ref.kind != static_cast<std::int32_t>(CudaSamplerKind::None)) {
        throw Error("CUDA exogenous condition is assigned by more than one sampler");
    }
    ref.kind = static_cast<std::int32_t>(kind);
    ref.index = index;
    ref.sampler_id = sampler_id;
    ref.condition_offset = condition_offset;
}

void append_sampling_data(CudaProgramData& out, const FactoredInstructionProgram& program) {
    prepare_condition_sampler_refs(out, program.nsymbols);

    for (const auto& distribution : program.sampled_categorical_distributions) {
        const auto distribution_index = checked_i32(out.categorical_distributions.size(), "categorical distribution index");
        const auto sampler_id = out.sampler_count++;
        CudaCategoricalDistribution gpu;
        gpu.sampler_id = sampler_id;
        gpu.nbits = distribution.nbits;
        gpu.condition_offset = append_conditions(out, distribution.conditions);
        gpu.row_count = checked_i32(distribution.assignments.size(), "categorical row count");
        gpu.assignment_offset = append_assignments(out, distribution.assignments, distribution.nbits);
        gpu.probability_offset = append_probabilities(out, distribution.probabilities);
        for (std::size_t condition_idx = 0; condition_idx < distribution.conditions.size(); ++condition_idx) {
            assign_condition_sampler(
                out,
                distribution.conditions[condition_idx],
                CudaSamplerKind::Categorical,
                distribution_index,
                sampler_id,
                checked_i32(condition_idx, "categorical condition bit index"));
        }
        out.categorical_distributions.push_back(gpu);
    }

    for (const auto& group : program.sampled_rare_categorical_groups) {
        const auto group_index = checked_i32(out.rare_categorical_groups.size(), "rare categorical group index");
        const auto sampler_id = out.sampler_count++;
        CudaRareCategoricalGroup gpu;
        gpu.event_probability = group.event_probability;
        gpu.inverse_log_survival = group.event_probability > 0.0 ? 1.0 / std::log1p(-group.event_probability) : 0.0;
        gpu.sampler_id = sampler_id;
        gpu.nbits = group.nbits;
        gpu.set_count = checked_i32(group.conditions.size(), "rare categorical set count");
        std::vector<int> flat_conditions;
        flat_conditions.reserve(group.conditions.size() * static_cast<std::size_t>(group.nbits));
        for (std::size_t set_idx = 0; set_idx < group.conditions.size(); ++set_idx) {
            const auto& conditions = group.conditions[set_idx];
            if (static_cast<int>(conditions.size()) != group.nbits) {
                throw Error("rare categorical condition count does not match nbits");
            }
            for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
                assign_condition_sampler(
                    out,
                    conditions[bit_idx],
                    CudaSamplerKind::RareCategorical,
                    group_index,
                    sampler_id,
                    checked_i32(
                        set_idx * static_cast<std::size_t>(group.nbits) + bit_idx,
                        "rare categorical condition offset"));
            }
            flat_conditions.insert(flat_conditions.end(), conditions.begin(), conditions.end());
        }
        gpu.condition_offset = append_conditions(out, flat_conditions);
        gpu.row_count = checked_i32(group.assignments.size(), "rare categorical row count");
        gpu.assignment_offset = append_assignments(out, group.assignments, group.nbits);
        gpu.event_count = checked_i32(group.event_rows.size(), "rare categorical event count");
        gpu.event_row_offset = checked_i32(out.sample_event_row_table.size(), "event row table");
        out.sample_event_row_table.insert(
            out.sample_event_row_table.end(),
            group.event_rows.begin(),
            group.event_rows.end());
        gpu.event_probability_offset = append_probabilities(out, group.event_probabilities);
        out.rare_categorical_groups.push_back(gpu);
    }

    for (std::size_t idx = 0; idx < program.sampled_bernoulli_conditions.size(); ++idx) {
        const auto sampler_id = out.sampler_count++;
        const auto condition = program.sampled_bernoulli_conditions[idx];
        out.bernoulli_conditions.push_back(CudaBernoulliCondition{
            condition,
            sampler_id,
            program.sampled_bernoulli_probabilities[idx],
        });
        assign_condition_sampler(
            out,
            condition,
            CudaSamplerKind::Bernoulli,
            checked_i32(idx, "Bernoulli condition index"),
            sampler_id,
            0);
    }

    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        const auto group_index = checked_i32(
            out.low_probability_bernoulli_groups.size(),
            "low-probability Bernoulli group index");
        const auto sampler_id = out.sampler_count++;
        CudaBernoulliGroup gpu;
        gpu.probability = group.probability;
        gpu.inverse_log_survival = group.probability > 0.0 ? 1.0 / std::log1p(-group.probability) : 0.0;
        gpu.sampler_id = sampler_id;
        gpu.condition_offset = append_conditions(out, group.conditions);
        gpu.condition_count = checked_i32(group.conditions.size(), "low-probability Bernoulli group size");
        for (std::size_t condition_idx = 0; condition_idx < group.conditions.size(); ++condition_idx) {
            assign_condition_sampler(
                out,
                group.conditions[condition_idx],
                CudaSamplerKind::LowProbabilityBernoulli,
                group_index,
                sampler_id,
                checked_i32(condition_idx, "low-probability Bernoulli condition index"));
        }
        out.low_probability_bernoulli_groups.push_back(gpu);
    }
}

void append_logical_records(CudaProgramData& out, const std::vector<std::vector<int>>& logical_records) {
    out.logical_group_offsets.reserve(logical_records.size());
    out.logical_group_sizes.reserve(logical_records.size());
    for (const auto& group : logical_records) {
        out.logical_group_offsets.push_back(append_records(out, group));
        out.logical_group_sizes.push_back(checked_i32(group.size(), "logical record group size"));
    }
}

} // namespace

CudaProgramData build_cuda_program_data(
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const std::vector<std::vector<int>>& logical_records) {
    if (expression_plan.instruction_expressions.size() != program.instructions.size()) {
        throw Error("CUDA expression plan does not match instruction program");
    }

    CudaProgramData out;
    out.symbol_count = program.nsymbols;
    out.initial_k = program.initial_k;
    out.max_k = program.max_k;
    out.symbol_words = checked_i32(symbol_word_count(program.nsymbols), "symbol word count");
    out.record_words = checked_i32(symbol_word_count(program.nrecords), "record word count");
    out.block_expression_count = checked_i32(
        expression_plan.block_expressions.size(),
        "block expression count");
    append_block_expression_plans(out, expression_plan);

    out.instructions.reserve(program.instructions.size());
    out.expressions.reserve(expression_plan.instruction_expressions.size());
    for (const auto& expression : expression_plan.instruction_expressions) {
        append_expression(out, expression);
    }

    append_instruction_stream(out, program, expression_plan);

    append_logical_records(out, logical_records);
    append_sampling_data(out, program);

    return out;
}

CudaProgramData build_cuda_program_data_device_exogenous(
    const FactoredInstructionProgram& program,
    const std::vector<std::vector<int>>& logical_records) {
    CudaProgramData out;
    out.symbol_count = program.nsymbols;
    out.initial_k = program.initial_k;
    out.max_k = program.max_k;
    out.symbol_words = checked_i32(symbol_word_count(program.nsymbols), "symbol word count");
    out.record_words = checked_i32(symbol_word_count(program.nrecords), "record word count");
    out.block_expression_count = 0;

    out.instructions.reserve(program.instructions.size());
    out.expressions.reserve(program.instructions.size());
    for (const auto& instruction : program.instructions) {
        append_full_expression(out, full_instruction_expression_plan(instruction));
    }

    PresampledExpressionPlan unused_plan;
    append_instruction_stream(out, program, unused_plan);

    append_logical_records(out, logical_records);
    append_sampling_data(out, program);
    return out;
}

} // namespace symft::cuda
