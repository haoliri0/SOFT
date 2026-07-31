#include "sampler/presampled_expression.hpp"

#include "core/internal.hpp"
#include "sampler/exogenous.hpp"

#include <algorithm>
#include <limits>
#include <utility>
#include <variant>

namespace symft {
namespace {

using detail::fail;
using detail::symbol_bit_mask;
using detail::symbol_word_count;
using detail::symbol_word_index;

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_shot_word_mask(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

PresampledExpression split_presampled_expression(
    const SymbolicBoolEvaluationPlan& source,
    const std::vector<std::uint64_t>& exogenous_assigned_words) {
    PresampledExpression out;
    out.constant = source.constant;
    std::vector<int> residual_conditions;
    residual_conditions.reserve(source.conditions.size());
    for (int condition : source.conditions) {
        const std::size_t word = symbol_word_index(condition);
        const std::uint64_t mask = symbol_bit_mask(condition);
        const bool exogenous =
            word < exogenous_assigned_words.size() &&
            (exogenous_assigned_words[word] & mask) != 0;
        if (exogenous) {
            out.exogenous_conditions.push_back(condition);
        } else {
            residual_conditions.push_back(condition);
        }
    }
    out.residual_plan = SymbolicBoolEvaluationPlan(SymbolicBool(false, std::move(residual_conditions)));
    return out;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const ApplyPrecomputedActivePauliRotation& instruction) {
    return &instruction.sign_plan;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const PromoteDormantRotation& instruction) {
    return &instruction.sign_plan;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const RecordMeasurement& instruction) {
    return &instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const RecordDetector& instruction) {
    if (!instruction.records.empty() || instruction.outcome.conditions.empty()) {
        return nullptr;
    }
    return &instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const MeasurePrecomputedActivePauli& instruction) {
    return &instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const IntroduceDormantMeasurementBranch& instruction) {
    return &instruction.outcome_plan;
}

const SymbolicBoolEvaluationPlan* instruction_expression_plan(const FactoredInstruction& instruction) {
    return std::visit([](const auto& inst) { return instruction_expression_plan(inst); }, instruction);
}

bool same_exogenous_partial(
    const PresampledExpression& lhs,
    const PresampledExpression& rhs) {
    return lhs.constant == rhs.constant &&
           lhs.exogenous_conditions == rhs.exogenous_conditions;
}

int intern_exogenous_partial(
    std::vector<PresampledExpression>& block_expressions,
    const PresampledExpression& expression) {
    for (std::size_t idx = 0; idx < block_expressions.size(); ++idx) {
        if (same_exogenous_partial(block_expressions[idx], expression)) {
            return static_cast<int>(idx);
        }
    }
    PresampledExpression block_expression;
    block_expression.constant = expression.constant;
    block_expression.exogenous_conditions = expression.exogenous_conditions;
    block_expressions.push_back(std::move(block_expression));
    return static_cast<int>(block_expressions.size() - 1);
}

std::vector<int> symmetric_difference_conditions(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    std::vector<int> out;
    out.reserve(lhs.size() + rhs.size());
    auto lit = lhs.begin();
    auto rit = rhs.begin();
    while (lit != lhs.end() || rit != rhs.end()) {
        if (rit == rhs.end() || (lit != lhs.end() && *lit < *rit)) {
            out.push_back(*lit++);
        } else if (lit == lhs.end() || *rit < *lit) {
            out.push_back(*rit++);
        } else {
            ++lit;
            ++rit;
        }
    }
    return out;
}

void prepare_block_expression_parent_deltas(std::vector<PresampledExpression>& block_expressions) {
    for (std::size_t expression_index = 0; expression_index < block_expressions.size(); ++expression_index) {
        auto& expression = block_expressions[expression_index];
        expression.parent_block_expression_index = -1;
        expression.parent_delta_constant = expression.constant;
        expression.parent_delta_exogenous_conditions = expression.exogenous_conditions;

        std::size_t best_cost =
            expression.exogenous_conditions.size() + (expression.constant ? 1U : 0U);
        int best_parent = -1;
        bool best_delta_constant = expression.constant;
        std::vector<int> best_delta_conditions = expression.exogenous_conditions;
        for (std::size_t parent_index = 0; parent_index < expression_index; ++parent_index) {
            const auto& parent = block_expressions[parent_index];
            auto delta_conditions = symmetric_difference_conditions(
                parent.exogenous_conditions,
                expression.exogenous_conditions);
            const bool delta_constant = parent.constant != expression.constant;
            const std::size_t cost = delta_conditions.size() + (delta_constant ? 1U : 0U);
            if (cost < best_cost) {
                best_cost = cost;
                best_parent = static_cast<int>(parent_index);
                best_delta_constant = delta_constant;
                best_delta_conditions = std::move(delta_conditions);
            }
        }
        expression.parent_block_expression_index = best_parent;
        expression.parent_delta_constant = best_delta_constant;
        expression.parent_delta_exogenous_conditions = std::move(best_delta_conditions);
    }
}

void prepare_presampled_expression_plan(
    PresampledExpressionPlan& out,
    const FactoredInstructionProgram& program,
    const std::vector<std::uint64_t>& exogenous_assigned_words) {
    out.instruction_expressions.clear();
    out.block_expressions.clear();
    out.block_expression_last_use_by_index.clear();
    out.instruction_expressions.reserve(program.instructions.size());
    for (std::size_t instruction_index = 0; instruction_index < program.instructions.size(); ++instruction_index) {
        const auto& instruction = program.instructions[instruction_index];
        const auto* source = instruction_expression_plan(instruction);
        PresampledExpression expression;
        if (source == nullptr) {
            expression = PresampledExpression();
        } else {
            expression = split_presampled_expression(*source, exogenous_assigned_words);
        }
        expression.block_expression_index =
            intern_exogenous_partial(out.block_expressions, expression);
        const std::size_t block_index = static_cast<std::size_t>(expression.block_expression_index);
        if (out.block_expression_last_use_by_index.size() < out.block_expressions.size()) {
            out.block_expression_last_use_by_index.resize(out.block_expressions.size(), -1);
        }
        out.block_expression_last_use_by_index[block_index] =
            static_cast<int>(instruction_index);
        out.instruction_expressions.push_back(std::move(expression));
    }
    prepare_block_expression_parent_deltas(out.block_expressions);
}

} // namespace

std::size_t presampled_expression_block_offset(
    const PresampledExpressionBlock& block,
    std::size_t expression_index,
    std::size_t shot_word) {
    return expression_index * block.shot_words + shot_word;
}

bool presampled_expression_block_bit(
    const PresampledExpressionBlock& block,
    int block_expression_index,
    int shot_index) {
    const std::size_t word = static_cast<std::size_t>(shot_index >> 6);
    const std::uint64_t mask = std::uint64_t{1} << (shot_index & 63);
    return (block.expression_words[presampled_expression_block_offset(
                block,
                static_cast<std::size_t>(block_expression_index),
                word)] & mask) != 0;
}

void prepare_presampled_expression_plan(
    PresampledExpressionPlan& out,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples) {
    if (samples.nsymbols != program.nsymbols ||
        samples.exogenous_assigned_words.size() != symbol_word_count(program.nsymbols)) {
        fail("packed presampled exogenous storage was not prepared for this program");
    }
    prepare_presampled_expression_plan(out, program, samples.exogenous_assigned_words);
}

void evaluate_presampled_expression_block(
    PresampledExpressionBlock& out,
    const PresampledExpressionPlan& plan,
    const PackedPresampledExogenous& samples) {
    if (samples.value_words.size() != static_cast<std::size_t>(samples.nsymbols) * samples.shot_words) {
        fail("packed presampled exogenous values are not initialized");
    }
    out.nshots = samples.nshots;
    out.shot_words = samples.shot_words;
    const std::size_t nexpressions = plan.block_expressions.size();
    const std::size_t total_words = nexpressions * out.shot_words;
    if (out.expression_words.size() != total_words) {
        out.expression_words.resize(total_words, 0);
    }
    for (std::size_t expression_index = 0; expression_index < nexpressions; ++expression_index) {
        const auto& expression = plan.block_expressions[expression_index];
        const std::size_t expression_base = presampled_expression_block_offset(out, expression_index, 0);
        if (expression.parent_block_expression_index >= 0) {
            if (expression.parent_block_expression_index >= static_cast<int>(expression_index)) {
                fail("presampled expression parent must be earlier in the block");
            }
            const std::size_t parent_base = presampled_expression_block_offset(
                out,
                static_cast<std::size_t>(expression.parent_block_expression_index),
                0);
            std::copy_n(
                out.expression_words.data() + parent_base,
                out.shot_words,
                out.expression_words.data() + expression_base);
            if (expression.parent_delta_constant) {
                for (std::size_t shot_word = 0; shot_word < out.shot_words; ++shot_word) {
                    out.expression_words[expression_base + shot_word] ^=
                        live_shot_word_mask(samples.nshots, shot_word);
                }
            }
            for (int condition : expression.parent_delta_exogenous_conditions) {
                if (condition <= 0 || condition > samples.nsymbols) {
                    fail("presampled expression references an out-of-range exogenous condition");
                }
                const std::size_t condition_base =
                    static_cast<std::size_t>(condition - 1) * samples.shot_words;
                for (std::size_t shot_word = 0; shot_word < out.shot_words; ++shot_word) {
                    out.expression_words[expression_base + shot_word] ^=
                        samples.value_words[condition_base + shot_word];
                }
            }
            continue;
        }
        for (std::size_t shot_word = 0; shot_word < out.shot_words; ++shot_word) {
            out.expression_words[expression_base + shot_word] =
                expression.constant ? live_shot_word_mask(samples.nshots, shot_word) : 0;
        }
        for (int condition : expression.exogenous_conditions) {
            if (condition <= 0 || condition > samples.nsymbols) {
                fail("presampled expression references an out-of-range exogenous condition");
            }
            const std::size_t condition_base =
                static_cast<std::size_t>(condition - 1) * samples.shot_words;
            for (std::size_t shot_word = 0; shot_word < out.shot_words; ++shot_word) {
                out.expression_words[expression_base + shot_word] ^=
                    samples.value_words[condition_base + shot_word];
            }
        }
    }
}

} // namespace symft
