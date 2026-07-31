#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PackedPresampledExogenous;

struct PresampledExpression {
    bool constant = false;
    int block_expression_index = 0;
    int parent_block_expression_index = -1;
    bool parent_delta_constant = false;
    SymbolicBoolEvaluationPlan residual_plan;
    std::vector<int> exogenous_conditions;
    std::vector<int> parent_delta_exogenous_conditions;
};

struct PresampledExpressionPlan {
    std::vector<PresampledExpression> instruction_expressions;
    std::vector<PresampledExpression> block_expressions;
    std::vector<int> block_expression_last_use_by_index;
};

struct PresampledExpressionBlock {
    int nshots = 0;
    std::size_t shot_words = 0;
    std::vector<std::uint64_t> expression_words;
};

void prepare_presampled_expression_plan(
    PresampledExpressionPlan& out,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples);
void evaluate_presampled_expression_block(
    PresampledExpressionBlock& out,
    const PresampledExpressionPlan& plan,
    const PackedPresampledExogenous& samples);
std::size_t presampled_expression_block_offset(
    const PresampledExpressionBlock& block,
    std::size_t expression_index,
    std::size_t shot_word);
bool presampled_expression_block_bit(
    const PresampledExpressionBlock& block,
    int block_expression_index,
    int shot_index);

} // namespace symft
