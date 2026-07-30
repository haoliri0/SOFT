#pragma once

#include "factored/factored.hpp"
#include "sampler/presampled_expression.hpp"

#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous;
struct PackedPresampledExogenous;

struct SingleShotActiveComponent {
    int k = 0;
    bool active = false;
    std::vector<double> re;
    std::vector<double> im;
    std::vector<double> scratch_re;
    std::vector<double> scratch_im;
};

struct FactoredExecutorState {
    int n = 0;
    int k = 0;
    int ndormant = 0;
    int nsymbols = 0;
    int nrecords = 0;
    int ndetectors = 0;
    std::vector<double> active_re;
    std::vector<double> active_im;
    std::vector<double> active_scratch_re;
    std::vector<double> active_scratch_im;
    std::vector<std::uint64_t> value_words;
    std::vector<std::uint64_t> assigned_words;
    std::vector<std::uint64_t> measurement_words;
    std::vector<std::uint64_t> detector_words;
    bool active_components_enabled = false;
    std::vector<SingleShotActiveComponent> active_components;
    std::uint64_t rng_state = 1;

    explicit FactoredExecutorState(const FactoredInstructionProgram& program, std::uint64_t seed = 1);
};

void reset_executor(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    bool clear_detector_records = true);
void execute_in_place(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index);
void execute_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int shot_index);
std::vector<std::uint64_t> execute(FactoredExecutorState& runtime, const FactoredInstructionProgram& program);
std::vector<std::uint64_t> sample_measurements(const FactoredInstructionProgram& program, std::uint64_t seed = 1);
std::vector<std::vector<std::uint64_t>> sample_measurements(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);
void assign_presampled_exogenous_in_place(
    FactoredExecutorState& runtime,
    const PresampledExogenous& samples,
    int shot_index);
void execute_instruction_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstruction& instruction);
bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples,
    int shot_index);
bool execute_postselected_in_place(
    FactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int shot_index);
int default_single_shot_sample_chunk_shots();
std::vector<std::vector<std::uint64_t>> sample_measurements(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed,
    int sample_chunk_shots);
} // namespace symft
