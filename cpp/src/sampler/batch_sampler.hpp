#pragma once

#include "factored/factored.hpp"
#include "sampler/presampled_expression.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous;
struct PackedPresampledExogenous;

struct BatchActiveComponent {
    int k = 0;
    bool active = false;
    std::size_t stride = 0;
    std::vector<double> re;
    std::vector<double> im;
    std::vector<double> scratch_re;
    std::vector<double> scratch_im;
};

struct BatchDetectorPostselectionScratch {
    std::vector<std::uint64_t> dead_bits;
    std::vector<std::uint64_t> keep_bits;
    std::vector<std::uint64_t> scratch;
    std::vector<std::uint64_t> compact_scratch;
    std::vector<std::uint64_t> expression_words;
    std::vector<int> live_sources;
    std::vector<int> condition_last_use_by_index;
    std::vector<int> record_last_use_by_index;
    const FactoredInstructionProgram* postselection_metadata_program = nullptr;
    const std::vector<std::vector<int>>* postselection_metadata_retained_record_uses = nullptr;
    int dead_count = 0;
};

struct BatchDetectorPostselectionResult {
    int discarded = 0;
    int accepted = 0;
};

struct BatchDetectorPostselectionOptions {
    int mask_dead_shots_min_fraction_denominator = 2;
    const std::vector<std::vector<int>>* retained_record_uses = nullptr;
};

// The prepared batch sampler uses shot-major active storage:
// active_re[shot * active_stride + basis].
// Basis-major storage is still supported by lower-level helpers:
// active_re[basis * active_pitch + shot].
// active_pitch remains the padded shot-column capacity for packed batch lanes.
struct BatchFactoredExecutorState {
    int n = 0;
    int k = 0;
    int ndormant = 0;
    int batches = 0;
    int active_shots = 0;
    int active_pitch = 0;
    std::size_t active_stride = 0;
    int nsymbols = 0;
    int nrecords = 0;
    int ndetectors = 0;
    int max_k = 0;
    std::size_t batch_words = 0;
    bool store_detector_records = true;
    bool dense_shot_major_active = false;
    std::vector<double> active_re;
    std::vector<double> active_im;
    std::vector<double> scratch_re;
    std::vector<double> scratch_im;
    std::vector<std::uint64_t> value_words;
    std::vector<std::uint64_t> assigned_words;
    std::vector<std::uint64_t> measurement_words;
    std::vector<std::uint64_t> detector_words;
    std::vector<std::uint64_t> detector_any_words;
    std::vector<std::uint64_t> eval_scratch;
    std::vector<std::uint64_t> rotation_run_sign_words;
    // Per-shot scalar directions; O(active_pitch), never O(2^k).
    std::vector<double> shot_coefficient_scalars;
    std::vector<double> branch_prob_true;
    std::vector<double> branch_invnorms;
    bool active_components_enabled = false;
    std::vector<BatchActiveComponent> active_components;
    std::uint64_t rng_state = 1;

    explicit BatchFactoredExecutorState(
        const FactoredInstructionProgram& program,
        int batches = 0,
        std::uint64_t seed = 1);
};

int default_batch_count(int max_k);
void reset_batch_executor(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    int shots,
    bool clear_symbol_values = true);
void execute_batch_in_place(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program);
void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExogenous& samples);
void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PackedPresampledExogenous& samples,
    int first_sample_shot = 0);
void execute_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot = 0);
void prepare_batch_detector_postselection_scratch(
    BatchDetectorPostselectionScratch& scratch,
    const BatchFactoredExecutorState& runtime);
void prepare_batch_detector_postselection_scratch(
    BatchDetectorPostselectionScratch& scratch,
    const BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const BatchDetectorPostselectionOptions& options = {});
BatchDetectorPostselectionResult execute_batch_postselected_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const PresampledExpressionBlock& expression_block,
    int first_sample_shot,
    BatchDetectorPostselectionScratch& scratch,
    const BatchDetectorPostselectionOptions& options = {});
// Compatibility query for the prepared batch sampler's dense-vector backend.
// Prepared batches use the same runtime-dispatched kernels as single-shot.
const char* active_batch_backend();
std::vector<std::vector<std::uint64_t>> sample_measurements_batch(
    const FactoredInstructionProgram& program,
    int shots,
    int batches = 0,
    std::uint64_t seed = 1);
void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples);
void assign_presampled_exogenous_batch_in_place(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int first_sample_shot = 0);
void execute_batch_instruction_in_place(
    BatchFactoredExecutorState& runtime,
    const FactoredInstruction& instruction);

} // namespace symft
