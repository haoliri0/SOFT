#pragma once

#include "sampler/prepared_sampler.hpp"
#include "sampler/presampled_expression.hpp"

#include <cstdint>
#include <vector>

namespace symft::cuda {

#ifdef SYMFT_CUDA_REAL_DOUBLE
using CudaReal = double;
#else
using CudaReal = float;
#endif

enum class CudaInstructionKind : std::int32_t {
    ActiveRotation = 0,
    PromoteDormantRotation = 1,
    RecordMeasurement = 2,
    RecordDetector = 3,
    ActiveMeasurement = 4,
    IntroduceDormantBranch = 5,
    ActiveRotationRun = 6,
};

struct CudaComplex {
    CudaReal re = 0.0f;
    CudaReal im = 0.0f;
};

struct CudaWordMask {
    std::int32_t word = 0;
    std::uint64_t mask = 0;
};

enum class CudaRotationKernelKind : std::int32_t {
    Diagonal = 0,
    UniformImagPairs = 1,
    RealPairFlip = 2,
    GeneralPair = 3,
};

struct CudaExpression {
    std::int32_t block_expression = 0;
    std::int32_t residual_offset = 0;
    std::int32_t residual_count = 0;
    std::int32_t residual_constant = 0;
};

struct CudaBlockExpression {
    std::int32_t parent = -1;
    std::int32_t constant = 0;
    std::int32_t parent_delta_constant = 0;
    std::int32_t condition_offset = 0;
    std::int32_t condition_count = 0;
    std::int32_t delta_condition_offset = 0;
    std::int32_t delta_condition_count = 0;
    std::int32_t mask_offset = 0;
    std::int32_t mask_count = 0;
};

struct CudaRotationKernel {
    CudaRotationKernelKind kind = CudaRotationKernelKind::GeneralPair;
    std::int32_t is_diagonal = 0;
    std::int32_t uniform_imag_pairs = 0;
    std::int32_t real_pair_flip = 0;
    std::int32_t xz_overlap_odd = 0;
    std::int32_t pair_bit = 0;
    std::int32_t pair_count = 0;
    std::uint64_t xmask = 0;
    std::uint64_t zmask = 0;
    CudaReal minus_even_re = 0.0f;
    CudaReal minus_even_im = 0.0f;
    CudaReal cos_angle = 1.0f;
};

struct CudaMeasurementKernel {
    std::int32_t is_diagonal = 0;
    std::int32_t pivot = 0;
    std::int32_t diagonal_phase_bit = 0;
    std::int32_t out_dim = 0;
    std::uint64_t xmask = 0;
    std::uint64_t zmask = 0;
    CudaReal even_phase_re = 1.0f;
    CudaReal even_phase_im = 0.0f;
};

struct CudaInstruction {
    CudaInstructionKind kind = CudaInstructionKind::RecordMeasurement;
    std::int32_t expression = 0;
    std::int32_t rotation = -1;
    std::int32_t measurement = -1;
    std::int32_t branch_condition = 0;
    std::int32_t record = 0;
    std::int32_t record_condition = 0;
    std::int32_t detector = 0;
    std::int32_t record_list_offset = 0;
    std::int32_t record_list_count = 0;
    std::int32_t rotation_run_offset = 0;
    std::int32_t rotation_run_count = 0;
    CudaReal kernel_angle = 0.0f;
    CudaReal kernel_cos_angle = 1.0f;
    CudaReal kernel_sin_angle = 0.0f;
};

struct CudaRotationRunItem {
    std::int32_t expression = 0;
    std::int32_t rotation = 0;
};

struct CudaBernoulliCondition {
    std::int32_t condition = 0;
    std::int32_t sampler_id = -1;
    double probability = 0.0;
};

struct CudaBernoulliGroup {
    double probability = 0.0;
    double inverse_log_survival = 0.0;
    std::int32_t sampler_id = -1;
    std::int32_t condition_offset = 0;
    std::int32_t condition_count = 0;
};

struct CudaCategoricalDistribution {
    std::int32_t sampler_id = -1;
    std::int32_t nbits = 0;
    std::int32_t condition_offset = 0;
    std::int32_t row_count = 0;
    std::int32_t assignment_offset = 0;
    std::int32_t probability_offset = 0;
};

struct CudaRareCategoricalGroup {
    double event_probability = 0.0;
    double inverse_log_survival = 0.0;
    std::int32_t sampler_id = -1;
    std::int32_t nbits = 0;
    std::int32_t set_count = 0;
    std::int32_t condition_offset = 0;
    std::int32_t row_count = 0;
    std::int32_t assignment_offset = 0;
    std::int32_t event_count = 0;
    std::int32_t event_row_offset = 0;
    std::int32_t event_probability_offset = 0;
};

enum class CudaSamplerKind : std::int32_t {
    None = 0,
    Categorical = 1,
    RareCategorical = 2,
    Bernoulli = 3,
    LowProbabilityBernoulli = 4,
};

struct CudaConditionSamplerRef {
    std::int32_t kind = static_cast<std::int32_t>(CudaSamplerKind::None);
    std::int32_t index = -1;
    std::int32_t sampler_id = -1;
    std::int32_t condition_offset = -1;
};

struct CudaProgramData {
    std::int32_t symbol_count = 0;
    std::int32_t initial_k = 0;
    std::int32_t max_k = 0;
    std::int32_t symbol_words = 0;
    std::int32_t record_words = 0;
    std::int32_t block_expression_count = 0;
    std::int32_t max_rotation_run_length = 0;
    std::int32_t sampler_count = 0;

    std::vector<CudaInstruction> instructions;
    std::vector<CudaRotationRunItem> rotation_run_items;
    std::vector<CudaExpression> expressions;
    std::vector<CudaBlockExpression> block_expression_plans;
    std::vector<CudaWordMask> block_expression_masks;
    std::vector<CudaWordMask> residual_masks;
    std::vector<CudaRotationKernel> rotations;
    std::vector<CudaMeasurementKernel> measurements;
    std::vector<CudaComplex> complex_table;
    std::vector<CudaReal> real_table;
    std::vector<std::int32_t> source_table;
    std::vector<std::int32_t> record_table;
    std::vector<std::int32_t> block_expression_condition_table;
    std::vector<std::int32_t> logical_group_offsets;
    std::vector<std::int32_t> logical_group_sizes;
    // Flattened exogenous sampling plan used by the persistent kernel. These
    // tables mirror FactoredInstructionProgram's grouped low-entropy samplers.
    std::vector<CudaCategoricalDistribution> categorical_distributions;
    std::vector<CudaRareCategoricalGroup> rare_categorical_groups;
    std::vector<CudaBernoulliCondition> bernoulli_conditions;
    std::vector<CudaBernoulliGroup> low_probability_bernoulli_groups;
    std::vector<std::int32_t> sample_condition_table;
    std::vector<std::uint64_t> sample_assignment_table;
    std::vector<double> sample_probability_table;
    std::vector<std::int32_t> sample_event_row_table;
    std::vector<CudaConditionSamplerRef> condition_sampler_refs;
};

CudaProgramData build_cuda_program_data(
    const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expression_plan,
    const std::vector<std::vector<int>>& logical_records);
CudaProgramData build_cuda_program_data_device_exogenous(
    const FactoredInstructionProgram& program,
    const std::vector<std::vector<int>>& logical_records);

} // namespace symft::cuda
