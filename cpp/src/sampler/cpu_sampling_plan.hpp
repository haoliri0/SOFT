#pragma once

#include "sampler/presampled_expression.hpp"
#include "sampler/real_active.hpp"
#include <functional>
#include <utility>

namespace symft {

struct CpuExpression {
    bool constant = false;
    std::vector<int> noise;
    std::vector<std::pair<unsigned, std::uint64_t>> branch_masks;
    int packed_noise_index = -1;
};

enum class CpuOpKind { Rotation, Promotion, Measurement, DormantBranch };
struct CpuSamplingOp {
    CpuOpKind kind = CpuOpKind::DormantBranch;
    int instruction = 0;
    int k = 0;
    unsigned gauge = 0;
    int branch = -1;
    double c = 1, s = 0;
    CpuExpression sign;
    PrecomputedActivePauliRotationKernel rotation;
    PrecomputedActivePauliMeasurementKernel measurement;
    detail::RealRotationKernel real_rotation;
    detail::RealMeasurementKernel real_measurement;
};

// Internal, counts-only CPU plan. No CUDA headers or dependencies.
struct CpuSamplingPlan {
    int initial_k = 0, max_k = 0, branches = 0;
    unsigned final_gauge = 0;
    bool real_gauge = false;
    bool hoist = true;
    int noise_only_detectors = 0;
    std::vector<CpuSamplingOp> ops;
    std::vector<unsigned> live_ops;
    std::vector<unsigned> skipped_rng;
    std::vector<std::vector<CpuExpression>> live_checks;
    std::vector<CpuExpression> initial_checks;
    std::vector<std::vector<CpuExpression>> checks;
    std::vector<CpuExpression> detectors;
    std::vector<CpuExpression> records;
    CpuExpression logical;
    int source_noise_count = 0;
    std::vector<CpuExpression> noise_outputs;

    CpuSamplingPlan(const FactoredInstructionProgram& program,
                    const PresampledExpressionPlan& expressions,
                    const std::vector<std::vector<int>>& logical_records,
                    bool allow_real = true, bool hoist_detectors = true,
                    const std::vector<std::uint64_t>& expected_detector_words = {},
                    bool expected_observable = false);
};

struct CpuSamplingWorkspace {
    std::vector<double> re, im, scratch_re, scratch_im;
    std::vector<std::uint64_t> branches, rejected;
    PresampledExpressionBlock packed_noise;
    explicit CpuSamplingWorkspace(const CpuSamplingPlan& plan);
};

struct CpuShotResult { bool discarded = false, logical_error = false; };
struct CpuChunkResult {
    std::uint64_t shots = 0, discarded = 0, accepted = 0, logical_errors = 0;
};

// Debug replay callback: state AFTER an operation. probability is >=0 only
// for active measurements; branch is -1 for non-measurement operations.
using CpuTrace = std::function<void(const CpuSamplingOp&, const CpuSamplingWorkspace&, double, int)>;

bool evaluate_cpu_expression(const CpuExpression& expression,
    const PresampledExpressionBlock& noise, int shot, const std::vector<std::uint64_t>& branches);
std::uint64_t cpu_shot_seed(std::uint64_t stream, std::uint64_t shot);
CpuShotResult execute_cpu_shot(const CpuSamplingPlan& plan, CpuSamplingWorkspace& workspace,
    const PresampledExpressionBlock& noise, int shot, std::uint64_t seed,
    bool postselect = true, const CpuTrace* trace = nullptr, bool initial_checked = false);
CpuChunkResult execute_cpu_chunk(const CpuSamplingPlan& plan, CpuSamplingWorkspace& workspace,
    const PresampledExpressionBlock& noise, std::uint64_t stream, std::uint64_t first_shot);

} // namespace symft
