#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace symft {

enum class ActiveComponentStepKind : std::uint8_t {
    None,
    IgnoredGlobalPhase,
    Rotation,
    Promotion,
    Measurement,
};

struct ActiveComponentStepRef {
    ActiveComponentStepKind kind = ActiveComponentStepKind::None;
    std::uint32_t payload = 0;
};

struct ActiveComponentRotationStep {
    int component = -1;
    std::uint32_t merge_offset = 0;
    std::uint16_t merge_count = 0;
    PrecomputedActivePauliRotationKernel kernel;
};

struct ActiveComponentPromotionStep {
    int component = -1;
};

struct ActiveComponentMeasurementStep {
    int component = -1;
    std::uint32_t merge_offset = 0;
    std::uint16_t merge_count = 0;
    PrecomputedActivePauliMeasurementKernel kernel;
    bool deactivate_after = false;
};

struct ActiveComponentPlan {
    bool selected = false;
    int initial_components = 0;
    int component_count = 0;
    std::vector<int> component_max_k;
    std::vector<ActiveComponentStepRef> instruction_steps;
    std::vector<int> merge_components;
    std::vector<ActiveComponentRotationStep> rotations;
    std::vector<ActiveComponentPromotionStep> promotions;
    std::vector<ActiveComponentMeasurementStep> measurements;

    long double estimated_dense_work = 0.0L;
    long double estimated_component_work = 0.0L;
    std::size_t dense_peak_dimension = 1;
    std::size_t component_peak_live_dimension = 0;
    std::size_t component_allocated_dimension = 0;
};

std::shared_ptr<const ActiveComponentPlan> build_active_component_plan(
    const FactoredInstructionProgram& program);

} // namespace symft
