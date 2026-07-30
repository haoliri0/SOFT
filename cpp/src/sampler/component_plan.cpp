#include "sampler/component_plan.hpp"

#include "core/internal.hpp"
#include "sampler/active_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace symft {
namespace {

using detail::active_length;
using detail::fail;

constexpr long double kComponentDispatchWork = 32.0L;
constexpr long double kMinimumSavedWork = 8192.0L;
constexpr long double kRequiredWorkRatio = 1.8L;
constexpr long double kMaximumAllocationRatio = 1.25L;

struct PlanningComponent {
    bool active = false;
    std::vector<int> coordinates;
};

std::size_t saturating_add(std::size_t lhs, std::size_t rhs) {
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs) {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs + rhs;
}

std::size_t live_component_dimension(const std::vector<PlanningComponent>& components) {
    std::size_t total = 0;
    for (const auto& component : components) {
        if (component.active && !component.coordinates.empty()) {
            total = saturating_add(total, active_length(static_cast<int>(component.coordinates.size())));
        }
    }
    return total;
}

void refresh_coordinate_positions(
    const std::vector<int>& active_order,
    std::vector<int>& coordinate_positions) {
    std::fill(coordinate_positions.begin(), coordinate_positions.end(), -1);
    for (std::size_t position = 0; position < active_order.size(); ++position) {
        const int coordinate = active_order[position];
        if (coordinate < 0 || coordinate >= static_cast<int>(coordinate_positions.size())) {
            fail("component planner encountered an invalid active coordinate");
        }
        coordinate_positions[static_cast<std::size_t>(coordinate)] =
            static_cast<int>(position);
    }
}

ActivePauliAction remap_action(
    const ActivePauliAction& action,
    const std::vector<int>& component_coordinates,
    const std::vector<int>& coordinate_positions) {
    ActivePauliAction local = action;
    local.nqubits = static_cast<int>(component_coordinates.size());
    local.xmask = 0;
    local.zmask = 0;
    for (std::size_t local_position = 0;
         local_position < component_coordinates.size();
         ++local_position) {
        const int coordinate = component_coordinates[local_position];
        const int global_position =
            coordinate_positions[static_cast<std::size_t>(coordinate)];
        if (global_position < 0) {
            fail("component planner tried to remap an inactive coordinate");
        }
        const std::uint64_t global_bit =
            std::uint64_t{1} << static_cast<unsigned>(global_position);
        const std::uint64_t local_bit =
            std::uint64_t{1} << static_cast<unsigned>(local_position);
        if ((action.xmask & global_bit) != 0) {
            local.xmask |= local_bit;
        }
        if ((action.zmask & global_bit) != 0) {
            local.zmask |= local_bit;
        }
    }
    local.xz_overlap_odd =
        detail::is_odd_popcount(local.xmask & local.zmask);
    return local;
}

std::vector<int> touched_components(
    const ActivePauliAction& action,
    const std::vector<int>& active_order,
    const std::vector<int>& coordinate_components) {
    std::vector<int> touched;
    const std::uint64_t support = action.xmask | action.zmask;
    for (std::size_t position = 0; position < active_order.size(); ++position) {
        if ((support & (std::uint64_t{1} << position)) == 0) {
            continue;
        }
        const int coordinate = active_order[position];
        const int component =
            coordinate_components[static_cast<std::size_t>(coordinate)];
        if (component < 0) {
            fail("component planner found an active coordinate without a component");
        }
        if (std::find(touched.begin(), touched.end(), component) == touched.end()) {
            touched.push_back(component);
        }
    }
    // An identity action only contributes a global phase. Keep it local to an
    // arbitrary live component when possible so dense/component states agree
    // even before global phases are discarded by measurement.
    if (touched.empty() && !active_order.empty()) {
        touched.push_back(
            coordinate_components[static_cast<std::size_t>(active_order.front())]);
    }
    return touched;
}

int select_merge_target(
    const std::vector<int>& touched,
    const std::vector<PlanningComponent>& components) {
    if (touched.empty()) {
        return -1;
    }
    return *std::max_element(
        touched.begin(),
        touched.end(),
        [&](int lhs, int rhs) {
            const auto lhs_size =
                components[static_cast<std::size_t>(lhs)].coordinates.size();
            const auto rhs_size =
                components[static_cast<std::size_t>(rhs)].coordinates.size();
            if (lhs_size != rhs_size) {
                return lhs_size < rhs_size;
            }
            return lhs > rhs;
        });
}

std::vector<int> ordered_merge_sources(
    int target,
    std::vector<int> touched) {
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
    touched.erase(std::remove(touched.begin(), touched.end(), target), touched.end());
    touched.insert(touched.begin(), target);
    return touched;
}

long double merge_planning_components(
    const std::vector<int>& sources,
    std::vector<PlanningComponent>& components,
    std::vector<int>& coordinate_components,
    std::vector<int>& component_max_k) {
    if (sources.empty()) {
        return 0.0L;
    }
    const int target = sources.front();
    auto& target_component = components[static_cast<std::size_t>(target)];
    if (!target_component.active) {
        fail("component merge target is inactive");
    }
    long double work = 0.0L;
    int target_k = static_cast<int>(target_component.coordinates.size());
    for (std::size_t source_index = 1;
         source_index < sources.size();
         ++source_index) {
        const int source = sources[source_index];
        auto& source_component = components[static_cast<std::size_t>(source)];
        if (!source_component.active) {
            fail("component merge source is inactive");
        }
        target_k += static_cast<int>(source_component.coordinates.size());
        work += 2.0L * std::ldexp(1.0L, target_k);
        for (int coordinate : source_component.coordinates) {
            coordinate_components[static_cast<std::size_t>(coordinate)] = target;
            target_component.coordinates.push_back(coordinate);
        }
        source_component.coordinates.clear();
        source_component.active = false;
    }
    component_max_k[static_cast<std::size_t>(target)] =
        std::max(component_max_k[static_cast<std::size_t>(target)], target_k);
    return work;
}

std::pair<std::uint32_t, std::uint16_t> append_merge_sources(
    ActiveComponentPlan& plan,
    const std::vector<int>& sources) {
    if (sources.size() > std::numeric_limits<std::uint16_t>::max()) {
        fail("component merge has too many sources");
    }
    if (plan.merge_components.size() >
        std::numeric_limits<std::uint32_t>::max() - sources.size()) {
        fail("component merge table is too large");
    }
    const auto offset =
        static_cast<std::uint32_t>(plan.merge_components.size());
    plan.merge_components.insert(
        plan.merge_components.end(),
        sources.begin(),
        sources.end());
    return {offset, static_cast<std::uint16_t>(sources.size())};
}

void update_peak_live_dimension(
    ActiveComponentPlan& plan,
    const std::vector<PlanningComponent>& components) {
    plan.component_peak_live_dimension = std::max(
        plan.component_peak_live_dimension,
        live_component_dimension(components));
}

bool should_select_component_plan(
    const FactoredInstructionProgram& program,
    const ActiveComponentPlan& plan,
    int quantum_instruction_count) {
    if (program.max_k < 8 || quantum_instruction_count < 4) {
        return false;
    }
    if (!(plan.estimated_dense_work >=
          kRequiredWorkRatio * plan.estimated_component_work)) {
        return false;
    }
    if (!(plan.estimated_dense_work - plan.estimated_component_work >=
          kMinimumSavedWork)) {
        return false;
    }
    const long double allocation_limit =
        kMaximumAllocationRatio *
        static_cast<long double>(plan.dense_peak_dimension);
    return static_cast<long double>(plan.component_allocated_dimension) <=
           allocation_limit;
}

} // namespace

std::shared_ptr<const ActiveComponentPlan> build_active_component_plan(
    const FactoredInstructionProgram& program) {
    auto plan = std::make_shared<ActiveComponentPlan>();
    plan->initial_components = program.initial_k;
    plan->dense_peak_dimension = active_length(program.max_k);
    // Small active states are already cache-resident and cannot repay
    // component dispatch. Avoid even constructing per-instruction metadata for
    // the overwhelmingly common pure-Clifford/small-active fallback.
    if (program.max_k < 8) {
        return plan;
    }
    plan->instruction_steps.resize(program.instructions.size());

    std::vector<PlanningComponent> components;
    std::vector<int> active_order;
    std::vector<int> coordinate_components;
    std::vector<int> coordinate_positions;
    components.reserve(
        static_cast<std::size_t>(program.initial_k) +
        program.instructions.size());
    active_order.reserve(
        static_cast<std::size_t>(program.initial_k) +
        program.instructions.size());
    coordinate_components.reserve(active_order.capacity());
    coordinate_positions.reserve(active_order.capacity());

    for (int coordinate = 0; coordinate < program.initial_k; ++coordinate) {
        const int component = static_cast<int>(components.size());
        components.push_back(PlanningComponent{true, {coordinate}});
        active_order.push_back(coordinate);
        coordinate_components.push_back(component);
        coordinate_positions.push_back(coordinate);
        plan->component_max_k.push_back(1);
    }
    update_peak_live_dimension(*plan, components);

    int quantum_instruction_count = 0;
    for (std::size_t instruction_index = 0;
         instruction_index < program.instructions.size();
         ++instruction_index) {
        const auto& instruction = program.instructions[instruction_index];
        const int global_k = static_cast<int>(active_order.size());
        refresh_coordinate_positions(active_order, coordinate_positions);
        const long double dense_dim = std::ldexp(1.0L, global_k);

        if (const auto* rotation =
                std::get_if<ApplyPrecomputedActivePauliRotation>(&instruction)) {
            if (rotation->rotation_kernel.action.nqubits != global_k) {
                fail("component planner saw a rotation with the wrong active width");
            }
            ++quantum_instruction_count;
            plan->estimated_dense_work += dense_dim;
            auto touched = touched_components(
                rotation->rotation_kernel.action,
                active_order,
                coordinate_components);
            if (touched.empty()) {
                // A phase on the k=0 scalar has no observable consequence.
                plan->instruction_steps[instruction_index] = {
                    ActiveComponentStepKind::IgnoredGlobalPhase,
                    0,
                };
                continue;
            }
            const int target = select_merge_target(touched, components);
            const auto sources = ordered_merge_sources(target, std::move(touched));
            plan->estimated_component_work += merge_planning_components(
                sources,
                components,
                coordinate_components,
                plan->component_max_k);
            const auto [merge_offset, merge_count] =
                append_merge_sources(*plan, sources);
            const auto local_action = remap_action(
                rotation->rotation_kernel.action,
                components[static_cast<std::size_t>(target)].coordinates,
                coordinate_positions);
            ActiveComponentRotationStep step;
            step.component = target;
            step.merge_offset = merge_offset;
            step.merge_count = merge_count;
            step.kernel = PrecomputedActivePauliRotationKernel(
                local_action,
                rotation->kernel_angle);
            const auto payload =
                static_cast<std::uint32_t>(plan->rotations.size());
            plan->rotations.push_back(std::move(step));
            plan->instruction_steps[instruction_index] = {
                ActiveComponentStepKind::Rotation,
                payload,
            };
            plan->estimated_component_work +=
                std::ldexp(
                    1.0L,
                    static_cast<int>(
                        components[static_cast<std::size_t>(target)]
                            .coordinates.size())) +
                kComponentDispatchWork;
            update_peak_live_dimension(*plan, components);
            continue;
        }

        if (std::holds_alternative<PromoteDormantRotation>(instruction)) {
            ++quantum_instruction_count;
            plan->estimated_dense_work += 2.0L * dense_dim;
            const int coordinate =
                static_cast<int>(coordinate_components.size());
            const int component = static_cast<int>(components.size());
            components.push_back(PlanningComponent{true, {coordinate}});
            active_order.push_back(coordinate);
            coordinate_components.push_back(component);
            coordinate_positions.push_back(
                static_cast<int>(active_order.size()) - 1);
            plan->component_max_k.push_back(1);
            const auto payload =
                static_cast<std::uint32_t>(plan->promotions.size());
            plan->promotions.push_back(
                ActiveComponentPromotionStep{component});
            plan->instruction_steps[instruction_index] = {
                ActiveComponentStepKind::Promotion,
                payload,
            };
            plan->estimated_component_work +=
                2.0L + kComponentDispatchWork;
            update_peak_live_dimension(*plan, components);
            continue;
        }

        if (const auto* measurement =
                std::get_if<MeasurePrecomputedActivePauli>(&instruction)) {
            if (measurement->kernel.action.nqubits != global_k) {
                fail("component planner saw a measurement with the wrong active width");
            }
            ++quantum_instruction_count;
            plan->estimated_dense_work += 2.0L * dense_dim;
            auto touched = touched_components(
                measurement->kernel.action,
                active_order,
                coordinate_components);
            if (touched.empty()) {
                fail("active measurement has no component support");
            }
            const int target = select_merge_target(touched, components);
            const auto sources = ordered_merge_sources(target, std::move(touched));
            plan->estimated_component_work += merge_planning_components(
                sources,
                components,
                coordinate_components,
                plan->component_max_k);
            update_peak_live_dimension(*plan, components);
            const auto [merge_offset, merge_count] =
                append_merge_sources(*plan, sources);
            refresh_coordinate_positions(active_order, coordinate_positions);
            auto& target_coordinates =
                components[static_cast<std::size_t>(target)].coordinates;
            const auto local_action = remap_action(
                measurement->kernel.action,
                target_coordinates,
                coordinate_positions);
            const int global_pivot = measurement->kernel.pivot;
            if (global_pivot < 0 ||
                global_pivot >= static_cast<int>(active_order.size())) {
                fail("component planner saw an invalid measurement pivot");
            }
            const int pivot_coordinate =
                active_order[static_cast<std::size_t>(global_pivot)];
            const auto pivot_it = std::find(
                target_coordinates.begin(),
                target_coordinates.end(),
                pivot_coordinate);
            if (pivot_it == target_coordinates.end()) {
                fail("component measurement pivot is not in its target component");
            }
            const int local_pivot =
                static_cast<int>(pivot_it - target_coordinates.begin());
            ActiveComponentMeasurementStep step;
            step.component = target;
            step.merge_offset = merge_offset;
            step.merge_count = merge_count;
            step.kernel = PrecomputedActivePauliMeasurementKernel(
                local_action,
                local_pivot);
            plan->estimated_component_work +=
                2.0L *
                    std::ldexp(
                        1.0L,
                        static_cast<int>(target_coordinates.size())) +
                kComponentDispatchWork;

            target_coordinates.erase(pivot_it);
            active_order.erase(
                active_order.begin() + static_cast<std::ptrdiff_t>(global_pivot));
            coordinate_components[static_cast<std::size_t>(pivot_coordinate)] =
                -1;
            if (target_coordinates.empty()) {
                components[static_cast<std::size_t>(target)].active = false;
                step.deactivate_after = true;
            }
            const auto payload =
                static_cast<std::uint32_t>(plan->measurements.size());
            plan->measurements.push_back(std::move(step));
            plan->instruction_steps[instruction_index] = {
                ActiveComponentStepKind::Measurement,
                payload,
            };
            update_peak_live_dimension(*plan, components);
        }
    }

    plan->component_count = static_cast<int>(components.size());
    for (int max_k : plan->component_max_k) {
        plan->component_allocated_dimension = saturating_add(
            plan->component_allocated_dimension,
            active_length(max_k));
    }
    plan->selected = should_select_component_plan(
        program,
        *plan,
        quantum_instruction_count);
    return plan;
}

} // namespace symft
