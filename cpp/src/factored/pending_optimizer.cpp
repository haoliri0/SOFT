#include "factored/factored.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace symft {
namespace {

struct CanonicalRotation {
    PauliString body;
    SymbolicBool sign;
};

CanonicalRotation canonical_rotation(const PendingPauliRotation& rotation) {
    if (!pauli_squares_to_identity(rotation.pauli.pauli)) {
        detail::fail("pending Pauli rotation must have a Hermitian generator");
    }
    PauliString body = rotation.pauli.pauli;
    const SymbolicBool sign = xor_bool(rotation.pauli.sign, measurement_phase_sign(body));
    body.set_phase(pauli_body_y_count(body));
    return {std::move(body), sign};
}

bool pauli_bodies_commute(const PauliString& lhs, const PauliString& rhs) {
    return !pauli_anticommutes(lhs, rhs);
}

bool rotation_can_cross(
    const CanonicalRotation& rotation,
    const PendingOperation& operation) {
    if (const auto* other = std::get_if<PendingPauliRotation>(&operation)) {
        return pauli_bodies_commute(rotation.body, other->pauli.pauli);
    }
    if (const auto* measurement = std::get_if<PendingPauliMeasurement>(&operation)) {
        return pauli_bodies_commute(rotation.body, measurement->pauli.pauli);
    }
    // A classical record has no quantum action. Fusion keeps all record events
    // in their original relative order and moves the earlier rotation later,
    // so it cannot introduce an assignment-before-use dependency.
    if (std::holds_alternative<PendingClassicalRecord>(operation)) {
        return true;
    }
    detail::fail("unsupported pending operation in commutation optimizer");
}

bool try_fuse_rotations(
    const PendingPauliRotation& earlier,
    const PendingPauliRotation& later,
    PendingPauliRotation& fused,
    bool& cancelled) {
    const CanonicalRotation a = canonical_rotation(earlier);
    const CanonicalRotation b = canonical_rotation(later);
    if (!a.body.same_body(b.body) || a.sign.conditions != b.sign.conditions) {
        return false;
    }

    const double earlier_direction = a.sign.constant == b.sign.constant ? 1.0 : -1.0;
    const double angle = later.kernel_angle + earlier_direction * earlier.kernel_angle;
    cancelled = angle == 0.0;
    fused = PendingPauliRotation{
        angle,
        SymbolicPauliString(std::move(b.body), b.sign),
    };
    return true;
}

void fuse_commuting_rotations(
    std::vector<PendingOperation>& operations,
    PendingOptimizationStats& stats) {
    std::vector<bool> deleted(operations.size(), false);
    for (std::size_t i = 0; i < operations.size(); ++i) {
        if (deleted[i]) {
            continue;
        }
        const auto* earlier = std::get_if<PendingPauliRotation>(&operations[i]);
        if (earlier == nullptr) {
            continue;
        }
        const CanonicalRotation earlier_canonical = canonical_rotation(*earlier);
        for (std::size_t j = i + 1; j < operations.size(); ++j) {
            if (deleted[j]) {
                continue;
            }
            if (!rotation_can_cross(earlier_canonical, operations[j])) {
                break;
            }
            auto* later = std::get_if<PendingPauliRotation>(&operations[j]);
            if (later == nullptr) {
                continue;
            }
            PendingPauliRotation fused;
            bool cancelled = false;
            if (!try_fuse_rotations(*earlier, *later, fused, cancelled)) {
                continue;
            }
            deleted[i] = true;
            ++stats.fused_rotations;
            if (cancelled) {
                deleted[j] = true;
                ++stats.cancelled_rotations;
            } else {
                operations[j] = std::move(fused);
            }
            break;
        }
    }

    std::size_t write = 0;
    for (std::size_t read = 0; read < operations.size(); ++read) {
        if (deleted[read]) {
            continue;
        }
        if (write != read) {
            operations[write] = std::move(operations[read]);
        }
        ++write;
    }
    operations.resize(write);
}

void move_measurements_earlier(
    std::vector<PendingOperation>& operations,
    PendingOptimizationStats& stats,
    bool allow_all_commuting) {
    // In a segment where fusion already removed rotations, moving commuting
    // measurements earlier can further reduce the active-coordinate lifetime.
    // Otherwise require the commuting run to contain the same Pauli body. Once
    // the measurement reaches that rotation, it acts only as a branch phase
    // and planning can discard it. This avoids pure schedule churn in segments
    // where reordering cannot otherwise demonstrate any removed quantum work.
    // Measurements never cross another measurement or classical record
    // producer, preserving record order and assignment-before-use.
    for (std::size_t i = 1; i < operations.size(); ++i) {
        const auto* measurement = std::get_if<PendingPauliMeasurement>(&operations[i]);
        if (measurement == nullptr) {
            continue;
        }

        std::size_t target = i;
        for (std::size_t cursor = i; cursor > 0; --cursor) {
            const auto* rotation = std::get_if<PendingPauliRotation>(&operations[cursor - 1]);
            if (rotation == nullptr ||
                !pauli_bodies_commute(measurement->pauli.pauli, rotation->pauli.pauli)) {
                break;
            }
            if (allow_all_commuting) {
                target = cursor - 1;
                continue;
            }
            if (measurement->pauli.pauli.same_body(rotation->pauli.pauli)) {
                target = cursor - 1;
                break;
            }
        }

        std::size_t current = i;
        while (current > target) {
            std::swap(operations[current - 1], operations[current]);
            --current;
            ++stats.measurement_left_swaps;
        }
    }
}

void optimize_segment(
    std::vector<PendingOperation>& operations,
    PendingOptimizationStats& stats) {
    const int fused_before = stats.fused_rotations;
    fuse_commuting_rotations(operations, stats);
    move_measurements_earlier(
        operations,
        stats,
        stats.fused_rotations != fused_before);
}

} // namespace

PendingOptimizationStats optimize_pending_operations(
    PendingFactoredState& state,
    const std::vector<int>& preserved_prefixes) {
    if (!state.instructions.empty() || !state.pending_prefix_instruction_indices.empty()) {
        detail::fail("pending-operation optimization must run before planning");
    }

    PendingOptimizationStats stats;
    const int operation_count = static_cast<int>(state.pending_operations.size());
    stats.input_operations = operation_count;
    stats.prefix_remap.assign(static_cast<std::size_t>(operation_count + 1), -1);
    stats.prefix_remap[0] = 0;

    std::vector<int> segment_ends;
    segment_ends.reserve(preserved_prefixes.size() + 1);
    for (int prefix : preserved_prefixes) {
        if (prefix < 0 || prefix > operation_count) {
            detail::fail("preserved pending-operation prefix is out of range");
        }
        if (prefix > 0) {
            segment_ends.push_back(prefix);
        }
    }
    segment_ends.push_back(operation_count);
    std::sort(segment_ends.begin(), segment_ends.end());
    segment_ends.erase(std::unique(segment_ends.begin(), segment_ends.end()), segment_ends.end());

    std::vector<PendingOperation> input = std::move(state.pending_operations);
    std::vector<PendingOperation> output;
    output.reserve(input.size());
    int segment_start = 0;
    for (int segment_end : segment_ends) {
        std::vector<PendingOperation> segment;
        segment.reserve(static_cast<std::size_t>(segment_end - segment_start));
        std::move(
            input.begin() + segment_start,
            input.begin() + segment_end,
            std::back_inserter(segment));
        optimize_segment(segment, stats);
        std::move(segment.begin(), segment.end(), std::back_inserter(output));
        stats.prefix_remap[static_cast<std::size_t>(segment_end)] = static_cast<int>(output.size());
        segment_start = segment_end;
    }

    state.pending_operations = std::move(output);
    state.pending_operations_optimized = true;
    stats.output_operations = static_cast<int>(state.pending_operations.size());
    return stats;
}

} // namespace symft
