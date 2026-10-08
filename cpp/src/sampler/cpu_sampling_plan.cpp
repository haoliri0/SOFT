#include "sampler/cpu_sampling_plan.hpp"
#include "sampler/contiguous_active.hpp"
#include "sampler/cpu_profile.hpp"
#include "sampler/random.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace symft {
namespace {
struct LinearBool {
    bool constant = false;
    std::set<int> terms;
    void add(const LinearBool& other) {
        constant ^= other.constant;
        for (int term : other.terms) if (!terms.erase(term)) terms.insert(term);
    }
};
CpuExpression lower(const LinearBool& value, int noise_count) {
    CpuExpression out;
    out.constant = value.constant;
    std::map<unsigned, std::uint64_t> masks;
    for (int t : value.terms) {
        if (t < noise_count) out.noise.push_back(t);
        else {
            const auto b = static_cast<unsigned>(t - noise_count);
            masks[b / 64] ^= std::uint64_t{1} << (b % 64);
        }
    }
    for (const auto& mask : masks) out.branch_masks.push_back(mask);
    return out;
}
std::uint64_t noise_word(const CpuExpression& e, const PresampledExpressionBlock& block, std::size_t word) {
    if (e.packed_noise_index >= 0 && (static_cast<std::size_t>(e.packed_noise_index) + 1) * block.shot_words <= block.expression_words.size()) {
        return block.expression_words[static_cast<std::size_t>(e.packed_noise_index) * block.shot_words + word];
    }
    std::uint64_t v = e.constant ? ~std::uint64_t{0} : 0;
    for (int id : e.noise) v ^= block.expression_words[static_cast<std::size_t>(id) * block.shot_words + word];
    return v;
}
void set_branch(std::vector<std::uint64_t>& bits, int branch, bool value) {
    const auto mask = std::uint64_t{1} << (branch % 64);
    auto& word = bits[branch / 64];
    word = (word & ~mask) | (value ? mask : 0);
}
}

CpuSamplingPlan::CpuSamplingPlan(const FactoredInstructionProgram& program,
    const PresampledExpressionPlan& expressions,
    const std::vector<std::vector<int>>& logical_records, bool allow_real, bool hoist_detectors)
    : initial_k(program.initial_k), max_k(program.max_k), hoist(hoist_detectors) {
    if (max_k > 10 || initial_k < 0 || initial_k > max_k) {
        throw std::runtime_error("compiled CPU plan requires 0 <= initial_k <= max_k <= 10");
    }
    if (expressions.instruction_expressions.size() != program.instructions.size()) {
        throw std::runtime_error("CPU plan expression count mismatch");
    }
    const int noise_count = static_cast<int>(expressions.block_expressions.size());
    source_noise_count = noise_count;
    std::vector<std::optional<LinearBool>> conditions(program.nsymbols + 1), record_values(program.nrecords + 1);
    std::vector<int> branch_producer;
    std::vector<LinearBool> detector_values;
    std::vector<int> detector_boundaries;
    int k = initial_k;
    unsigned gauge = 0;
    bool compatible = true;

    for (std::size_t ix = 0; ix < program.instructions.size(); ++ix) {
        auto evaluate = [&](const SymbolicBool& expr) {
            if (expr.conditions.empty()) return LinearBool{expr.constant, {}};
            const auto& split = expressions.instruction_expressions[ix];
            const auto& block = expressions.block_expressions.at(split.block_expression_index);
            LinearBool value{split.residual_plan.constant, {}};
            if (block.exogenous_conditions.empty()) value.constant ^= block.constant;
            else value.terms.insert(split.block_expression_index);
            for (int id : split.residual_plan.conditions) {
                if (id <= 0 || id > program.nsymbols || !conditions[id]) {
                    throw std::runtime_error("unassigned condition in compiled CPU plan");
                }
                value.add(*conditions[id]);
            }
            return value;
        };
        auto save_record = [&](const auto& inst, const LinearBool& value) {
            if (inst.record) record_values.at(*inst.record) = value;
            if (inst.record_condition) conditions.at(*inst.record_condition) = value;
        };
        std::visit([&](const auto& inst) {
            using T = std::decay_t<decltype(inst)>;
            CpuSamplingOp op;
            op.instruction = static_cast<int>(ix);
            op.k = k;
            op.gauge = gauge;
            if constexpr (std::is_same_v<T, ApplyPrecomputedActivePauliRotation>) {
                if (inst.rotation_kernel.action.nqubits != k) throw std::runtime_error("CPU rotation width mismatch");
                op.kind = CpuOpKind::Rotation;
                op.sign = lower(evaluate(inst.sign), noise_count);
                op.rotation = inst.rotation_kernel;
                compatible = compatible && detail::real_rotation_compatible(op.rotation, gauge);
                ops.push_back(std::move(op));
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                op.kind = CpuOpKind::Promotion;
                op.sign = lower(evaluate(inst.sign), noise_count);
                op.c = std::cos(inst.kernel_angle);
                op.s = std::sin(inst.kernel_angle);
                ops.push_back(std::move(op));
                gauge |= 1U << k++;
            } else if constexpr (std::is_same_v<T, MeasurePrecomputedActivePauli> ||
                                 std::is_same_v<T, IntroduceDormantMeasurementBranch>) {
                op.branch = branches++;
                branch_producer.push_back(static_cast<int>(ops.size()));
                conditions.at(inst.branch) = LinearBool{false, {noise_count + op.branch}};
                if constexpr (std::is_same_v<T, MeasurePrecomputedActivePauli>) {
                    if (k <= 0 || inst.kernel.action.nqubits != k) throw std::runtime_error("CPU measurement width mismatch");
                    op.kind = CpuOpKind::Measurement;
                    op.measurement = inst.kernel;
                    compatible = compatible && detail::real_measurement_compatible(op.measurement, gauge);
                    gauge = detail::real_measurement_next_gauge(op.measurement, gauge);
                    --k;
                } else op.kind = CpuOpKind::DormantBranch;
                ops.push_back(std::move(op));
                save_record(inst, evaluate(inst.outcome));
            } else if constexpr (std::is_same_v<T, RecordMeasurement>) {
                save_record(inst, evaluate(inst.outcome));
            } else if constexpr (std::is_same_v<T, RecordDetector>) {
                LinearBool value;
                if (inst.records.empty()) value = evaluate(inst.outcome);
                else for (int record : inst.records) {
                    if (!record_values.at(record)) throw std::runtime_error("unassigned detector record in CPU plan");
                    value.add(*record_values[record]);
                }
                detector_values.push_back(std::move(value));
                detector_boundaries.push_back(static_cast<int>(ops.size()));
            }
        }, program.instructions[ix]);
        if (k < 0 || k > max_k) throw std::runtime_error("CPU plan active width out of range");
    }
    final_gauge = gauge;
    real_gauge = allow_real && compatible;
    if (real_gauge) for (auto& op : ops) {
        if (op.kind == CpuOpKind::Rotation) op.real_rotation = detail::make_real_rotation(op.rotation, op.gauge);
        if (op.kind == CpuOpKind::Measurement) op.real_measurement = detail::make_real_measurement(op.measurement, op.gauge);
    }
    checks.resize(ops.size() + 1);
    std::vector<LinearBool> initial;
    for (std::size_t d = 0; d < detector_values.size(); ++d) {
        const auto& value = detector_values[d];
        int ready = 0;
        for (int t : value.terms) if (t >= noise_count) ready = std::max(ready, branch_producer.at(t - noise_count) + 1);
        if (ready == 0) ++noise_only_detectors;
        const auto e = lower(value, noise_count);
        detectors.push_back(e);
        if (hoist && ready == 0) initial.push_back(value);
        else checks.at(hoist ? ready : detector_boundaries[d]).push_back(e);
    }
    // Exact row reduction of simultaneous zero checks. Preserve contradictory
    // constant rows; do not substitute reduced rows for a full detector vector.
    std::map<int, LinearBool> rows;
    for (auto row : initial) {
        while (!row.terms.empty()) {
            const int pivot = *row.terms.rbegin();
            const auto found = rows.find(pivot);
            if (found == rows.end()) { rows.emplace(pivot, row); break; }
            row.add(found->second);
        }
        if (row.terms.empty() && row.constant) initial_checks.push_back(lower(row, noise_count));
    }
    for (auto it = rows.begin(); it != rows.end(); ++it) {
        for (auto later = std::next(it); later != rows.end(); ++later) {
            if (later->second.terms.count(it->first)) later->second.add(it->second);
        }
    }
    for (const auto& [pivot, row] : rows) initial_checks.push_back(lower(row, noise_count));
    LinearBool logical_value;
    for (const auto& group : logical_records) for (int record : group) {
        if (!record_values.at(record)) throw std::runtime_error("unassigned logical record in CPU plan");
        logical_value.add(*record_values[record]);
    }
    logical = lower(logical_value, noise_count);
    records.resize(record_values.size());
    for (std::size_t i = 1; i < records.size(); ++i) if (record_values[i]) records[i] = lower(*record_values[i], noise_count);
    // Drop only unused dormant random bits, never an active measurement.
    // Preserve the versioned random tape by skipping its counter positions.
    // Full trace mode keeps all original operations/records for verification.
    std::vector<bool> used(branches);
    auto mark = [&](const CpuExpression& e) {
        for (const auto& [word, mask] : e.branch_masks) {
            auto bits = mask;
            while (bits) {
                used.at(word * 64 + std::countr_zero(bits)) = true;
                bits &= bits - 1;
            }
        }
    };
    mark(logical);
    for (const auto& e : detectors) mark(e);
    for (const auto& op : ops) mark(op.sign);
    live_checks.emplace_back();
    unsigned skipped = 0;
    for (unsigned i = 0; i <= ops.size(); ++i) {
        live_checks.back().insert(live_checks.back().end(), checks[i].begin(), checks[i].end());
        if (i == ops.size()) break;
        if (ops[i].kind == CpuOpKind::DormantBranch && !used[ops[i].branch]) { ++skipped; continue; }
        live_ops.push_back(i);
        skipped_rng.push_back(skipped);
        skipped = 0;
        live_checks.emplace_back();
    }
    // Precompute affine noise parts once per 64 shots, not once per op/shot.
    // Keep the original expressions for independent trace/replay on raw blocks.
    std::map<std::pair<bool, std::vector<int>>, int> interned;
    auto intern_noise = [&](CpuExpression& e) {
        if (e.noise.size() <= 1) return;
        const auto key = std::make_pair(e.constant, e.noise);
        auto [it, added] = interned.emplace(key, noise_count + static_cast<int>(noise_outputs.size()));
        if (added) {
            CpuExpression source;
            source.constant = e.constant;
            source.noise = e.noise;
            noise_outputs.push_back(std::move(source));
        }
        e.packed_noise_index = it->second;
    };
    for (auto& e : initial_checks) intern_noise(e);
    for (auto& op : ops) intern_noise(op.sign);
    for (auto& group : checks) for (auto& e : group) intern_noise(e);
    for (auto& group : live_checks) for (auto& e : group) intern_noise(e);
    intern_noise(logical);
}

CpuSamplingWorkspace::CpuSamplingWorkspace(const CpuSamplingPlan& plan)
    : re(std::size_t{1} << plan.max_k), im(plan.real_gauge ? 0 : re.size()),
      scratch_re(re.size()), scratch_im(im.size()), branches((plan.branches + 63) / 64) {}

bool evaluate_cpu_expression(const CpuExpression& expression, const PresampledExpressionBlock& noise,
    int shot, const std::vector<std::uint64_t>& branches) {
    bool v = (noise_word(expression, noise, static_cast<std::size_t>(shot / 64)) >> (shot % 64)) & 1;
    for (const auto& [word, mask] : expression.branch_masks) v ^= (std::popcount(branches[word] & mask) & 1) != 0;
    return v;
}

std::uint64_t cpu_shot_seed(std::uint64_t stream, std::uint64_t shot) {
    // Explicit opt-in RNG contract "cpu-shot-v1": separate seed per global
    // shot, then exactly one SplitMix64 draw per original stochastic site,
    // including deterministic Born probabilities. Early exits cannot perturb
    // other shots; real/complex and hoist on/off consume identical live tapes.
    auto mix = [](std::uint64_t x) { return next_random_u64(x); };
    return mix(stream ^ 0x4350555f76310000ULL) ^ mix(shot ^ 0x9e3779b97f4a7c15ULL);
}

namespace {
template<bool Real, bool Trace>
CpuShotResult execute_shot_impl(const CpuSamplingPlan& plan, CpuSamplingWorkspace& w,
    const PresampledExpressionBlock& noise, int shot, std::uint64_t rng, bool postselect,
    const CpuTrace* trace, bool initial_checked) {
    auto eval = [&](const CpuExpression& e) { return evaluate_cpu_expression(e, noise, shot, w.branches); };
    if (postselect && !initial_checked) for (const auto& check : plan.initial_checks) if (eval(check)) return {true, false};
    {
        ScopedCpuTimer timer(CpuPhase::Reset);
        std::fill_n(w.re.data(), std::size_t{1} << plan.initial_k, 0.0);
        if constexpr (!Real) std::fill_n(w.im.data(), std::size_t{1} << plan.initial_k, 0.0);
        w.re[0] = 1;
        std::fill(w.branches.begin(), w.branches.end(), 0);
    }
    const std::size_t op_count = Trace ? plan.ops.size() : plan.live_ops.size();
    for (std::size_t j = 0; j <= op_count; ++j) {
        const auto& checks = Trace ? plan.checks[j] : plan.live_checks[j];
        if (postselect) for (const auto& check : checks) if (eval(check)) return {true, false};
        if (j == op_count) break;
        if constexpr (!Trace) rng += 0x9e3779b97f4a7c15ULL * plan.skipped_rng[j];
        const auto& op = plan.ops[Trace ? j : plan.live_ops[j]];
        const unsigned dim = 1U << op.k;
        double probability = -1;
        int branch_value = -1;
        switch (op.kind) {
        case CpuOpKind::Rotation: {
            ScopedCpuTimer timer(CpuPhase::Rotation, op.k, 1);
            const bool sign = eval(op.sign);
            if constexpr (Real) detail::rotate_real_active(w.re.data(), op.real_rotation, sign);
            else detail::rotate_contiguous_active(w.re.data(), w.im.data(), dim, op.rotation, sign);
            break;
        }
        case CpuOpKind::Promotion: {
            ScopedCpuTimer timer(CpuPhase::Promotion, op.k, 1);
            const double q = eval(op.sign) ? op.s : -op.s;
            if constexpr (Real) detail::promote_real_active(w.re.data(), dim, op.gauge, op.c, q);
            else detail::promote_contiguous_active(w.re.data(), w.im.data(), dim, op.c, q);
            break;
        }
        case CpuOpKind::Measurement: {
            ScopedCpuTimer timer(CpuPhase::Measurement, op.k, 1);
            const auto& m = op.measurement;
            if constexpr (Real) probability = detail::real_measurement_probability(w.re.data(), op.real_measurement);
            else probability = m.is_diagonal
                ? detail::diagonal_probability_contiguous(w.re.data(), w.im.data(), m, true)
                : detail::nondiagonal_probability_contiguous(w.re.data(), w.im.data(), m, true);
            const bool branch = rand_float(rng) < probability;
            branch_value = branch;
            const double norm = branch ? probability : 1 - probability;
            if (!(norm > 0) || !std::isfinite(norm)) throw std::runtime_error("impossible compiled CPU measurement branch");
            const double invnorm = 1 / std::sqrt(norm);
            if constexpr (Real) {
                detail::project_real_active(w.re.data(), w.scratch_re.data(), op.real_measurement, branch, invnorm);
                w.re.swap(w.scratch_re);
            } else if (m.is_diagonal) detail::project_diagonal_contiguous(w.re.data(), w.im.data(), m, branch, invnorm);
            else detail::project_nondiagonal_contiguous(w.re.data(), w.im.data(), w.scratch_re.data(), w.scratch_im.data(), m, branch, invnorm);
            set_branch(w.branches, op.branch, branch);
            break;
        }
        case CpuOpKind::DormantBranch: {
            ScopedCpuTimer timer(CpuPhase::Classical, op.k, 1);
            const bool branch = (next_random_u64(rng) & 1) != 0;
            branch_value = branch;
            set_branch(w.branches, op.branch, branch);
            break;
        }
        }
        if constexpr (Trace) (*trace)(op, w, probability, branch_value);
    }
    if (!postselect) for (const auto& detector : plan.detectors) if (eval(detector)) return {true, false};
    return {false, eval(plan.logical)};
}
}

CpuShotResult execute_cpu_shot(const CpuSamplingPlan& plan, CpuSamplingWorkspace& w,
    const PresampledExpressionBlock& noise, int shot, std::uint64_t seed,
    bool postselect, const CpuTrace* trace, bool initial_checked) {
    if (shot < 0 || shot >= noise.nshots) throw std::runtime_error("compiled CPU shot index out of range");
    if (trace) {
        if (plan.real_gauge) return execute_shot_impl<true, true>(plan, w, noise, shot, seed, postselect, trace, initial_checked);
        return execute_shot_impl<false, true>(plan, w, noise, shot, seed, postselect, trace, initial_checked);
    }
    if (plan.real_gauge) return execute_shot_impl<true, false>(plan, w, noise, shot, seed, postselect, nullptr, initial_checked);
    return execute_shot_impl<false, false>(plan, w, noise, shot, seed, postselect, nullptr, initial_checked);
}

CpuChunkResult execute_cpu_chunk(const CpuSamplingPlan& plan, CpuSamplingWorkspace& w,
    const PresampledExpressionBlock& source, std::uint64_t stream, std::uint64_t first_shot) {
    {
        ScopedCpuTimer timer(CpuPhase::Expression);
        auto& packed = w.packed_noise;
        packed.nshots = source.nshots;
        packed.shot_words = source.shot_words;
        packed.expression_words.resize((plan.source_noise_count + plan.noise_outputs.size()) * source.shot_words);
        std::copy_n(source.expression_words.data(), plan.source_noise_count * source.shot_words, packed.expression_words.data());
        for (std::size_t e = 0; e < plan.noise_outputs.size(); ++e) {
            const auto& form = plan.noise_outputs[e];
            auto* dst = packed.expression_words.data() + (plan.source_noise_count + e) * source.shot_words;
            std::fill_n(dst, source.shot_words, form.constant ? ~std::uint64_t{0} : 0);
            for (int id : form.noise) {
                const auto* src = source.expression_words.data() + static_cast<std::size_t>(id) * source.shot_words;
                for (std::size_t word = 0; word < source.shot_words; ++word) dst[word] ^= src[word];
            }
        }
    }
    const auto& noise = w.packed_noise;
    CpuChunkResult counts;
    counts.shots = noise.nshots;
    {
        ScopedCpuTimer timer(CpuPhase::Entrance);
        w.rejected.assign(noise.shot_words, 0);
        for (const auto& check : plan.initial_checks) {
            for (std::size_t word = 0; word < noise.shot_words; ++word) w.rejected[word] |= noise_word(check, noise, word);
        }
    }
    for (std::size_t word = 0; word < noise.shot_words; ++word) {
        const int n = std::min(64, noise.nshots - static_cast<int>(word * 64));
        const std::uint64_t mask = n == 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << n) - 1;
        counts.discarded += std::popcount(w.rejected[word] & mask);
        auto live = ~w.rejected[word] & mask;
        while (live) {
            const int shot = static_cast<int>(word * 64) + std::countr_zero(live);
            live &= live - 1;
            const auto result = execute_cpu_shot(plan, w, noise, shot, cpu_shot_seed(stream, first_shot + shot), true, nullptr, true);
            if (result.discarded) ++counts.discarded;
            else { ++counts.accepted; counts.logical_errors += result.logical_error; }
        }
    }
    return counts;
}
} // namespace symft
