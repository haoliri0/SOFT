#include "frontend/stim_prepared_sampler.hpp"
#include "sampler/cpu_sampling_plan.hpp"
#include "sampler/exogenous.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <iostream>
#include <random>
#include <stdexcept>
#include <type_traits>

using namespace symft;
using U = std::uint64_t;
using Z = std::complex<long double>;
namespace {
double max_state_error = 0, max_probability_error = 0;
std::uint64_t compared_states = 0, compared_shots = 0;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
U draw(U& s) {
    U z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
unsigned insert(unsigned x, unsigned bit) {
    const unsigned low = (1U << bit) - 1;
    return (x & low) | ((x & ~low) << 1);
}
bool parity(U x) { return std::popcount(x) & 1; }
Z to_z(Complex z) { return {z.real(), z.imag()}; }

struct Snapshot {
    std::vector<Z> state;
    long double probability = -1;
    int branch = -1;
};
struct Reference {
    std::vector<Snapshot> snapshots;
    std::vector<bool> records, detectors;
    bool discarded = false, logical_error = false;
};

// Independent long-double full-vector reference: original GF(2) expressions,
// original records, full Pauli action and explicit measurement projections.
// Neither compiled expressions, real kernels nor SIMD probability helpers
// participate. The same frozen exogenous samples and stochastic tape are used.
Reference reference(const CircuitSamplingInput& input, const PackedPresampledExogenous& noise,
                    int shot, U seed) {
    const auto& p = input.program;
    Reference result;
    result.snapshots.resize(p.instructions.size());
    result.records.resize(p.nrecords + 1);
    std::vector<bool> conditions(p.nsymbols + 1);
    for (int c = 1; c <= p.nsymbols; ++c) {
        conditions[c] = (noise.value_words[(c - 1) * noise.shot_words + shot / 64] >> (shot % 64)) & 1;
    }
    auto eval = [&](const SymbolicBool& value) {
        bool b = value.constant;
        for (int c : value.conditions) b ^= conditions.at(c);
        return b;
    };
    auto record = [&](const auto& inst) {
        bool b = eval(inst.outcome);
        if (inst.record) result.records.at(*inst.record) = b;
        if (inst.record_condition) conditions.at(*inst.record_condition) = b;
    };
    std::vector<Z> state(std::size_t{1} << p.initial_k);
    state[0] = 1;
    // Frozen one-word-per-site tape, including probability 0 and 1.
    std::vector<U> tape;
    for (const auto& inst : p.instructions) {
        if (std::holds_alternative<MeasurePrecomputedActivePauli>(inst) ||
            std::holds_alternative<IntroduceDormantMeasurementBranch>(inst)) tape.push_back(draw(seed));
    }
    std::size_t site = 0;
    for (std::size_t ix = 0; ix < p.instructions.size(); ++ix) {
        auto& snapshot = result.snapshots[ix];
        std::visit([&](const auto& inst) {
            using T = std::decay_t<decltype(inst)>;
            if constexpr (std::is_same_v<T, ApplyPrecomputedActivePauliRotation>) {
                const auto& a = inst.rotation_kernel.action;
                const long double c = std::cos(static_cast<long double>(inst.kernel_angle));
                const Z q = Z(0, -std::sin(static_cast<long double>(inst.kernel_angle))) * to_z(a.even_phase);
                const bool sign = eval(inst.sign);
                std::vector<Z> out(state.size());
                for (std::size_t dst = 0; dst < state.size(); ++dst) {
                    const auto src = dst ^ a.xmask;
                    out[dst] = c * state[dst] + (sign ^ parity(src & a.zmask) ? -q : q) * state[src];
                }
                state = std::move(out);
                snapshot.state = state;
            } else if constexpr (std::is_same_v<T, PromoteDormantRotation>) {
                const long double c = std::cos(static_cast<long double>(inst.kernel_angle));
                long double q = std::sin(static_cast<long double>(inst.kernel_angle));
                if (!eval(inst.sign)) q = -q;
                std::vector<Z> out(2 * state.size());
                for (std::size_t j = 0; j < state.size(); ++j) {
                    out[j] = c * state[j]; out[j + state.size()] = Z(0, q) * state[j];
                }
                state = std::move(out);
                snapshot.state = state;
            } else if constexpr (std::is_same_v<T, MeasurePrecomputedActivePauli>) {
                const auto& m = inst.kernel;
                auto project = [&](bool branch) {
                    std::vector<Z> out(m.out_dim);
                    for (unsigned j = 0; j < m.out_dim; ++j) {
                        const unsigned base = insert(j, m.pivot);
                        if (m.is_diagonal) {
                            const bool pv = branch ^ m.diagonal_phase_bit ^ parity(base & m.action.zmask);
                            out[j] = state[base | (static_cast<unsigned>(pv) << m.pivot)];
                        } else {
                            Z coefficient = std::conj(to_z(m.action.even_phase));
                            if (branch ^ parity(base & m.action.zmask)) coefficient = -coefficient;
                            out[j] = (state[base] + coefficient * state[base ^ m.action.xmask]) / std::sqrt(2.L);
                        }
                    }
                    return out;
                };
                auto out = project(true);
                long double probability = 0;
                for (auto z : out) probability += std::norm(z);
                probability = std::clamp(probability, 0.L, 1.L);
                const long double u = static_cast<long double>(tape.at(site++) >> 11) * 0x1.0p-53L;
                const bool branch = u < probability;
                if (!branch) out = project(false);
                const long double norm = branch ? probability : 1 - probability;
                require(norm > 0, "reference selected impossible branch");
                for (auto& z : out) z /= std::sqrt(norm);
                state = std::move(out);
                conditions.at(inst.branch) = branch;
                record(inst);
                snapshot.state = state;
                snapshot.probability = probability;
                snapshot.branch = branch;
            } else if constexpr (std::is_same_v<T, IntroduceDormantMeasurementBranch>) {
                const bool branch = tape.at(site++) & 1;
                conditions.at(inst.branch) = branch;
                record(inst);
                snapshot.branch = branch;
            } else if constexpr (std::is_same_v<T, RecordMeasurement>) record(inst);
            else if constexpr (std::is_same_v<T, RecordDetector>) {
                bool b = inst.records.empty() ? eval(inst.outcome) : false;
                for (int r : inst.records) b ^= result.records.at(r);
                if (!input.expected_detector_words.empty()) {
                    b ^= packed_bit(input.expected_detector_words, static_cast<int>(result.detectors.size()));
                }
                result.detectors.push_back(b);
                result.discarded |= b;
            }
        }, p.instructions[ix]);
    }
    bool logical = false;
    for (const auto& group : input.logical_records) for (int r : group) logical ^= result.records.at(r);
    if (!input.expected_observable_words.empty()) logical ^= packed_bit(input.expected_observable_words, input.observable);
    result.logical_error = !result.discarded && logical;
    return result;
}

void compare_input(CircuitSamplingInput input, const std::string& name, int shots) {
    auto noise = presample_exogenous_packed(input.program, shots, 832718);
    PresampledExpressionPlan expressions;
    prepare_presampled_expression_plan(expressions, input.program, noise);
    PresampledExpressionBlock block;
    evaluate_presampled_expression_block(block, expressions, noise);
    const bool expected_observable = !input.expected_observable_words.empty() &&
        packed_bit(input.expected_observable_words, input.observable);
    CpuSamplingPlan real(input.program, expressions, input.logical_records, true, true,
        input.expected_detector_words, expected_observable);
    CpuSamplingPlan complex(input.program, expressions, input.logical_records, false, true,
        input.expected_detector_words, expected_observable);
    CpuSamplingPlan late(input.program, expressions, input.logical_records, true, false,
        input.expected_detector_words, expected_observable);
    CpuSamplingWorkspace wr(real), wc(complex), wl(late);
    for (U stream : {0ULL, 932ULL, 0xffff000000000001ULL}) {
        for (int shot = 0; shot < shots; ++shot) {
            const auto seed = cpu_shot_seed(stream, shot);
            const auto expected = reference(input, noise, shot, seed);
            for (const auto& pair : {std::make_pair(&real, &wr), std::make_pair(&complex, &wc)}) {
                const auto& plan = *pair.first;
                auto& w = *pair.second;
                CpuTrace trace = [&](const CpuSamplingOp& op, const CpuSamplingWorkspace& state, double probability, int branch) {
                    const auto& ref = expected.snapshots.at(op.instruction);
                    require(branch == ref.branch, name + " branch mismatch at " + std::to_string(op.instruction));
                    if (probability >= 0) {
                        const double error = std::abs(static_cast<double>(ref.probability - probability));
                        max_probability_error = std::max(max_probability_error, error);
                        require(error < 2e-11, name + " probability mismatch");
                    }
                    if (ref.state.empty()) return;
                    unsigned gauge = op.gauge;
                    if (op.kind == CpuOpKind::Promotion) gauge |= 1U << op.k;
                    if (op.kind == CpuOpKind::Measurement) gauge = detail::real_measurement_next_gauge(op.measurement, gauge);
                    std::vector<Z> actual(ref.state.size());
                    std::size_t anchor = 0;
                    for (std::size_t b = 0; b < actual.size(); ++b) {
                        actual[b] = plan.real_gauge ? (parity(b & gauge) ? Z(0, state.re[b]) : Z(state.re[b], 0)) : Z(state.re[b], state.im[b]);
                        if (std::norm(ref.state[b]) > std::norm(ref.state[anchor])) anchor = b;
                    }
                    Z phase = actual[anchor] / ref.state[anchor];
                    phase /= std::abs(phase);
                    for (std::size_t b = 0; b < actual.size(); ++b) {
                        const double error = static_cast<double>(std::abs(actual[b] - phase * ref.state[b]));
                        max_state_error = std::max(max_state_error, error);
                        require(error < 2e-10, name + " state mismatch at " + std::to_string(op.instruction) + " basis=" + std::to_string(b));
                    }
                    ++compared_states;
                };
                const auto full = execute_cpu_shot(plan, w, block, shot, seed, false, &trace);
                require(full.discarded == expected.discarded && full.logical_error == expected.logical_error, name + " full counts mismatch");
                for (std::size_t d = 0; d < expected.detectors.size(); ++d) {
                    require(evaluate_cpu_expression(plan.detectors[d], block, shot, w.branches) == expected.detectors[d], name + " detector mismatch");
                }
                for (std::size_t r = 1; r < expected.records.size(); ++r) {
                    require(evaluate_cpu_expression(plan.records[r], block, shot, w.branches) == expected.records[r], name + " record mismatch");
                }
                const auto early = execute_cpu_shot(plan, w, block, shot, seed, true);
                require(early.discarded == full.discarded && early.logical_error == full.logical_error, name + " hoisted result mismatch");
            }
            const auto original_checks = execute_cpu_shot(late, wl, block, shot, seed, true);
            require(original_checks.discarded == expected.discarded && original_checks.logical_error == expected.logical_error, name + " late checks mismatch");
            ++compared_shots;
        }
        auto count = execute_cpu_chunk(real, wr, block, stream, 0);
        CpuChunkResult expected_count;
        for (int shot = 0; shot < shots; ++shot) {
            const auto r = reference(input, noise, shot, cpu_shot_seed(stream, shot));
            ++expected_count.shots;
            expected_count.discarded += r.discarded;
            expected_count.accepted += !r.discarded;
            expected_count.logical_errors += r.logical_error;
        }
        require(count.shots == expected_count.shots && count.discarded == expected_count.discarded &&
                count.accepted == expected_count.accepted && count.logical_errors == expected_count.logical_errors, name + " chunk tail/mask mismatch");
    }
    std::cout << "PASS " << name << " shots=" << shots * 3 << " real=" << real.real_gauge
              << " noise_only=" << real.noise_only_detectors << "\n";
}

void api_tests() {
    {
        auto normalized = with_reference_sample(make_stim_circuit_sampling_input(parse_stim_circuit_text(
            "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n")));
        CircuitSamplingOptions settings;
        settings.cpu_compiled = true;
        settings.postselect_detectors = true;
        PreparedCircuitBatchSampler selected(normalized, settings);
        require(selected.info().cpu_compiled, "normalized compiled path not active");
        const auto normalized_counts = selected.sample(65, 77).counts;
        require(normalized_counts.accepted == 65 && normalized_counts.discarded == 0 &&
                normalized_counts.logical_errors == 0, "one-based detector reference normalization");
    }
    auto input = make_stim_circuit_sampling_input(parse_stim_circuit_text(
        "H 0\nT 0\nH 0\nX_ERROR(0.13) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"));
    input.program.use_active_components = false;
    CircuitSamplingOptions options;
    options.cpu_compiled = true;
    options.postselect_detectors = true;
    options.sample_chunk_shots = 65;
    PreparedCircuitBatchSampler sampler(input, options);
    require(sampler.info().cpu_compiled, "CPU opt-in not active");
    const auto count = sampler.sample(257, 982).counts;
    auto moved = std::move(sampler);
    auto moved_again = PreparedCircuitBatchSampler(input, options);
    moved_again = std::move(moved);
    const auto repeat = moved_again.sample(257, 982).counts;
    require(count.discarded == repeat.discarded && count.logical_errors == repeat.logical_errors, "move/repeat reproducibility");
    require(moved_again.sample(0, 55).counts.shots == 0, "zero shots");
    for (int batch : {1, 7, 32, 257}) {
        options.batch_size = batch;
        auto alternate = PreparedCircuitBatchSampler(input, options).sample(257, 982).counts;
        require(alternate.discarded == count.discarded && alternate.logical_errors == count.logical_errors, "compiled batch independence");
    }
    options.postselect_detectors = false;
    PreparedCircuitBatchSampler fallback(input, options);
    require(!fallback.info().cpu_compiled && fallback.info().cpu_fallback_reason == "postselection_disabled", "no-postselection fallback");
    options.postselect_detectors = true;
    options.threads = 2;
    PreparedCircuitBatchSampler threaded(input, options);
    require(!threaded.info().cpu_compiled, "multiworker fallback");
    options.threads = 1;
    auto wider = input;
    wider.program.max_k = 11;
    PreparedCircuitBatchSampler wide(wider, options);
    require(!wide.info().cpu_compiled && wide.info().cpu_fallback_reason == "max_k_exceeds_10", "width fallback");
    options.cpu_compiled = false;
    require(!PreparedCircuitBatchSampler(input, options).info().cpu_compiled, "default backend changed");
}

void promotion_kernel_tests() {
    std::mt19937_64 random(261008);
    std::uniform_real_distribution<double> uniform(-1, 1);
    unsigned checked = 0;
    for (unsigned k = 0; k <= 10; ++k) {
        const unsigned dim = 1U << k;
        for (unsigned test = 0; test < std::min(dim, 128U); ++test) {
            const unsigned gauge = dim <= 128 ? test : random() % dim;
            for (double q : {0.0, -0.0, 0.37, -0.37}) {
                const double c = std::sqrt(1 - q * q);
                std::vector<double> state(2 * dim + 8, 7.0), original(dim);
                for (unsigned j = 0; j < dim; ++j) state[j] = original[j] = uniform(random);
                detail::promote_real_active(state.data(), dim, gauge, c, q);
                for (unsigned j = 0; j < dim; ++j) {
                    const double upper = (parity(j & gauge) ? -q : q) * original[j];
                    require(std::bit_cast<U>(state[j]) == std::bit_cast<U>(c * original[j]),
                            "promotion lower amplitude or signed zero");
                    require(std::bit_cast<U>(state[dim + j]) == std::bit_cast<U>(upper),
                            "promotion upper amplitude or signed zero");
                }
                for (unsigned j = 2 * dim; j < state.size(); ++j)
                    require(state[j] == 7.0, "promotion wrote past active vector");
                ++checked;
            }
        }
    }
    std::cout << "PASS promotion_gauges_and_tails " << checked << "\n";
}

void kernel_tests() {
    std::mt19937_64 random(726335);
    std::uniform_real_distribution<double> uniform(-1, 1);
    unsigned checked = 0;
    for (int k = 1; k <= 10; ++k) {
        const unsigned dim = 1U << k;
        const unsigned cases = k <= 4 ? dim * dim : 256;
        for (unsigned sample = 0; sample < cases; ++sample) {
            const unsigned x = k <= 4 ? sample % dim : random() % dim;
            const unsigned z = k <= 4 ? sample / dim : random() % dim;
            if (x == 0 && z == 0) continue;
            const unsigned gauge = random() % dim;
            PauliString pauli(k);
            pauli.x[0] = x; pauli.z[0] = z;
            pauli.set_phase(std::popcount(x & z) + 2 * (sample & 1));
            const ActivePauliAction action(pauli);
            std::vector<double> r(dim), out(dim);
            long double norm = 0;
            for (auto& value : r) { value = uniform(random); norm += static_cast<long double>(value) * value; }
            for (auto& value : r) value /= std::sqrt(static_cast<double>(norm));
            std::vector<Z> a(dim);
            for (unsigned j = 0; j < dim; ++j) a[j] = parity(j & gauge) ? Z(0, r[j]) : Z(r[j], 0);
            const double angle = sample % 3 == 0 ? 0.0 : (sample % 3 == 1 ? 0.237 : -0.912);
            const PrecomputedActivePauliRotationKernel rotation(action, angle);
            if (detail::real_rotation_compatible(rotation, gauge)) {
                const auto kernel = detail::make_real_rotation(rotation, gauge);
                for (bool sign : {false, true}) {
                    auto actual = r;
                    detail::rotate_real_active(actual.data(), kernel, sign);
                    const Z q = Z(0, -std::sin(static_cast<long double>(angle))) * to_z(action.even_phase);
                    for (unsigned dst = 0; dst < dim; ++dst) {
                        const auto src = dst ^ x;
                        const auto ref = std::cos(static_cast<long double>(angle)) * a[dst] +
                            (sign ^ parity(src & z) ? -q : q) * a[src];
                        const Z value = parity(dst & gauge) ? Z(0, actual[dst]) : Z(actual[dst], 0);
                        require(std::abs(value - ref) < 1e-12L, "random real rotation kernel");
                    }
                    ++checked;
                }
            }
            // Every supported pivot, not just the planner's highest one.
            const unsigned pivot_mask = x ? x : z;
            for (int pivot = 0; pivot < k; ++pivot) {
                if (!(pivot_mask & (1U << pivot))) continue;
                const PrecomputedActivePauliMeasurementKernel measurement(action, pivot);
                if (!detail::real_measurement_compatible(measurement, gauge)) continue;
                const auto kernel = detail::make_real_measurement(measurement, gauge);
                for (bool branch : {false, true}) {
                    std::vector<Z> ref(dim / 2), actual(dim / 2);
                    long double probability = 0;
                    for (unsigned j = 0; j < dim / 2; ++j) {
                        const unsigned base = insert(j, pivot);
                        if (x == 0) {
                            const bool bit = branch ^ measurement.diagonal_phase_bit ^ parity(base & z);
                            ref[j] = a[base | (static_cast<unsigned>(bit) << pivot)];
                        } else {
                            Z q = std::conj(to_z(action.even_phase));
                            if (branch ^ parity(base & z)) q = -q;
                            ref[j] = (a[base] + q * a[base ^ x]) / std::sqrt(2.L);
                        }
                        probability += std::norm(ref[j]);
                    }
                    if (branch) require(std::abs(detail::real_measurement_probability(r.data(), kernel) - probability) < 1e-12L,
                                        "random real probability kernel");
                    detail::project_real_active(r.data(), out.data(), kernel, branch, 1);
                    unsigned anchor = 0;
                    for (unsigned j = 0; j < dim / 2; ++j) {
                        actual[j] = parity(j & kernel.next_gauge) ? Z(0, out[j]) : Z(out[j], 0);
                        if (std::norm(ref[j]) > std::norm(ref[anchor])) anchor = j;
                    }
                    Z phase = actual[anchor] / ref[anchor];
                    phase /= std::abs(phase);
                    for (unsigned j = 0; j < dim / 2; ++j) require(std::abs(actual[j] - phase * ref[j]) < 1e-12L,
                        "random real projection kernel");
                    ++checked;
                }
            }
        }
    }
    std::cout << "PASS randomized_gauge_kernels " << checked << "\n";
}
}

int main() {
    try {
        promotion_kernel_tests();
        kernel_tests();
        for (const auto& [name, text] : std::vector<std::pair<std::string, std::string>>{
            {"real_T", "H 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"complex_TS", "H 0\nT 0\nS 0\nH 0\nT 0\nH 0\nM 0\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"feedback_Y", "H 0 1\nT 0\nT_DAG 1\nCX 0 1\nMY 0\nCY rec[-1] 1\nH 1\nT 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"all_live", "X_ERROR(0) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"all_dead", "X_ERROR(1) 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"high_noise", "H 0 1\nT 0 1\nDEPOLARIZE2(0.2) 0 1\nCX 0 1\nMX 0\nDETECTOR rec[-1]\nMY 1\nOBSERVABLE_INCLUDE(0) rec[-1]\n"},
            {"contradiction", "X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\nX 0\nM 0\nDETECTOR rec[-1]\n"}}) {
            compare_input(make_stim_circuit_sampling_input(parse_stim_circuit_text(text)), name, 65);
        }
        std::string algebra;
        for (int q = 0; q < 70; ++q) algebra += "H " + std::to_string(q) + "\nM " + std::to_string(q) + "\n";
        algebra += "X_ERROR(0.1) 70\nCX rec[-1] 70 rec[-2] 70 rec[-3] 70\nM 70\n"
                   "DETECTOR rec[-1] rec[-2] rec[-3] rec[-4]\nDETECTOR rec[-1] rec[-2] rec[-3] rec[-4]\n"
                   "H 71\nT 71\nCX rec[-1] 71\nH 71\nM 71\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2] rec[-35] rec[-71]\n";
        compare_input(make_stim_circuit_sampling_input(parse_stim_circuit_text(algebra)), "70_branches", 257);
        for (int shots : {1, 2, 7, 31, 32, 63, 64}) compare_input(
            make_stim_circuit_sampling_input(parse_stim_circuit_text("X_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n")),
            "tail_" + std::to_string(shots), shots);
        for (int distance : {3, 5}) {
            compare_input(make_stim_circuit_sampling_input_from_file(
                "benchmark/circuit/msc_d" + std::to_string(distance) + "_inject_cultivate_p1e-3.stim"),
                "msc_d" + std::to_string(distance), distance == 3 ? 257 : 65);
        }
        auto normalized = make_stim_circuit_sampling_input(parse_stim_circuit_text(
            "X 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"
            "H 1\nT 1\nH 1\nM 1\nDETECTOR rec[-1]\n"
            "OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n"));
        normalized.expected_detector_words = {3};
        normalized.expected_observable_words = {1};
        normalized.reference_normalized = true;
        compare_input(normalized, "normalized_quantum_and_noise_checks", 65);
        auto reference_input = with_reference_sample(make_stim_circuit_sampling_input(parse_stim_circuit_text(
            "X 0\nX_ERROR(0.1) 0\nM 0\nDETECTOR rec[-1]\n"
            "X 0\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1] rec[-2]\n")));
        compare_input(reference_input, "normalized_affine_rows", 65);
        api_tests();
        std::cout << "PASS total_shots=" << compared_shots << " compared_states=" << compared_states
                  << " max_state_error=" << max_state_error << " max_probability_error=" << max_probability_error << "\n";
    } catch (const std::exception& e) {
        std::cerr << "cpu_sampling_plan_tests: " << e.what() << "\n";
        return 1;
    }
}
