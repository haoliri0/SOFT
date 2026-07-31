#include "frontend/stim.hpp"

#include "core/internal.hpp"
#include "frontend/stim_prepared_sampler.hpp"
#include "sampler/single_shot.hpp"

#include <chrono>
#include <limits>
#include <utility>
#include <vector>

namespace symft {
namespace {

using detail::fail;

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

int instruction_checkpoint_for_pending_prefix(
    const FactoredInstructionProgram& program,
    int pending_prefix) {
    if (pending_prefix < 0) {
        fail("detector pending-operation position is negative");
    }
    if (pending_prefix == 0 && program.pending_prefix_instruction_indices.empty()) {
        return 0;
    }
    if (pending_prefix >= static_cast<int>(program.pending_prefix_instruction_indices.size())) {
        fail("detector pending-operation position is out of range");
    }
    const int checkpoint = program.pending_prefix_instruction_indices[static_cast<std::size_t>(pending_prefix)];
    if (checkpoint < 0 || checkpoint > static_cast<int>(program.instructions.size())) {
        fail("detector instruction checkpoint is out of range");
    }
    return checkpoint;
}

SymbolicBool detector_expression(
    const CircuitDetector& detector,
    const std::vector<SymbolicBool>& measurement_records) {
    SymbolicBool out(false);
    for (int record : detector.records) {
        if (record <= 0 || record > static_cast<int>(measurement_records.size())) {
            fail("detector references an out-of-range measurement record");
        }
        out = xor_bool(out, measurement_records[static_cast<std::size_t>(record - 1)]);
    }
    return out;
}

FactoredInstructionProgram insert_stim_detector_events(
    FactoredInstructionProgram program,
    const std::vector<CircuitDetector>& detectors,
    const std::vector<SymbolicBool>& measurement_records) {
    if (detectors.empty()) {
        return program;
    }
    std::vector<std::vector<RecordDetector>> events(program.instructions.size() + 1);
    for (std::size_t idx = 0; idx < detectors.size(); ++idx) {
        const auto& detector = detectors[idx];
        RecordDetector instruction;
        instruction.detector = static_cast<int>(idx + 1);
        instruction.records = detector.records;
        instruction.outcome = detector_expression(detector, measurement_records);
        const int checkpoint = instruction_checkpoint_for_pending_prefix(program, detector.after_pending_operation);
        events[static_cast<std::size_t>(checkpoint)].push_back(std::move(instruction));
    }

    std::vector<FactoredInstruction> instructions;
    instructions.reserve(program.instructions.size() + detectors.size());
    for (auto& detector_instruction : events[0]) {
        instructions.push_back(std::move(detector_instruction));
    }
    for (std::size_t idx = 0; idx < program.instructions.size(); ++idx) {
        instructions.push_back(std::move(program.instructions[idx]));
        for (auto& detector_instruction : events[idx + 1]) {
            instructions.push_back(std::move(detector_instruction));
        }
    }

    return FactoredInstructionProgram(
        program.n,
        program.initial_k,
        std::move(instructions),
        program.max_k,
        std::move(program.context));
}

std::vector<CircuitDetector> detectors_with_lowered_positions(
    const QuantumCircuit& circuit,
    const CircuitLoweringResult& lowered) {
    std::vector<CircuitDetector> detectors = circuit.detectors;
    for (auto& detector : detectors) {
        if (detector.after_instruction < 0 ||
            detector.after_instruction >= static_cast<int>(lowered.instruction_pending_operation_counts.size())) {
            fail("detector source position is out of range");
        }
        detector.after_pending_operation =
            lowered.instruction_pending_operation_counts[static_cast<std::size_t>(detector.after_instruction)];
    }
    return detectors;
}

} // namespace

FactoredInstructionProgram plan_stim_factored_program(const StimParseResult& parsed) {
    PendingFactoredState pending(parsed.state);
    std::vector<int> detector_prefixes;
    detector_prefixes.reserve(parsed.detectors.size());
    for (const auto& detector : parsed.detectors) {
        detector_prefixes.push_back(detector.after_pending_operation);
    }
    const PendingOptimizationStats optimization =
        optimize_pending_operations(pending, detector_prefixes);
    std::vector<CircuitDetector> detectors = parsed.detectors;
    for (auto& detector : detectors) {
        const int original_prefix = detector.after_pending_operation;
        if (original_prefix < 0 ||
            original_prefix >= static_cast<int>(optimization.prefix_remap.size()) ||
            optimization.prefix_remap[static_cast<std::size_t>(original_prefix)] < 0) {
            fail("detector pending-operation prefix was not preserved by optimization");
        }
        detector.after_pending_operation =
            optimization.prefix_remap[static_cast<std::size_t>(original_prefix)];
    }
    auto program = plan_factored_updates(std::move(pending));
    return insert_stim_detector_events(
        std::move(program),
        detectors,
        parsed.measurement_records);
}

CircuitSamplingInput make_stim_circuit_sampling_input(
    const QuantumCircuit& circuit,
    const CircuitSamplingOptions& options) {
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    auto detectors = detectors_with_lowered_positions(circuit, lowered);
    StimParseResult parsed{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        std::move(detectors),
        circuit.observables};
    auto program = plan_stim_factored_program(parsed);
    return make_circuit_sampling_input(
        std::move(program),
        logical_records_for_observable(parsed.observables, options.observable),
        options.observable,
        static_cast<int>(parsed.observables.size()));
}

CircuitSamplingInput make_stim_circuit_sampling_input_from_file(
    const std::string& path,
    const CircuitSamplingOptions& options) {
    CircuitSamplingInput input;
    const auto parse_start = Clock::now();
    const QuantumCircuit circuit = parse_stim_circuit_file(path);
    const auto parse_stop = Clock::now();

    const auto plan_start = Clock::now();
    input = make_stim_circuit_sampling_input(circuit, options);
    const auto plan_stop = Clock::now();
    input.preprocessing_timing.parse_s = seconds_between(parse_start, parse_stop);
    input.preprocessing_timing.plan_s = seconds_between(plan_start, plan_stop);
    return input;
}

PreparedCircuitSingleShotSampler prepare_single_shot_sampler_from_stim_file(
    const std::string& path,
    CircuitSamplingOptions options) {
    return PreparedCircuitSingleShotSampler(
        make_stim_circuit_sampling_input_from_file(path, options),
        options);
}

PreparedCircuitBatchSampler prepare_batch_sampler_from_stim_file(
    const std::string& path,
    CircuitSamplingOptions options) {
    return PreparedCircuitBatchSampler(
        make_stim_circuit_sampling_input_from_file(path, options),
        options);
}

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed) {
    const auto program = plan_stim_factored_program(parsed);
    FactoredExecutorState runtime(program, seed);
    StimSampleSummary summary;
    summary.shots = shots;
    for (int shot = 0; shot < shots; ++shot) {
        reset_executor(runtime, program);
        execute_in_place(runtime, program);
        bool discarded = false;
        for (int detector = 0; detector < program.ndetectors; ++detector) {
            discarded = discarded || packed_bit(runtime.detector_words, detector);
        }
        if (discarded) {
            ++summary.discarded;
            continue;
        }
        ++summary.accepted;
        bool logical = false;
        for (const auto& observable : parsed.observables) {
            bool parity = false;
            for (int record : observable.records) {
                parity ^= packed_bit(runtime.measurement_words, record - 1);
            }
            logical ^= parity;
        }
        if (logical) {
            ++summary.logical_errors;
        }
    }
    return summary;
}

double discard_rate(const StimSampleSummary& summary) {
    return summary.shots == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : static_cast<double>(summary.discarded) / static_cast<double>(summary.shots);
}

double logical_error_rate(const StimSampleSummary& summary) {
    return summary.accepted == 0 ? std::numeric_limits<double>::quiet_NaN()
                                 : static_cast<double>(summary.logical_errors) / static_cast<double>(summary.accepted);
}

} // namespace symft
