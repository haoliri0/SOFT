#pragma once

#include "circuit/circuit.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace symft {

using StimDetector = CircuitDetector;
using StimObservableInclude = CircuitObservableInclude;

struct StimParseResult {
    FrameFactoredState state;
    std::vector<SymbolicBool> measurement_records;
    std::vector<StimDetector> detectors;
    std::vector<StimObservableInclude> observables;
};

QuantumCircuit parse_stim_circuit_lines(const std::vector<std::string>& lines);
QuantumCircuit parse_stim_circuit_text(const std::string& text);
QuantumCircuit parse_stim_circuit_file(const std::string& path);
StimParseResult parse_stim_lines(const std::vector<std::string>& lines);
StimParseResult parse_stim_text(const std::string& text);
StimParseResult parse_stim_file(const std::string& path);
FactoredInstructionProgram plan_stim_factored_program(const StimParseResult& parsed);

struct StimSampleSummary {
    int shots = 0;
    int discarded = 0;
    int accepted = 0;
    int logical_errors = 0;
};

StimSampleSummary estimate_stim_logical_error_rate(const StimParseResult& parsed, int shots, std::uint64_t seed = 1);
double discard_rate(const StimSampleSummary& summary);
double logical_error_rate(const StimSampleSummary& summary);

} // namespace symft
