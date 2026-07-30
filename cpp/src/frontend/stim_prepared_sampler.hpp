#pragma once

#include "frontend/stim.hpp"
#include "sampler/prepared_sampler.hpp"

#include <string>

namespace symft {

CircuitSamplingInput make_stim_circuit_sampling_input(
    const QuantumCircuit& circuit,
    const CircuitSamplingOptions& options = {});
CircuitSamplingInput make_stim_circuit_sampling_input_from_file(
    const std::string& path,
    const CircuitSamplingOptions& options = {});
PreparedCircuitSingleShotSampler prepare_single_shot_sampler_from_stim_file(
    const std::string& path,
    CircuitSamplingOptions options = {});
PreparedCircuitBatchSampler prepare_batch_sampler_from_stim_file(
    const std::string& path,
    CircuitSamplingOptions options = {});

} // namespace symft
