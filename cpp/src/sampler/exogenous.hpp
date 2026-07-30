#pragma once

#include "factored/factored.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace symft {

struct PresampledExogenous {
    int nshots = 0;
    int nsymbols = 0;
    std::size_t nwords = 0;
    std::uint64_t next_rng_state = 0;
    std::vector<std::uint64_t> exogenous_assigned_words;
    std::vector<std::uint64_t> value_words;
};

struct PackedPresampledExogenous {
    int nshots = 0;
    int nsymbols = 0;
    std::size_t shot_words = 0;
    std::uint64_t next_rng_state = 0;
    std::vector<std::uint64_t> exogenous_assigned_words;
    // Symbol-major packed layout:
    // value_words[(condition - 1) * shot_words + shot_word].
    std::vector<std::uint64_t> value_words;
};

PresampledExogenous presample_exogenous(const FactoredInstructionProgram& program, int shots, std::uint64_t seed = 1);
PackedPresampledExogenous presample_exogenous_packed(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed = 1);
void prepare_presampled_exogenous(PresampledExogenous& samples, const FactoredInstructionProgram& program);
void prepare_presampled_exogenous_packed(
    PackedPresampledExogenous& samples,
    const FactoredInstructionProgram& program);
void presample_exogenous_in_place(
    PresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed = 1);
void presample_exogenous_packed_in_place(
    PackedPresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed = 1);
void resample_prepared_exogenous_in_place(
    PresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed = 1);
void resample_prepared_exogenous_packed_in_place(
    PackedPresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed = 1);

} // namespace symft
