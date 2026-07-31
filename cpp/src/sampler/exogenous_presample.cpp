#include "sampler/exogenous.hpp"

#include "core/internal.hpp"
#include "sampler/random.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace symft {
using namespace detail;

namespace {

std::size_t presampled_word_index(std::size_t nwords, int shot_index, std::size_t word) {
    return static_cast<std::size_t>(shot_index) * nwords + word;
}

std::size_t packed_shot_word_count(int shots) {
    return shots <= 0 ? 0 : static_cast<std::size_t>((shots + 63) >> 6);
}

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t packed_live_word_mask(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

std::size_t packed_condition_offset(std::size_t shot_words, int condition, std::size_t shot_word) {
    return static_cast<std::size_t>(condition - 1) * shot_words + shot_word;
}

void xor_presampled_condition(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    int shot_index,
    int condition) {
    const std::size_t word = symbol_word_index(condition);
    value_words[presampled_word_index(nwords, shot_index, word)] ^= symbol_bit_mask(condition);
}

void xor_packed_presampled_condition(
    std::vector<std::uint64_t>& value_words,
    std::size_t shot_words,
    int shot_index,
    int condition) {
    if (shot_words == 0) {
        return;
    }
    const std::size_t word = static_cast<std::size_t>(shot_index >> 6);
    const std::uint64_t mask = std::uint64_t{1} << (shot_index & 63);
    value_words[packed_condition_offset(shot_words, condition, word)] ^= mask;
}

void or_low_probability_bits_packed(
    std::uint64_t* row,
    std::size_t shot_words,
    std::uint64_t& rng_state,
    double probability,
    int shots) {
    if (probability <= 0.0 || shots <= 0) {
        return;
    }
    int draw = 0;
    while (true) {
        const int gap = static_cast<int>(sample_geometric_gap(rng_state, probability));
        if (gap >= shots - draw) {
            return;
        }
        draw += gap;
        const std::size_t word = static_cast<std::size_t>(draw >> 6);
        if (word >= shot_words) {
            return;
        }
        row[word] |= std::uint64_t{1} << (draw & 63);
        ++draw;
    }
}

void generate_packed_biased_bits(
    std::uint64_t* row,
    std::size_t shot_words,
    std::uint64_t& rng_state,
    double probability,
    int shots) {
    if (shots <= 0 || shot_words == 0 || probability <= 0.0) {
        return;
    }
    if (probability >= 1.0) {
        for (std::size_t word = 0; word < shot_words; ++word) {
            row[word] = packed_live_word_mask(shots, word);
        }
        return;
    }

    const bool invert = probability > 0.5;
    const double p = invert ? 1.0 - probability : probability;
    if (p <= 0.0) {
        for (std::size_t word = 0; word < shot_words; ++word) {
            row[word] = invert ? packed_live_word_mask(shots, word) : 0;
        }
        return;
    }
    if (p == 0.5) {
        for (std::size_t word = 0; word < shot_words; ++word) {
            row[word] = next_random_u64(rng_state) & packed_live_word_mask(shots, word);
        }
    } else if (p < kLowProbabilitySampleThreshold) {
        or_low_probability_bits_packed(row, shot_words, rng_state, p, shots);
    } else {
        // Decompose p into bit-sliced fair-coin prefixes, then add the
        // remaining probability mass with a sparse geometric pass.
        constexpr int kCoinFlips = 8;
        constexpr double kBuckets = static_cast<double>(std::uint64_t{1} << kCoinFlips);
        const double scaled = p * kBuckets;
        const auto raw_top_bits = static_cast<std::uint64_t>(scaled);
        const auto top_bits = raw_top_bits < (std::uint64_t{1} << (kCoinFlips - 1))
                                  ? raw_top_bits
                                  : (std::uint64_t{1} << (kCoinFlips - 1)) - 1;
        const double p_truncated = static_cast<double>(top_bits) / kBuckets;
        for (std::size_t word = 0; word < shot_words; ++word) {
            std::uint64_t alive = next_random_u64(rng_state);
            std::uint64_t result = 0;
            for (int bit = kCoinFlips - 2; bit >= 0; --bit) {
                const std::uint64_t shoot = next_random_u64(rng_state);
                if (((top_bits >> bit) & 1ULL) != 0) {
                    result |= shoot & alive;
                }
                alive &= ~shoot;
            }
            row[word] = result & packed_live_word_mask(shots, word);
        }

        const double p_leftover = p - p_truncated;
        if (p_leftover > 0.0) {
            or_low_probability_bits_packed(
                row,
                shot_words,
                rng_state,
                p_leftover / (1.0 - p_truncated),
                shots);
        }
    }

    if (invert) {
        for (std::size_t word = 0; word < shot_words; ++word) {
            row[word] = (~row[word]) & packed_live_word_mask(shots, word);
        }
    }
}

void mark_exogenous_conditions(std::vector<std::uint64_t>& words, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        const std::size_t word = symbol_word_index(condition);
        if (word >= words.size()) {
            fail("exogenous condition exceeds program symbol table");
        }
        words[word] |= symbol_bit_mask(condition);
    }
}

std::vector<std::uint64_t> exogenous_assigned_words(const FactoredInstructionProgram& program) {
    std::vector<std::uint64_t> words(symbol_word_count(program.nsymbols), 0);
    for (const auto& distribution : program.sampled_categorical_distributions) {
        mark_exogenous_conditions(words, distribution.conditions);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        for (const auto& conditions : group.conditions) {
            mark_exogenous_conditions(words, conditions);
        }
    }
    mark_exogenous_conditions(words, program.sampled_bernoulli_conditions);
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        mark_exogenous_conditions(words, group.conditions);
    }
    return words;
}

void presample_categorical_distribution(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    const SymbolicCategoricalDistribution& distribution,
    int shots) {
    for (int shot = 0; shot < shots; ++shot) {
        const int row = sample_categorical_row(rng_state, distribution.probabilities);
        const auto& assignment = distribution.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < distribution.conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                xor_presampled_condition(value_words, nwords, shot, distribution.conditions[bit_idx]);
            }
        }
    }
}

void presample_categorical_distribution_packed(
    std::vector<std::uint64_t>& value_words,
    std::size_t shot_words,
    std::uint64_t& rng_state,
    const SymbolicCategoricalDistribution& distribution,
    int shots) {
    for (int shot = 0; shot < shots; ++shot) {
        const int row = sample_categorical_row(rng_state, distribution.probabilities);
        const auto& assignment = distribution.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < distribution.conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                xor_packed_presampled_condition(value_words, shot_words, shot, distribution.conditions[bit_idx]);
            }
        }
    }
}

void presample_rare_categorical_group(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    const RareCategoricalSampleGroup& group,
    int shots) {
    const int nsets = static_cast<int>(group.conditions.size());
    const std::int64_t total_draws = static_cast<std::int64_t>(shots) * static_cast<std::int64_t>(nsets);
    if (group.event_probability <= 0.0 || nsets == 0) {
        return;
    }
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(rng_state, group.event_probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nsets);
        const int set_idx = static_cast<int>(draw % nsets);
        const int row = group.event_rows[static_cast<std::size_t>(sample_categorical_row(rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(set_idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                xor_presampled_condition(value_words, nwords, shot, conditions[bit_idx]);
            }
        }
        ++draw;
    }
}

void presample_rare_categorical_group_packed(
    std::vector<std::uint64_t>& value_words,
    std::size_t shot_words,
    std::uint64_t& rng_state,
    const RareCategoricalSampleGroup& group,
    int shots) {
    const int nsets = static_cast<int>(group.conditions.size());
    const std::int64_t total_draws = static_cast<std::int64_t>(shots) * static_cast<std::int64_t>(nsets);
    if (group.event_probability <= 0.0 || nsets == 0) {
        return;
    }
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(rng_state, group.event_probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nsets);
        const int set_idx = static_cast<int>(draw % nsets);
        const int row =
            group.event_rows[static_cast<std::size_t>(sample_categorical_row(rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(set_idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                xor_packed_presampled_condition(value_words, shot_words, shot, conditions[bit_idx]);
            }
        }
        ++draw;
    }
}

void presample_bernoulli_condition(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    int condition,
    double probability,
    int shots) {
    const double p = check_probability(probability);
    if (p <= 0.0) {
        return;
    }
    if (p >= 1.0) {
        for (int shot = 0; shot < shots; ++shot) {
            xor_presampled_condition(value_words, nwords, shot, condition);
        }
        return;
    }
    if (p < kLowProbabilitySampleThreshold) {
        int draw = 0;
        while (true) {
            const int gap = static_cast<int>(sample_geometric_gap(rng_state, p));
            if (gap >= shots - draw) {
                return;
            }
            draw += gap;
            xor_presampled_condition(value_words, nwords, draw, condition);
            ++draw;
        }
    }
    for (int shot = 0; shot < shots; ++shot) {
        if (sample_bernoulli(rng_state, p)) {
            xor_presampled_condition(value_words, nwords, shot, condition);
        }
    }
}

void presample_bernoulli_condition_packed(
    std::vector<std::uint64_t>& value_words,
    std::size_t shot_words,
    std::uint64_t& rng_state,
    int condition,
    double probability,
    int shots) {
    const double p = check_probability(probability);
    if (p <= 0.0) {
        return;
    }
    generate_packed_biased_bits(
        value_words.data() + packed_condition_offset(shot_words, condition, 0),
        shot_words,
        rng_state,
        p,
        shots);
}

void presample_low_probability_bernoulli_group(
    std::vector<std::uint64_t>& value_words,
    std::size_t nwords,
    std::uint64_t& rng_state,
    const BernoulliSampleGroup& group,
    int shots) {
    const int nconditions = static_cast<int>(group.conditions.size());
    const std::int64_t total_draws = static_cast<std::int64_t>(shots) * static_cast<std::int64_t>(nconditions);
    if (group.probability <= 0.0 || nconditions == 0) {
        return;
    }
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(rng_state, group.probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nconditions);
        const int condition_idx = static_cast<int>(draw % nconditions);
        xor_presampled_condition(value_words, nwords, shot, group.conditions[static_cast<std::size_t>(condition_idx)]);
        ++draw;
    }
}

void presample_low_probability_bernoulli_group_packed(
    std::vector<std::uint64_t>& value_words,
    std::size_t shot_words,
    std::uint64_t& rng_state,
    const BernoulliSampleGroup& group,
    int shots) {
    const int nconditions = static_cast<int>(group.conditions.size());
    const std::int64_t total_draws = static_cast<std::int64_t>(shots) * static_cast<std::int64_t>(nconditions);
    if (group.probability <= 0.0 || nconditions == 0) {
        return;
    }
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(rng_state, group.probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nconditions);
        const int condition_idx = static_cast<int>(draw % nconditions);
        xor_packed_presampled_condition(
            value_words,
            shot_words,
            shot,
            group.conditions[static_cast<std::size_t>(condition_idx)]);
        ++draw;
    }
}

} // namespace

void presample_exogenous_in_place(
    PresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    prepare_presampled_exogenous(samples, program);
    resample_prepared_exogenous_in_place(samples, program, shots, seed);
}

void presample_exogenous_packed_in_place(
    PackedPresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    prepare_presampled_exogenous_packed(samples, program);
    resample_prepared_exogenous_packed_in_place(samples, program, shots, seed);
}

void prepare_presampled_exogenous(PresampledExogenous& samples, const FactoredInstructionProgram& program) {
    samples.nshots = 0;
    samples.nsymbols = program.nsymbols;
    samples.nwords = symbol_word_count(program.nsymbols);
    samples.exogenous_assigned_words = exogenous_assigned_words(program);
    samples.value_words.clear();
    samples.next_rng_state = 0;
}

void prepare_presampled_exogenous_packed(
    PackedPresampledExogenous& samples,
    const FactoredInstructionProgram& program) {
    samples.nshots = 0;
    samples.nsymbols = program.nsymbols;
    samples.shot_words = 0;
    samples.exogenous_assigned_words = exogenous_assigned_words(program);
    samples.value_words.clear();
    samples.next_rng_state = 0;
}

void resample_prepared_exogenous_in_place(
    PresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    if (shots < 0) {
        fail("presampled shot count must be nonnegative");
    }
    if (samples.nsymbols != program.nsymbols ||
        samples.nwords != symbol_word_count(program.nsymbols) ||
        samples.exogenous_assigned_words.size() != samples.nwords) {
        fail("presampled exogenous storage was not prepared for this program");
    }
    samples.nshots = shots;
    samples.value_words.assign(static_cast<std::size_t>(shots) * samples.nwords, 0);

    std::uint64_t rng_state = seed;
    for (const auto& distribution : program.sampled_categorical_distributions) {
        presample_categorical_distribution(samples.value_words, samples.nwords, rng_state, distribution, shots);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        presample_rare_categorical_group(samples.value_words, samples.nwords, rng_state, group, shots);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        presample_bernoulli_condition(
            samples.value_words,
            samples.nwords,
            rng_state,
            program.sampled_bernoulli_conditions[i],
            program.sampled_bernoulli_probabilities[i],
            shots);
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        presample_low_probability_bernoulli_group(samples.value_words, samples.nwords, rng_state, group, shots);
    }
    samples.next_rng_state = rng_state;
}

void resample_prepared_exogenous_packed_in_place(
    PackedPresampledExogenous& samples,
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    if (shots < 0) {
        fail("presampled shot count must be nonnegative");
    }
    if (samples.nsymbols != program.nsymbols ||
        samples.exogenous_assigned_words.size() != symbol_word_count(program.nsymbols)) {
        fail("packed presampled exogenous storage was not prepared for this program");
    }
    samples.nshots = shots;
    samples.shot_words = packed_shot_word_count(shots);
    samples.value_words.assign(static_cast<std::size_t>(program.nsymbols) * samples.shot_words, 0);

    std::uint64_t rng_state = seed;
    for (const auto& distribution : program.sampled_categorical_distributions) {
        presample_categorical_distribution_packed(
            samples.value_words,
            samples.shot_words,
            rng_state,
            distribution,
            shots);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        presample_rare_categorical_group_packed(
            samples.value_words,
            samples.shot_words,
            rng_state,
            group,
            shots);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        presample_bernoulli_condition_packed(
            samples.value_words,
            samples.shot_words,
            rng_state,
            program.sampled_bernoulli_conditions[i],
            program.sampled_bernoulli_probabilities[i],
            shots);
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        presample_low_probability_bernoulli_group_packed(
            samples.value_words,
            samples.shot_words,
            rng_state,
            group,
            shots);
    }
    samples.next_rng_state = rng_state;
}

PresampledExogenous presample_exogenous(const FactoredInstructionProgram& program, int shots, std::uint64_t seed) {
    PresampledExogenous samples;
    presample_exogenous_in_place(samples, program, shots, seed);
    return samples;
}

PackedPresampledExogenous presample_exogenous_packed(
    const FactoredInstructionProgram& program,
    int shots,
    std::uint64_t seed) {
    PackedPresampledExogenous samples;
    presample_exogenous_packed_in_place(samples, program, shots, seed);
    return samples;
}

} // namespace symft
