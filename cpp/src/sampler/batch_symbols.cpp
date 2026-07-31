#include "batch_internal.hpp"

namespace symft {

void check_batch_symbol_slot(const BatchFactoredExecutorState& runtime, int condition) {
    if (condition <= 0 || condition > runtime.nsymbols) {
        fail("symbolic condition exceeds batch executor symbol table");
    }
}

bool batch_symbol_assigned(const BatchFactoredExecutorState& runtime, int condition) {
    check_batch_symbol_slot(runtime, condition);
    const std::size_t word = symbol_word_index(condition);
    return word < runtime.assigned_words.size() && (runtime.assigned_words[word] & symbol_bit_mask(condition)) != 0;
}

void mark_batch_symbol_assigned_unchecked(BatchFactoredExecutorState& runtime, int condition) {
    runtime.assigned_words[symbol_word_index(condition)] |= symbol_bit_mask(condition);
}

void set_batch_symbol_false_unchecked(BatchFactoredExecutorState& runtime, int condition) {
    check_batch_symbol_slot(runtime, condition);
    mark_batch_symbol_assigned_unchecked(runtime, condition);
}

void set_batch_symbol_true_unchecked(BatchFactoredExecutorState& runtime, int condition, int shot) {
    check_batch_symbol_slot(runtime, condition);
    runtime.value_words[batch_condition_offset(runtime, condition, batch_shot_word(shot))] |= batch_shot_mask(shot);
}

void set_batch_symbol_true_all_unchecked(BatchFactoredExecutorState& runtime, int condition) {
    check_batch_symbol_slot(runtime, condition);
    mark_batch_symbol_assigned_unchecked(runtime, condition);
    const std::size_t nwords = runtime_batch_word_count(runtime);
    for (std::size_t word = 0; word < runtime.batch_words; ++word) {
        runtime.value_words[batch_condition_offset(runtime, condition, word)] =
            word < nwords ? batch_live_word_mask(runtime, word) : 0;
    }
}

void assign_batch_conditions_false(BatchFactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        set_batch_symbol_false_unchecked(runtime, condition);
    }
}

bool any_batch_assigned(BatchFactoredExecutorState& runtime, const std::vector<int>& conditions) {
    for (int condition : conditions) {
        if (batch_symbol_assigned(runtime, condition)) {
            return true;
        }
    }
    return false;
}

bool any_batch_categorical_group_assigned(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        if (any_batch_assigned(runtime, conditions)) {
            return true;
        }
    }
    return false;
}

bool batch_symbol_matches_bits(
    const BatchFactoredExecutorState& runtime,
    int condition,
    const std::vector<std::uint64_t>& bits) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < nwords) {
        fail("batch bit vector is too short");
    }
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t mask = batch_live_word_mask(runtime, word);
        const std::uint64_t actual = runtime.value_words[batch_condition_offset(runtime, condition, word)];
        if (((actual ^ bits[word]) & mask) != 0) {
            return false;
        }
    }
    return true;
}

void copy_bits_to_batch_symbol_unchecked(
    BatchFactoredExecutorState& runtime,
    int condition,
    const std::vector<std::uint64_t>& bits) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (bits.size() < nwords) {
        fail("batch bit vector is too short");
    }
    const std::size_t base = batch_condition_offset(runtime, condition, 0);
    if (nwords == runtime.batch_words && (runtime.active_shots & 63) == 0) {
        std::copy_n(bits.data(), nwords, runtime.value_words.data() + base);
        return;
    }
    for (std::size_t word = 0; word < nwords; ++word) {
        runtime.value_words[base + word] = bits[word] & batch_live_word_mask(runtime, word);
    }
    for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
        runtime.value_words[base + word] = 0;
    }
}

void assign_batch_symbol(BatchFactoredExecutorState& runtime, int condition, const std::vector<std::uint64_t>& bits) {
    check_batch_symbol_slot(runtime, condition);
    if (batch_symbol_assigned(runtime, condition)) {
        if (!batch_symbol_matches_bits(runtime, condition, bits)) {
            fail("symbolic condition was assigned inconsistent concrete batch values");
        }
        return;
    }
    mark_batch_symbol_assigned_unchecked(runtime, condition);
    copy_bits_to_batch_symbol_unchecked(runtime, condition, bits);
}

void assign_batch_symbol(
    BatchFactoredExecutorState& runtime,
    std::optional<int> condition,
    const std::vector<std::uint64_t>& bits) {
    if (condition) {
        assign_batch_symbol(runtime, *condition, bits);
    }
}

void eval_symbolic_bool_batch(
    std::vector<std::uint64_t>& out,
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime) {
    fill_batch_bits(out, runtime, plan.constant);
    if (plan.conditions.empty()) {
        return;
    }
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (plan.word_indices.empty() || plan.conditions.size() <= kBatchScalarSymbolicEvalThreshold) {
        if (nwords == 1) {
            std::uint64_t out0 = out[0];
            if (runtime.batch_words == 1) {
                for (int condition : plan.conditions) {
                    if (!batch_symbol_assigned(runtime, condition)) {
                        fail("symbolic condition expression has no concrete batch value");
                    }
                    out0 ^= runtime.value_words[static_cast<std::size_t>(condition - 1)];
                }
            } else {
                for (int condition : plan.conditions) {
                    if (!batch_symbol_assigned(runtime, condition)) {
                        fail("symbolic condition expression has no concrete batch value");
                    }
                    out0 ^= runtime.value_words[batch_condition_offset(runtime, condition, 0)];
                }
            }
            out[0] = out0;
            mask_batch_bits(out, runtime);
            return;
        }
        if (nwords == 2) {
            std::uint64_t out0 = out[0];
            std::uint64_t out1 = out[1];
            for (int condition : plan.conditions) {
                if (!batch_symbol_assigned(runtime, condition)) {
                    fail("symbolic condition expression has no concrete batch value");
                }
                const std::size_t base = batch_condition_offset(runtime, condition, 0);
                out0 ^= runtime.value_words[base];
                out1 ^= runtime.value_words[base + 1];
            }
            out[0] = out0;
            out[1] = out1;
            mask_batch_bits(out, runtime);
            return;
        }
        for (int condition : plan.conditions) {
            if (!batch_symbol_assigned(runtime, condition)) {
                fail("symbolic condition expression has no concrete batch value");
            }
            const std::size_t base = batch_condition_offset(runtime, condition, 0);
            for (std::size_t word = 0; word < nwords; ++word) {
                out[word] ^= runtime.value_words[base + word];
            }
        }
        mask_batch_bits(out, runtime);
        return;
    }
    const std::size_t max_word = static_cast<std::size_t>(plan.word_indices.back());
    if (max_word >= runtime.assigned_words.size()) {
        fail("symbolic condition expression has no concrete batch value");
    }
    std::uint64_t missing = 0;
    for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
        const std::size_t word = static_cast<std::size_t>(plan.word_indices[i]);
        missing |= plan.word_masks[i] & ~runtime.assigned_words[word];
    }
    if (missing != 0) {
        fail("symbolic condition expression has no concrete batch value");
    }
    if (nwords == 1) {
        std::uint64_t out0 = out[0];
        if (runtime.batch_words == 1) {
            for (int condition : plan.conditions) {
                out0 ^= runtime.value_words[static_cast<std::size_t>(condition - 1)];
            }
        } else {
            for (int condition : plan.conditions) {
                out0 ^= runtime.value_words[batch_condition_offset(runtime, condition, 0)];
            }
        }
        out[0] = out0;
        mask_batch_bits(out, runtime);
        return;
    }
    if (nwords == 2) {
        std::uint64_t out0 = out[0];
        std::uint64_t out1 = out[1];
        for (int condition : plan.conditions) {
            const std::size_t base = batch_condition_offset(runtime, condition, 0);
            out0 ^= runtime.value_words[base];
            out1 ^= runtime.value_words[base + 1];
        }
        out[0] = out0;
        out[1] = out1;
        mask_batch_bits(out, runtime);
        return;
    }
    for (int condition : plan.conditions) {
        const std::size_t base = batch_condition_offset(runtime, condition, 0);
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= runtime.value_words[base + word];
        }
    }
    mask_batch_bits(out, runtime);
}

void check_batch_symbolic_plan_assigned(
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime) {
    if (plan.word_indices.empty()) {
        for (int condition : plan.conditions) {
            if (!batch_symbol_assigned(runtime, condition)) {
                fail("symbolic condition expression has no concrete batch value");
            }
        }
        return;
    }
    const std::size_t max_word = static_cast<std::size_t>(plan.word_indices.back());
    if (max_word >= runtime.assigned_words.size()) {
        fail("symbolic condition expression has no concrete batch value");
    }
    std::uint64_t missing = 0;
    for (std::size_t i = 0; i < plan.word_indices.size(); ++i) {
        const std::size_t word = static_cast<std::size_t>(plan.word_indices[i]);
        missing |= plan.word_masks[i] & ~runtime.assigned_words[word];
    }
    if (missing != 0) {
        fail("symbolic condition expression has no concrete batch value");
    }
}

void xor_symbolic_bool_batch_into(
    std::vector<std::uint64_t>& out,
    const SymbolicBoolEvaluationPlan& plan,
    const BatchFactoredExecutorState& runtime) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (out.size() < runtime.batch_words) {
        out.resize(runtime.batch_words, 0);
    }
    if (plan.conditions.empty()) {
        if (plan.constant) {
            invert_batch_bits(out, runtime);
        }
        return;
    }
    const bool checked_per_condition =
        plan.word_indices.empty() || plan.conditions.size() <= kBatchScalarSymbolicEvalThreshold;
    if (!checked_per_condition) {
        check_batch_symbolic_plan_assigned(plan, runtime);
    }
    if (nwords == 1) {
        std::uint64_t out0 = out[0] ^ (plan.constant ? batch_live_word_mask(runtime, 0) : 0);
        if (runtime.batch_words == 1) {
            for (int condition : plan.conditions) {
                if (checked_per_condition && !batch_symbol_assigned(runtime, condition)) {
                    fail("symbolic condition expression has no concrete batch value");
                }
                out0 ^= runtime.value_words[static_cast<std::size_t>(condition - 1)];
            }
        } else {
            for (int condition : plan.conditions) {
                if (checked_per_condition && !batch_symbol_assigned(runtime, condition)) {
                    fail("symbolic condition expression has no concrete batch value");
                }
                out0 ^= runtime.value_words[batch_condition_offset(runtime, condition, 0)];
            }
        }
        out[0] = out0 & batch_live_word_mask(runtime, 0);
        std::fill(out.begin() + 1, out.end(), 0);
        return;
    }
    if (nwords == 2) {
        std::uint64_t out0 = out[0] ^ (plan.constant ? batch_live_word_mask(runtime, 0) : 0);
        std::uint64_t out1 = out[1] ^ (plan.constant ? batch_live_word_mask(runtime, 1) : 0);
        for (int condition : plan.conditions) {
            if (checked_per_condition && !batch_symbol_assigned(runtime, condition)) {
                fail("symbolic condition expression has no concrete batch value");
            }
            const std::size_t base = batch_condition_offset(runtime, condition, 0);
            out0 ^= runtime.value_words[base];
            out1 ^= runtime.value_words[base + 1];
        }
        out[0] = out0 & batch_live_word_mask(runtime, 0);
        out[1] = out1 & batch_live_word_mask(runtime, 1);
        std::fill(out.begin() + 2, out.end(), 0);
        return;
    }
    if (plan.constant) {
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= batch_live_word_mask(runtime, word);
        }
    }
    for (int condition : plan.conditions) {
        if (checked_per_condition && !batch_symbol_assigned(runtime, condition)) {
            fail("symbolic condition expression has no concrete batch value");
        }
        const std::size_t base = batch_condition_offset(runtime, condition, 0);
        for (std::size_t word = 0; word < nwords; ++word) {
            out[word] ^= runtime.value_words[base + word];
        }
    }
    mask_batch_bits(out, runtime);
    std::fill(out.begin() + static_cast<std::ptrdiff_t>(nwords), out.end(), 0);
}

void ensure_batch_measurement_storage(BatchFactoredExecutorState& runtime, int record) {
    if (record <= runtime.nrecords) {
        return;
    }
    std::vector<std::uint64_t> next(static_cast<std::size_t>(record) * runtime.batch_words, 0);
    for (int old_record = 1; old_record <= runtime.nrecords; ++old_record) {
        const std::size_t old_base = batch_record_offset(runtime, old_record, 0);
        const std::size_t new_base = static_cast<std::size_t>(old_record - 1) * runtime.batch_words;
        std::copy(
            runtime.measurement_words.begin() + static_cast<std::ptrdiff_t>(old_base),
            runtime.measurement_words.begin() + static_cast<std::ptrdiff_t>(old_base + runtime.batch_words),
            next.begin() + static_cast<std::ptrdiff_t>(new_base));
    }
    runtime.nrecords = record;
    runtime.measurement_words = std::move(next);
}

void ensure_batch_detector_storage(BatchFactoredExecutorState& runtime, int detector) {
    if (detector <= runtime.ndetectors) {
        return;
    }
    std::vector<std::uint64_t> next(static_cast<std::size_t>(detector) * runtime.batch_words, 0);
    for (int old_detector = 1; old_detector <= runtime.ndetectors; ++old_detector) {
        const std::size_t old_base = batch_detector_offset(runtime, old_detector, 0);
        const std::size_t new_base = static_cast<std::size_t>(old_detector - 1) * runtime.batch_words;
        std::copy(
            runtime.detector_words.begin() + static_cast<std::ptrdiff_t>(old_base),
            runtime.detector_words.begin() + static_cast<std::ptrdiff_t>(old_base + runtime.batch_words),
            next.begin() + static_cast<std::ptrdiff_t>(new_base));
    }
    runtime.ndetectors = detector;
    runtime.detector_words = std::move(next);
}

void write_batch_measurement_record(
    BatchFactoredExecutorState& runtime,
    std::optional<int> record,
    const std::vector<std::uint64_t>& outcome_bits,
    std::optional<int> record_condition) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (record) {
        if (*record <= 0) {
            fail("measurement record id must be positive");
        }
        ensure_batch_measurement_storage(runtime, *record);
        const std::size_t base = batch_record_offset(runtime, *record, 0);
        if (nwords == runtime.batch_words && (runtime.active_shots & 63) == 0) {
            std::copy_n(outcome_bits.data(), nwords, runtime.measurement_words.data() + base);
        } else {
            for (std::size_t word = 0; word < nwords; ++word) {
                runtime.measurement_words[base + word] = outcome_bits[word] & batch_live_word_mask(runtime, word);
            }
            for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
                runtime.measurement_words[base + word] = 0;
            }
        }
    }
    assign_batch_symbol(runtime, record_condition, outcome_bits);
}

void write_batch_detector_record(
    BatchFactoredExecutorState& runtime,
    int detector,
    const std::vector<std::uint64_t>& outcome_bits) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    if (detector <= 0) {
        fail("detector id must be positive");
    }
    if (!runtime.store_detector_records) {
        for (std::size_t word = 0; word < nwords; ++word) {
            runtime.detector_any_words[word] |= outcome_bits[word] & batch_live_word_mask(runtime, word);
        }
        return;
    }
    ensure_batch_detector_storage(runtime, detector);
    const std::size_t base = batch_detector_offset(runtime, detector, 0);
    if (nwords == runtime.batch_words && (runtime.active_shots & 63) == 0) {
        std::copy_n(outcome_bits.data(), nwords, runtime.detector_words.data() + base);
        for (std::size_t word = 0; word < nwords; ++word) {
            runtime.detector_any_words[word] |= outcome_bits[word];
        }
    } else {
        for (std::size_t word = 0; word < nwords; ++word) {
            const std::uint64_t bits = outcome_bits[word] & batch_live_word_mask(runtime, word);
            runtime.detector_words[base + word] = bits;
            runtime.detector_any_words[word] |= bits;
        }
        for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
            runtime.detector_words[base + word] = 0;
        }
    }
}

void assign_presampled_exogenous_batch(
    BatchFactoredExecutorState& runtime,
    const PresampledExogenous& samples) {
    if (runtime.active_shots > samples.nshots || runtime.nsymbols != samples.nsymbols ||
        samples.exogenous_assigned_words.size() != samples.nwords ||
        samples.value_words.size() != static_cast<std::size_t>(samples.nshots) * samples.nwords) {
        fail("presampled exogenous table does not match batch executor");
    }
    for (std::size_t symbol_word = 0; symbol_word < samples.nwords; ++symbol_word) {
        runtime.assigned_words[symbol_word] |= samples.exogenous_assigned_words[symbol_word];
    }
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const std::size_t shot_word = batch_shot_word(shot);
        const std::uint64_t shot_mask = batch_shot_mask(shot);
        const std::size_t base = static_cast<std::size_t>(shot) * samples.nwords;
        for (std::size_t symbol_word = 0; symbol_word < samples.nwords; ++symbol_word) {
            std::uint64_t row = samples.value_words[base + symbol_word] &
                                samples.exogenous_assigned_words[symbol_word];
            while (row != 0) {
                const int bit = trailing_zeros64(row);
                const int condition = static_cast<int>(symbol_word * 64 + static_cast<std::size_t>(bit) + 1);
                if (condition <= runtime.nsymbols) {
                    runtime.value_words[batch_condition_offset(runtime, condition, shot_word)] |= shot_mask;
                }
                row &= row - 1;
            }
        }
    }
}

std::uint64_t packed_condition_slice_word(
    const PackedPresampledExogenous& samples,
    int condition,
    int first_sample_shot,
    int active_shots,
    std::size_t dest_word) {
    const int bit_offset = first_sample_shot & 63;
    const std::size_t src_word = static_cast<std::size_t>(first_sample_shot >> 6) + dest_word;
    const std::size_t base = static_cast<std::size_t>(condition - 1) * samples.shot_words;
    std::uint64_t out = 0;
    if (src_word < samples.shot_words) {
        out = samples.value_words[base + src_word] >> bit_offset;
    }
    if (bit_offset != 0 && src_word + 1 < samples.shot_words) {
        out |= samples.value_words[base + src_word + 1] << (64 - bit_offset);
    }
    return out & low_bits_mask(active_shots - static_cast<int>(dest_word << 6));
}

void copy_packed_exogenous_condition_to_batch(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int condition,
    int first_sample_shot) {
    const std::size_t nwords = runtime_batch_word_count(runtime);
    const std::size_t base = batch_condition_offset(runtime, condition, 0);
    for (std::size_t word = 0; word < nwords; ++word) {
        runtime.value_words[base + word] = packed_condition_slice_word(
            samples,
            condition,
            first_sample_shot,
            runtime.active_shots,
            word);
    }
    for (std::size_t word = nwords; word < runtime.batch_words; ++word) {
        runtime.value_words[base + word] = 0;
    }
}

void assign_presampled_exogenous_batch(
    BatchFactoredExecutorState& runtime,
    const PackedPresampledExogenous& samples,
    int first_sample_shot) {
    if (first_sample_shot < 0 ||
        runtime.active_shots > samples.nshots - first_sample_shot ||
        runtime.nsymbols != samples.nsymbols ||
        samples.exogenous_assigned_words.size() != symbol_word_count(samples.nsymbols) ||
        samples.value_words.size() != static_cast<std::size_t>(samples.nsymbols) * samples.shot_words) {
        fail("packed presampled exogenous table does not match batch executor");
    }
    for (std::size_t symbol_word = 0; symbol_word < samples.exogenous_assigned_words.size(); ++symbol_word) {
        std::uint64_t assigned = samples.exogenous_assigned_words[symbol_word];
        runtime.assigned_words[symbol_word] |= assigned;
        while (assigned != 0) {
            const int bit = trailing_zeros64(assigned);
            const int condition = static_cast<int>(symbol_word * 64 + static_cast<std::size_t>(bit) + 1);
            if (condition <= runtime.nsymbols) {
                copy_packed_exogenous_condition_to_batch(runtime, samples, condition, first_sample_shot);
            }
            assigned &= assigned - 1;
        }
    }
}

void sample_categorical_distribution_batch(
    BatchFactoredExecutorState& runtime,
    const std::vector<int>& conditions,
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    if (static_cast<int>(conditions.size()) != nbits) {
        fail("categorical condition count does not match assignment bit count");
    }
    bool all_assigned = true;
    bool any_assigned = false;
    for (int condition : conditions) {
        const bool assigned = batch_symbol_assigned(runtime, condition);
        all_assigned = all_assigned && assigned;
        any_assigned = any_assigned || assigned;
    }
    if (all_assigned) {
        return;
    }
    if (any_assigned) {
        fail("categorical symbolic distribution was only partially preassigned");
    }

    assign_batch_conditions_false(runtime, conditions);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        const int row = sample_categorical_row(runtime.rng_state, probabilities);
        const auto& assignment = assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                set_batch_symbol_true_unchecked(runtime, conditions[bit_idx], shot);
            }
        }
    }
}

void sample_categorical_distribution_batch(
    BatchFactoredExecutorState& runtime,
    const SymbolicCategoricalDistribution& distribution) {
    sample_categorical_distribution_batch(
        runtime,
        distribution.conditions,
        distribution.nbits,
        distribution.assignments,
        distribution.probabilities);
}

void assign_batch_categorical_group_false(
    BatchFactoredExecutorState& runtime,
    const std::vector<std::vector<int>>& condition_sets) {
    for (const auto& conditions : condition_sets) {
        assign_batch_conditions_false(runtime, conditions);
    }
}

void sample_rare_categorical_group_batch(
    BatchFactoredExecutorState& runtime,
    const RareCategoricalSampleGroup& group) {
    if (any_batch_categorical_group_assigned(runtime, group.conditions)) {
        for (const auto& conditions : group.conditions) {
            sample_categorical_distribution_batch(runtime, conditions, group.nbits, group.assignments, group.probabilities);
        }
        return;
    }

    assign_batch_categorical_group_false(runtime, group.conditions);
    const int nsets = static_cast<int>(group.conditions.size());
    if (group.event_probability <= 0.0 || nsets == 0) {
        return;
    }
    const std::int64_t total_draws =
        static_cast<std::int64_t>(runtime.active_shots) * static_cast<std::int64_t>(nsets);
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(runtime.rng_state, group.event_probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nsets);
        const int set_idx = static_cast<int>(draw % nsets);
        const int row =
            group.event_rows[static_cast<std::size_t>(sample_categorical_row(runtime.rng_state, group.event_probabilities))];
        const auto& conditions = group.conditions[static_cast<std::size_t>(set_idx)];
        const auto& assignment = group.assignments[static_cast<std::size_t>(row)];
        for (std::size_t bit_idx = 0; bit_idx < conditions.size(); ++bit_idx) {
            if (packed_bit(assignment, static_cast<int>(bit_idx))) {
                set_batch_symbol_true_unchecked(runtime, conditions[bit_idx], shot);
            }
        }
        ++draw;
    }
}

void sample_bernoulli_condition_batch(BatchFactoredExecutorState& runtime, int condition, double probability) {
    if (batch_symbol_assigned(runtime, condition)) {
        return;
    }
    const double p = check_probability(probability);
    if (p <= 0.0) {
        set_batch_symbol_false_unchecked(runtime, condition);
        return;
    }
    if (p >= 1.0) {
        set_batch_symbol_true_all_unchecked(runtime, condition);
        return;
    }

    set_batch_symbol_false_unchecked(runtime, condition);
    for (int shot = 0; shot < runtime.active_shots; ++shot) {
        if (sample_bernoulli(runtime.rng_state, p)) {
            set_batch_symbol_true_unchecked(runtime, condition, shot);
        }
    }
}

void sample_low_probability_bernoulli_group_batch(
    BatchFactoredExecutorState& runtime,
    const BernoulliSampleGroup& group) {
    if (any_batch_assigned(runtime, group.conditions)) {
        for (int condition : group.conditions) {
            sample_bernoulli_condition_batch(runtime, condition, group.probability);
        }
        return;
    }

    assign_batch_conditions_false(runtime, group.conditions);
    const int nconditions = static_cast<int>(group.conditions.size());
    if (group.probability <= 0.0 || nconditions == 0) {
        return;
    }
    const std::int64_t total_draws =
        static_cast<std::int64_t>(runtime.active_shots) * static_cast<std::int64_t>(nconditions);
    std::int64_t draw = 0;
    while (true) {
        const auto gap = static_cast<std::int64_t>(sample_geometric_gap(runtime.rng_state, group.probability));
        if (gap >= total_draws - draw) {
            return;
        }
        draw += gap;
        const int shot = static_cast<int>(draw / nconditions);
        const int condition_idx = static_cast<int>(draw % nconditions);
        set_batch_symbol_true_unchecked(runtime, group.conditions[static_cast<std::size_t>(condition_idx)], shot);
        ++draw;
    }
}

void sample_exogenous_symbols_batch(BatchFactoredExecutorState& runtime, const FactoredInstructionProgram& program) {
    for (const auto& distribution : program.sampled_categorical_distributions) {
        sample_categorical_distribution_batch(runtime, distribution);
    }
    for (const auto& group : program.sampled_rare_categorical_groups) {
        sample_rare_categorical_group_batch(runtime, group);
    }
    for (std::size_t i = 0; i < program.sampled_bernoulli_conditions.size(); ++i) {
        sample_bernoulli_condition_batch(
            runtime,
            program.sampled_bernoulli_conditions[i],
            program.sampled_bernoulli_probabilities[i]);
    }
    for (const auto& group : program.sampled_low_probability_bernoulli_groups) {
        sample_low_probability_bernoulli_group_batch(runtime, group);
    }
}

} // namespace symft
