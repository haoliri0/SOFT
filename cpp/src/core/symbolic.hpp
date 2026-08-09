#pragma once

#include "core/common.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <string>
#include <unordered_map>

namespace symft {

struct SymbolicBool {
    bool constant = false;
    std::vector<int> conditions;

    SymbolicBool() = default;
    explicit SymbolicBool(bool constant);
    SymbolicBool(bool constant, std::vector<int> conditions);

    int max_condition() const;
    std::string str() const;
};

bool operator==(const SymbolicBool& lhs, const SymbolicBool& rhs);
bool operator!=(const SymbolicBool& lhs, const SymbolicBool& rhs);
SymbolicBool symbolic_bool(int condition);
SymbolicBool operator!(const SymbolicBool& expr);
SymbolicBool xor_bool(const SymbolicBool& lhs, const SymbolicBool& rhs);
SymbolicBool xor_bool(const SymbolicBool& lhs, bool rhs);
SymbolicBool xor_bool(bool lhs, const SymbolicBool& rhs);
std::ostream& operator<<(std::ostream& out, const SymbolicBool& expr);

struct SymbolicBoolEvaluationPlan {
    bool constant = false;
    std::vector<int> conditions;
    std::vector<int> word_indices;
    std::vector<std::uint64_t> word_masks;

    SymbolicBoolEvaluationPlan() = default;
    explicit SymbolicBoolEvaluationPlan(const SymbolicBool& expr);
};

struct SymbolicCategoricalDistribution {
    int nbits = 0;
    std::vector<int> conditions;
    std::vector<std::vector<std::uint64_t>> assignments;
    std::vector<double> probabilities;
};

struct SymbolicContext {
    int next_condition = 1;
    std::map<int, double> bernoulli_probabilities;
    std::vector<SymbolicCategoricalDistribution> categorical_distributions;
    std::unordered_map<int, std::size_t> condition_to_categorical;

    SymbolicContext() = default;
    explicit SymbolicContext(int next_condition);

    void bump_next_condition(int condition);
    void bump_next_condition(const SymbolicBool& expr);
    int fresh_condition();
    int fresh_bernoulli_condition(double probability);
    SymbolicBool fresh_bernoulli_bool(double probability);
    std::vector<int> fresh_categorical_conditions(
        int nbits,
        const std::vector<std::vector<std::uint64_t>>& assignments,
        const std::vector<double>& probabilities);
    std::vector<SymbolicBool> fresh_categorical_bools(
        int nbits,
        const std::vector<std::vector<std::uint64_t>>& assignments,
        const std::vector<double>& probabilities);
};

namespace detail {

// Reduces symbolic Boolean expressions using known XOR relations while
// minimizing the number of packed words, then the number of conditions.
class SymbolicRelationReducer {
  public:
    void add(SymbolicBool relation);
    bool empty() const;
    SymbolicBool reduce(SymbolicBool expression) const;

  private:
    struct PackedRelation {
        SymbolicBool expression;
        std::vector<std::size_t> word_indices;
        std::vector<std::uint64_t> word_masks;
    };

    std::vector<PackedRelation> relations_;
    std::vector<std::vector<std::size_t>> relation_index_;
    // -1 denotes an unfixed condition; the other values are Boolean.
    std::vector<std::int8_t> fixed_conditions_;
    std::size_t word_count_ = 0;

    mutable std::vector<std::uint64_t> words_;
    mutable std::vector<std::size_t> candidates_;
    mutable std::vector<std::uint8_t> candidate_marks_;
};

} // namespace detail

} // namespace symft
