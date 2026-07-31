#pragma once

#include "core/common.hpp"

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

} // namespace symft
