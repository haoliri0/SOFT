#include "core/symbolic.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <sstream>

namespace symft {
using namespace detail;

SymbolicBool::SymbolicBool(bool constant_) : constant(constant_) {}

SymbolicBool::SymbolicBool(bool constant_, std::vector<int> conditions_)
    : constant(constant_), conditions(normalize_conditions(conditions_)) {}

int SymbolicBool::max_condition() const {
    return conditions.empty() ? 0 : conditions.back();
}

std::string SymbolicBool::str() const {
    std::ostringstream out;
    bool first = true;
    if (constant) {
        out << "1";
        first = false;
    }
    for (int condition : conditions) {
        if (!first) {
            out << " xor ";
        }
        out << "s" << condition;
        first = false;
    }
    if (first) {
        out << "0";
    }
    return out.str();
}

bool operator==(const SymbolicBool& lhs, const SymbolicBool& rhs) {
    return lhs.constant == rhs.constant && lhs.conditions == rhs.conditions;
}

bool operator!=(const SymbolicBool& lhs, const SymbolicBool& rhs) {
    return !(lhs == rhs);
}

SymbolicBool symbolic_bool(int condition) {
    return SymbolicBool(false, {condition});
}

SymbolicBool operator!(const SymbolicBool& expr) {
    return SymbolicBool(!expr.constant, expr.conditions);
}

SymbolicBool xor_bool(const SymbolicBool& lhs, const SymbolicBool& rhs) {
    std::vector<int> conditions = lhs.conditions;
    for (int condition : rhs.conditions) {
        toggle_condition(conditions, condition);
    }
    return SymbolicBool(lhs.constant != rhs.constant, conditions);
}

SymbolicBool xor_bool(const SymbolicBool& lhs, bool rhs) {
    return SymbolicBool(lhs.constant != rhs, lhs.conditions);
}

SymbolicBool xor_bool(bool lhs, const SymbolicBool& rhs) {
    return xor_bool(rhs, lhs);
}

std::ostream& operator<<(std::ostream& out, const SymbolicBool& expr) {
    out << expr.str();
    return out;
}

SymbolicBoolEvaluationPlan::SymbolicBoolEvaluationPlan(const SymbolicBool& expr)
    : constant(expr.constant), conditions(expr.conditions) {
    for (int condition : conditions) {
        const std::size_t word = symbol_word_index(condition);
        const std::uint64_t mask = symbol_bit_mask(condition);
        if (word_indices.empty() || word_indices.back() != static_cast<int>(word)) {
            word_indices.push_back(static_cast<int>(word));
            word_masks.push_back(mask);
        } else {
            word_masks.back() |= mask;
        }
    }
}

SymbolicContext::SymbolicContext(int next_condition_) : next_condition(next_condition_) {
    if (next_condition <= 0) {
        fail("next condition id must be positive");
    }
}

void SymbolicContext::bump_next_condition(int condition) {
    if (condition < 0) {
        fail("condition id must be nonnegative");
    }
    next_condition = std::max(next_condition, condition + 1);
}

void SymbolicContext::bump_next_condition(const SymbolicBool& expr) {
    bump_next_condition(expr.max_condition());
}

int SymbolicContext::fresh_condition() {
    return next_condition++;
}

int SymbolicContext::fresh_bernoulli_condition(double probability) {
    const double p = check_probability(probability);
    const int condition = fresh_condition();
    if (condition_to_categorical.find(condition) != condition_to_categorical.end()) {
        fail("condition already belongs to a categorical distribution");
    }
    bernoulli_probabilities[condition] = p;
    return condition;
}

SymbolicBool SymbolicContext::fresh_bernoulli_bool(double probability) {
    return symbolic_bool(fresh_bernoulli_condition(probability));
}

std::vector<int> SymbolicContext::fresh_categorical_conditions(
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    if (assignments.empty()) {
        fail("categorical symbolic distribution needs at least one assignment");
    }
    if (nbits <= 0 || assignments.size() != probabilities.size()) {
        fail("invalid categorical symbolic distribution");
    }
    const std::size_t nwords = bit_word_count(nbits);
    double total = 0.0;
    for (const auto& assignment : assignments) {
        if (assignment.size() != nwords) {
            fail("categorical assignment length mismatch");
        }
    }
    for (double probability : probabilities) {
        total += check_probability(probability);
    }
    if (std::abs(total - 1.0) > 1e-12) {
        fail("categorical symbolic distribution probabilities must sum to 1");
    }
    std::vector<int> conditions;
    conditions.reserve(static_cast<std::size_t>(nbits));
    for (int i = 0; i < nbits; ++i) {
        conditions.push_back(fresh_condition());
    }
    const std::size_t group = categorical_distributions.size();
    categorical_distributions.push_back({nbits, conditions, assignments, probabilities});
    for (int condition : conditions) {
        condition_to_categorical[condition] = group;
    }
    return conditions;
}

std::vector<SymbolicBool> SymbolicContext::fresh_categorical_bools(
    int nbits,
    const std::vector<std::vector<std::uint64_t>>& assignments,
    const std::vector<double>& probabilities) {
    const auto conditions = fresh_categorical_conditions(nbits, assignments, probabilities);
    std::vector<SymbolicBool> out;
    out.reserve(conditions.size());
    for (int condition : conditions) {
        out.push_back(symbolic_bool(condition));
    }
    return out;
}

} // namespace symft
