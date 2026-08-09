#include "core/symbolic.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <ostream>
#include <sstream>
#include <utility>

namespace symft {
using namespace detail;

SymbolicBool::SymbolicBool(bool constant_) : constant(constant_) {}

SymbolicBool::SymbolicBool(bool constant_, std::vector<int> conditions_)
    : constant(constant_), conditions(normalize_conditions(std::move(conditions_))) {}

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
    SymbolicBool out;
    out.constant = lhs.constant != rhs.constant;
    out.conditions.reserve(lhs.conditions.size() + rhs.conditions.size());
    std::set_symmetric_difference(
        lhs.conditions.begin(),
        lhs.conditions.end(),
        rhs.conditions.begin(),
        rhs.conditions.end(),
        std::back_inserter(out.conditions));
    return out;
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

namespace symft::detail {

void SymbolicRelationReducer::add(SymbolicBool relation) {
    if (relation.conditions.empty()) {
        return;
    }
    if (relation.conditions.size() == 1) {
        const std::size_t condition =
            static_cast<std::size_t>(relation.conditions.front());
        if (fixed_conditions_.size() <= condition) {
            fixed_conditions_.resize(condition + 1, -1);
        }
        fixed_conditions_[condition] = relation.constant ? 1 : 0;
        word_count_ = std::max(
            word_count_,
            symbol_word_index(relation.conditions.front()) + 1);
        return;
    }

    const std::size_t relation_id = relations_.size();
    PackedRelation packed;
    packed.expression = std::move(relation);
    for (int condition : packed.expression.conditions) {
        const std::size_t condition_id = static_cast<std::size_t>(condition);
        if (relation_index_.size() <= condition_id) {
            relation_index_.resize(condition_id + 1);
        }
        relation_index_[condition_id].push_back(relation_id);

        const std::size_t word = symbol_word_index(condition);
        const std::uint64_t mask = symbol_bit_mask(condition);
        word_count_ = std::max(word_count_, word + 1);
        if (packed.word_indices.empty() || packed.word_indices.back() != word) {
            packed.word_indices.push_back(word);
            packed.word_masks.push_back(mask);
        } else {
            packed.word_masks.back() |= mask;
        }
    }
    relations_.push_back(std::move(packed));
}

bool SymbolicRelationReducer::empty() const {
    return relations_.empty() && fixed_conditions_.empty();
}

SymbolicBool SymbolicRelationReducer::reduce(SymbolicBool expression) const {
    if (expression.conditions.empty() || empty()) {
        return expression;
    }

    std::size_t required_words = word_count_;
    required_words = std::max(
        required_words,
        symbol_word_index(expression.conditions.back()) + 1);
    words_.assign(required_words, 0);
    std::size_t current_words = 0;
    for (int condition : expression.conditions) {
        const std::size_t word = symbol_word_index(condition);
        current_words += words_[word] == 0;
        words_[word] |= symbol_bit_mask(condition);
    }
    std::size_t current_size = expression.conditions.size();

    while (true) {
        candidates_.clear();
        if (candidate_marks_.size() < relations_.size()) {
            candidate_marks_.resize(relations_.size(), 0);
        }

        for (std::size_t word = 0; word < words_.size(); ++word) {
            std::uint64_t bits = words_[word];
            const bool occupied = bits != 0;
            while (bits) {
                const int bit = trailing_zeros64(bits);
                const int condition = static_cast<int>(
                    word * kWordBits + static_cast<std::size_t>(bit) + 1);
                const std::int8_t fixed =
                    static_cast<std::size_t>(condition) < fixed_conditions_.size()
                        ? fixed_conditions_[static_cast<std::size_t>(condition)]
                        : -1;
                if (fixed >= 0) {
                    expression.constant ^= fixed != 0;
                    words_[word] &= ~(std::uint64_t{1} << bit);
                    --current_size;
                } else if (static_cast<std::size_t>(condition) < relation_index_.size()) {
                    for (std::size_t relation :
                         relation_index_[static_cast<std::size_t>(condition)]) {
                        if (candidate_marks_[relation] == 0) {
                            candidate_marks_[relation] = 1;
                            candidates_.push_back(relation);
                        }
                    }
                }
                bits &= bits - 1;
            }
            if (occupied && words_[word] == 0) {
                --current_words;
            }
        }

        std::sort(candidates_.begin(), candidates_.end());
        for (std::size_t relation : candidates_) {
            candidate_marks_[relation] = 0;
        }

        bool changed = false;
        for (std::size_t relation_id : candidates_) {
            const auto& relation = relations_[relation_id];
            std::size_t overlap = 0;
            std::size_t candidate_words = current_words;
            for (std::size_t index = 0; index < relation.word_indices.size(); ++index) {
                const std::size_t word = relation.word_indices[index];
                const std::uint64_t old_value = words_[word];
                const std::uint64_t mask = relation.word_masks[index];
                const std::uint64_t new_value = old_value ^ mask;
                overlap += static_cast<std::size_t>(popcount64(old_value & mask));
                candidate_words -= old_value != 0;
                candidate_words += new_value != 0;
            }
            const std::size_t candidate_size =
                current_size + relation.expression.conditions.size() - 2 * overlap;
            if (candidate_words > current_words ||
                (candidate_words == current_words && candidate_size >= current_size)) {
                continue;
            }
            for (std::size_t index = 0; index < relation.word_indices.size(); ++index) {
                words_[relation.word_indices[index]] ^= relation.word_masks[index];
            }
            expression.constant ^= relation.expression.constant;
            current_size = candidate_size;
            current_words = candidate_words;
            changed = true;
        }
        if (!changed) {
            break;
        }
    }

    expression.conditions.clear();
    expression.conditions.reserve(current_size);
    for (std::size_t word = 0; word < words_.size(); ++word) {
        std::uint64_t bits = words_[word];
        while (bits) {
            const int bit = trailing_zeros64(bits);
            expression.conditions.push_back(static_cast<int>(
                word * kWordBits + static_cast<std::size_t>(bit) + 1));
            bits &= bits - 1;
        }
    }
    return expression;
}

} // namespace symft::detail
