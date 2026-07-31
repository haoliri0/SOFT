#pragma once

#include "core/pauli.hpp"
#include "core/symbolic.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace symft::detail {

inline constexpr int kWordBits = 64;
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kLowProbabilitySampleThreshold = 0.02;

[[noreturn]] inline void fail(const std::string& message) {
    throw Error(message);
}

inline int check_qubit(int nqubits, int q) {
    if (q < 0 || q >= nqubits) {
        fail("qubit index out of range");
    }
    return q;
}

inline void check_same_nqubits(const PauliString& a, const PauliString& b) {
    if (a.nqubits != b.nqubits) {
        fail("Pauli strings act on different numbers of qubits");
    }
}

inline std::uint64_t bit_mask(int q) {
    return std::uint64_t{1} << (q & 63);
}

inline int word_index(int q) {
    return q >> 6;
}

inline int popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(value);
#else
    int count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

inline std::uint64_t symbol_bit_mask(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return std::uint64_t{1} << ((condition - 1) & 63);
}

inline std::size_t symbol_word_index(int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    return static_cast<std::size_t>((condition - 1) >> 6);
}

inline std::size_t symbol_word_count(int nsymbols) {
    if (nsymbols <= 0) {
        return 0;
    }
    return static_cast<std::size_t>((nsymbols + 63) >> 6);
}

inline int trailing_zeros64(std::uint64_t value) {
    if (value == 0) {
        fail("trailing_zeros64 called with zero");
    }
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(value);
#else
    int count = 0;
    while ((value & 1u) == 0u) {
        value >>= 1;
        ++count;
    }
    return count;
#endif
}

inline int highest_set_bit64(std::uint64_t value) {
    if (value == 0) {
        fail("highest_set_bit64 called with zero");
    }
#if defined(__GNUC__) || defined(__clang__)
    return 63 - __builtin_clzll(value);
#else
    int bit = 0;
    while ((value >>= 1) != 0) {
        ++bit;
    }
    return bit;
#endif
}

inline bool is_odd_popcount(std::uint64_t value) {
    return (popcount64(value) & 1) != 0;
}

inline double check_probability(double probability) {
    if (!(probability >= 0.0 && probability <= 1.0)) {
        fail("probability must be between 0 and 1");
    }
    return probability;
}

inline void toggle_condition(std::vector<int>& conditions, int condition) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    auto it = std::lower_bound(conditions.begin(), conditions.end(), condition);
    if (it != conditions.end() && *it == condition) {
        conditions.erase(it);
    } else {
        conditions.insert(it, condition);
    }
}

inline std::vector<int> normalize_conditions(const std::vector<int>& conditions) {
    std::vector<int> out;
    for (int condition : conditions) {
        toggle_condition(out, condition);
    }
    return out;
}

inline bool optional_equal(const std::optional<int>& lhs, const std::optional<int>& rhs) {
    return lhs.has_value() == rhs.has_value() && (!lhs || *lhs == *rhs);
}

} // namespace symft::detail
