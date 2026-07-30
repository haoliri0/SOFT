#pragma once

#include "core/internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace symft {

inline std::uint64_t next_random_u64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline double rand_float(std::uint64_t& state) {
    return static_cast<double>(next_random_u64(state) >> 11) * 0x1.0p-53;
}

inline bool sample_bernoulli(std::uint64_t& rng_state, double probability) {
    const double p = detail::check_probability(probability);
    if (p <= 0.0) {
        return false;
    }
    if (p >= 1.0) {
        return true;
    }
    return rand_float(rng_state) < p;
}

inline int sample_categorical_row(std::uint64_t& rng_state, const std::vector<double>& probabilities) {
    const double r = rand_float(rng_state);
    double cumulative = 0.0;
    for (std::size_t i = 0; i < probabilities.size(); ++i) {
        cumulative += probabilities[i];
        if (r <= cumulative) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(probabilities.size() - 1);
}

inline double sample_geometric_gap(std::uint64_t& rng_state, double probability) {
    if (!(probability > 0.0 && probability < 1.0)) {
        detail::fail("geometric gap probability must be in (0, 1)");
    }
    const double u = std::max(rand_float(rng_state), std::numeric_limits<double>::min());
    const double gap = std::floor(std::log(u) / std::log1p(-probability));
    if (!std::isfinite(gap) || gap >= static_cast<double>(std::numeric_limits<int>::max())) {
        return static_cast<double>(std::numeric_limits<int>::max());
    }
    return gap;
}

} // namespace symft
