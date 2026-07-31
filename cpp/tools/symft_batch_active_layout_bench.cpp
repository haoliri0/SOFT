#include "sampler/active.hpp"
#include "sampler/active_internal.hpp"
#include "sampler/batch_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Complex = symft::Complex;

constexpr double kPi = 3.141592653589793238462643383279502884;

struct BatchState {
    std::vector<double> re;
    std::vector<double> im;
};

struct BenchCase {
    std::string kernel_name;
    std::string xmask_name;
    symft::PrecomputedActivePauliRotationKernel kernel;
};

struct BenchResult {
    double seconds = 0.0;
    double ns_per_apply = 0.0;
    double ns_per_amplitude_shot = 0.0;
    double checksum = 0.0;
};

enum class SignPattern {
    AllMinus,
    AllPlus,
    Mixed,
};

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::size_t active_dim(int k) {
    return std::size_t{1} << k;
}

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return ~std::uint64_t{0};
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int shots, std::size_t word) {
    return low_bits_mask(shots - static_cast<int>(word << 6));
}

std::string sign_pattern_name(SignPattern pattern) {
    switch (pattern) {
    case SignPattern::AllMinus:
        return "all_minus";
    case SignPattern::AllPlus:
        return "all_plus";
    case SignPattern::Mixed:
        return "mixed";
    }
    return "unknown";
}

symft::PauliString uniform_pauli(int k, unsigned pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(static_cast<int>(pair_bit));
    if (noncontiguous) {
        pauli.set_xbit(static_cast<int>(pair_bit) - 1);
    }
    return pauli;
}

symft::PauliString real_pauli(int k, unsigned pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(static_cast<int>(pair_bit));
    pauli.set_zbit(static_cast<int>(pair_bit));
    pauli.set_phase(1);
    if (noncontiguous) {
        pauli.set_xbit(static_cast<int>(pair_bit) - 1);
    }
    return pauli;
}

symft::PauliString general_pauli(int k, unsigned pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(static_cast<int>(pair_bit));
    if (noncontiguous) {
        pauli.set_xbit(static_cast<int>(pair_bit) - 1);
    }
    pauli.set_zbit((static_cast<int>(pair_bit) + 2) % k);
    return pauli;
}

BatchState initial_state(int k, int shots) {
    const std::size_t dim = active_dim(k);
    BatchState state;
    state.re.resize(dim * static_cast<std::size_t>(shots));
    state.im.resize(dim * static_cast<std::size_t>(shots));
    for (std::size_t basis = 0; basis < dim; ++basis) {
        const std::size_t base = basis * static_cast<std::size_t>(shots);
        for (int shot = 0; shot < shots; ++shot) {
            const double x = static_cast<double>((basis + 1) * 17 + static_cast<std::size_t>(shot + 3) * 5);
            state.re[base + static_cast<std::size_t>(shot)] = 0.01 * std::sin(0.013 * x);
            state.im[base + static_cast<std::size_t>(shot)] = 0.01 * std::cos(0.017 * x);
        }
    }
    return state;
}

std::vector<std::uint64_t> sign_bits(int shots, SignPattern pattern) {
    const std::size_t nwords = static_cast<std::size_t>((shots + 63) >> 6);
    std::vector<std::uint64_t> out(nwords, 0);
    for (std::size_t word = 0; word < nwords; ++word) {
        const std::uint64_t live = live_word_mask(shots, word);
        if (pattern == SignPattern::AllPlus) {
            out[word] = live;
        } else if (pattern == SignPattern::Mixed) {
            out[word] = (word & 1) == 0 ? (0xaaaaaaaaaaaaaaaaULL & live) : (0x9249249249249249ULL & live);
        }
    }
    return out;
}

std::vector<double> q_by_shot(int shots, const std::vector<std::uint64_t>& bits, double minus_coeff, double plus_coeff) {
    std::vector<double> out(static_cast<std::size_t>(shots), 0.0);
    for (int shot = 0; shot < shots; ++shot) {
        const bool sign = (bits[static_cast<std::size_t>(shot >> 6)] & (std::uint64_t{1} << (shot & 63))) != 0;
        out[static_cast<std::size_t>(shot)] = sign ? plus_coeff : minus_coeff;
    }
    return out;
}

double checksum(const BatchState& state) {
    double out = 0.0;
    for (std::size_t i = 0; i < state.re.size(); ++i) {
        out += (1.0 + static_cast<double>(i & 7)) * (state.re[i] + 0.25 * state.im[i]);
    }
    return out;
}


void apply_case(BatchState& state, const BenchCase& bench_case, int shots, SignPattern pattern) {
    const auto& kernel = bench_case.kernel;
    const auto bits = sign_bits(shots, pattern);
    const std::size_t pitch = static_cast<std::size_t>(shots);
    const std::size_t dim = state.re.size() / pitch;
    const auto sign_at = [&](int shot) {
        return (bits[static_cast<std::size_t>(shot >> 6)] & (std::uint64_t{1} << (shot & 63))) != 0;
    };
    const double c = kernel.cos_kernel_angle;
    if (kernel.is_diagonal) {
        for (std::size_t basis = 0; basis < dim; ++basis) {
            const Complex minus_coefficient = symft::detail::compact_rotation_coefficient(kernel, basis, false);
            const std::size_t base = basis * pitch;
            for (int shot = 0; shot < shots; ++shot) {
                const std::size_t lane = static_cast<std::size_t>(shot);
                const Complex coefficient = sign_at(shot) ? -minus_coefficient : minus_coefficient;
                const double fr = c + coefficient.real();
                const double fi = coefficient.imag();
                const double r = state.re[base + lane];
                const double i = state.im[base + lane];
                state.re[base + lane] = fr * r - fi * i;
                state.im[base + lane] = fr * i + fi * r;
            }
        }
        return;
    }
    for (std::size_t pair = 0; pair < kernel.pair_count; ++pair) {
        const std::size_t left = symft::detail::insert_zero_bit(pair, static_cast<int>(kernel.pair_bit));
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.action.xmask);
        const Complex left_minus = symft::detail::compact_rotation_coefficient(kernel, left, false);
        const Complex right_minus = symft::detail::compact_rotation_coefficient(kernel, right, false);
        const std::size_t left_base = left * pitch;
        const std::size_t right_base = right * pitch;
        for (int shot = 0; shot < shots; ++shot) {
            const std::size_t lane = static_cast<std::size_t>(shot);
            const double direction = sign_at(shot) ? -1.0 : 1.0;
            const Complex a0(state.re[left_base + lane], state.im[left_base + lane]);
            const Complex a1(state.re[right_base + lane], state.im[right_base + lane]);
            const Complex out0 = c * a0 + direction * right_minus * a1;
            const Complex out1 = c * a1 + direction * left_minus * a0;
            state.re[left_base + lane] = out0.real();
            state.im[left_base + lane] = out0.imag();
            state.re[right_base + lane] = out1.real();
            state.im[right_base + lane] = out1.imag();
        }
    }
}

BenchResult run_bench(const BatchState& initial, const BenchCase& bench_case, int shots, int iterations, SignPattern pattern) {
    BatchState state = initial;
    const auto start = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        apply_case(state, bench_case, shots, pattern);
    }
    const auto stop = Clock::now();
    const double seconds = seconds_between(start, stop);
    return {
        seconds,
        1.0e9 * seconds / static_cast<double>(iterations),
        1.0e9 * seconds / static_cast<double>(iterations * static_cast<int>(state.re.size())),
        checksum(state),
    };
}

std::vector<BenchCase> make_cases(int k, unsigned pair_bit, bool noncontiguous) {
    const std::string xmask_name = noncontiguous ? "noncontiguous" : "contiguous";
    std::vector<BenchCase> cases;
    if (pair_bit == 1 && !noncontiguous) {
        cases.push_back({"diagonal", "z0", symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(symft::pauli_z(k, 0)), kPi / 8.0)});
    }
    cases.push_back({"uniform", xmask_name, symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(uniform_pauli(k, pair_bit, noncontiguous)), kPi / 8.0)});
    cases.push_back({"real", xmask_name, symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(real_pauli(k, pair_bit, noncontiguous)), kPi / 8.0)});
    cases.push_back({"general", xmask_name, symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(general_pauli(k, pair_bit, noncontiguous)), kPi / 8.0)});
    return cases;
}

const char* pairing_name(const symft::PrecomputedActivePauliRotationKernel& kernel) {
    if (kernel.is_diagonal) {
        return "diagonal";
    }
    const std::size_t selector = std::size_t{1} << kernel.pair_bit;
    return (static_cast<std::size_t>(kernel.action.xmask) & (selector - 1)) == 0 ? "contiguous" : "noncontiguous";
}

void print_result(
    int k,
    int shots,
    const BenchCase& bench_case,
    SignPattern pattern,
    int iterations,
    const BenchResult& result) {
    std::cout << "k " << k << " dim " << active_dim(k) << " active_shots " << shots
              << " kernel " << bench_case.kernel_name << " xmask " << bench_case.xmask_name
              << " pair_bit " << bench_case.kernel.pair_bit
              << " raw_xmask 0x" << std::hex << bench_case.kernel.action.xmask << std::dec
              << " pairing " << pairing_name(bench_case.kernel)
              << " sign_pattern " << sign_pattern_name(pattern)
              << " iterations " << iterations
              << " seconds " << result.seconds
              << " ns_per_apply " << result.ns_per_apply
              << " ns_per_amplitude_shot " << result.ns_per_amplitude_shot
              << " checksum " << std::setprecision(17) << result.checksum << std::setprecision(6) << "\n";
}

int parse_int(const char* raw, const char* name) {
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(raw, &consumed);
        if (consumed != std::string(raw).size()) {
            throw std::runtime_error("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 4) {
            std::cerr << "usage: symft_batch_active_layout_bench [target_amplitude_shot_updates] [repeats] [max_k]\n";
            return 2;
        }
        const int target_updates = argc >= 2 ? parse_int(argv[1], "target_amplitude_shot_updates") : (1 << 22);
        const int repeats = argc >= 3 ? parse_int(argv[2], "repeats") : 1;
        const int max_k = argc >= 4 ? parse_int(argv[3], "max_k") : 10;
        if (target_updates <= 0 || repeats <= 0 || max_k < 4) {
            throw std::runtime_error("target updates and repeats must be positive; max_k must be at least 4");
        }

        std::cout << "SymFT C++ batch active layout microbenchmark\n";
        std::cout << "layout basis_major_interleaved\n";
        std::cout << "target_amplitude_shot_updates " << target_updates << "\n";
        std::cout << "repeats " << repeats << "\n";

        const std::vector<int> shot_counts = {64, 256, 1024};
        const std::vector<SignPattern> patterns = {SignPattern::AllMinus, SignPattern::AllPlus, SignPattern::Mixed};
        for (int k = 4; k <= max_k; k += 2) {
            std::vector<unsigned> pair_bits = {1, static_cast<unsigned>(std::min(3, k - 1)), static_cast<unsigned>(k - 1)};
            std::sort(pair_bits.begin(), pair_bits.end());
            pair_bits.erase(std::unique(pair_bits.begin(), pair_bits.end()), pair_bits.end());
            for (int shots : shot_counts) {
                const BatchState initial = initial_state(k, shots);
                const int iterations = std::max(2, target_updates / static_cast<int>(active_dim(k) * static_cast<std::size_t>(shots)));
                for (unsigned pair_bit : pair_bits) {
                    for (bool noncontiguous : {false, true}) {
                        for (const auto& bench_case : make_cases(k, pair_bit, noncontiguous)) {
                            for (SignPattern pattern : patterns) {
                                for (int repeat = 0; repeat < repeats; ++repeat) {
                                    (void)repeat;
                                    print_result(
                                        k,
                                        shots,
                                        bench_case,
                                        pattern,
                                        iterations,
                                        run_bench(initial, bench_case, shots, iterations, pattern));
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_batch_active_layout_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
