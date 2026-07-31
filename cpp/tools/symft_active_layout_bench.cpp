#include "simd/simd.hpp"
#include "sampler/active.hpp"
#include "sampler/active_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Complex = symft::Complex;

constexpr double kPi = 3.141592653589793238462643383279502884;

#if defined(__clang__)
#define SYMFT_BENCH_SIMD_LOOP _Pragma("clang loop vectorize(enable) interleave(enable)")
#elif defined(__GNUC__)
#define SYMFT_BENCH_SIMD_LOOP _Pragma("GCC ivdep")
#else
#define SYMFT_BENCH_SIMD_LOOP
#endif

#if defined(__AVX2__) && defined(__FMA__)
#define SYMFT_BENCH_INLINE_AVX2 1
#else
#define SYMFT_BENCH_INLINE_AVX2 0
#endif

struct SoAState {
    std::vector<double> re;
    std::vector<double> im;
};

struct BenchCase {
    std::string name;
    symft::PrecomputedActivePauliRotationKernel kernel;
};

struct BenchResult {
    double seconds = 0.0;
    double ns_per_apply = 0.0;
    double ns_per_amplitude = 0.0;
    double checksum = 0.0;
};

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::size_t active_dim(int k) {
    return std::size_t{1} << k;
}

symft::PauliString uniform_pauli(int k, int pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    if (noncontiguous) {
        pauli.set_xbit(pair_bit - 1);
    }
    return pauli;
}

symft::PauliString real_pauli(int k, int pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    pauli.set_zbit(pair_bit);
    pauli.set_phase(1);
    if (noncontiguous) {
        pauli.set_xbit(pair_bit - 1);
    }
    return pauli;
}

symft::PauliString general_pauli(int k, int pair_bit, bool noncontiguous) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    if (noncontiguous) {
        pauli.set_xbit(pair_bit - 1);
    }
    pauli.set_zbit((pair_bit + 1) % k);
    return pauli;
}

symft::PauliString uniform_pauli_with_lower_mask(int k, int pair_bit, std::uint64_t lower_mask) {
    symft::PauliString pauli(k);
    pauli.set_xbit(pair_bit);
    for (int bit = 0; bit < pair_bit; ++bit) {
        if ((lower_mask & (std::uint64_t{1} << bit)) != 0) {
            pauli.set_xbit(bit);
        }
    }
    return pauli;
}

symft::PauliString real_pauli_with_lower_mask(int k, int pair_bit, std::uint64_t lower_mask) {
    symft::PauliString pauli = uniform_pauli_with_lower_mask(k, pair_bit, lower_mask);
    pauli.set_zbit(pair_bit);
    pauli.set_phase(1);
    return pauli;
}

symft::PauliString general_pauli_with_lower_mask(int k, int pair_bit, std::uint64_t lower_mask) {
    symft::PauliString pauli = uniform_pauli_with_lower_mask(k, pair_bit, lower_mask);
    int zbit = 0;
    while (zbit == pair_bit || (lower_mask & (std::uint64_t{1} << zbit)) != 0) {
        ++zbit;
    }
    pauli.set_zbit(zbit);
    return pauli;
}

std::vector<Complex> initial_alpha(int k) {
    const std::size_t dim = active_dim(k);
    std::vector<Complex> alpha(dim);
    double norm2 = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double r = std::sin(0.17 * static_cast<double>(i + 1)) + 0.25 * std::cos(0.03 * static_cast<double>(i + 3));
        const double im = std::cos(0.11 * static_cast<double>(i + 5)) - 0.15 * std::sin(0.07 * static_cast<double>(i + 2));
        alpha[i] = Complex(r, im);
        norm2 += r * r + im * im;
    }
    const double invnorm = 1.0 / std::sqrt(norm2);
    for (auto& value : alpha) {
        value *= invnorm;
    }
    return alpha;
}

SoAState to_soa(const std::vector<Complex>& alpha) {
    SoAState out;
    out.re.resize(alpha.size());
    out.im.resize(alpha.size());
    for (std::size_t i = 0; i < alpha.size(); ++i) {
        out.re[i] = alpha[i].real();
        out.im[i] = alpha[i].imag();
    }
    return out;
}

double checksum_aos(const std::vector<Complex>& alpha) {
    double out = 0.0;
    for (std::size_t i = 0; i < alpha.size(); ++i) {
        out += (1.0 + static_cast<double>(i & 15)) * (alpha[i].real() + 0.5 * alpha[i].imag());
    }
    return out;
}

double checksum_soa(const SoAState& state) {
    double out = 0.0;
    for (std::size_t i = 0; i < state.re.size(); ++i) {
        out += (1.0 + static_cast<double>(i & 15)) * (state.re[i] + 0.5 * state.im[i]);
    }
    return out;
}


void apply_soa_block(SoAState& state, const symft::PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    const double c = kernel.cos_kernel_angle;
    if (kernel.is_diagonal) {
        SYMFT_BENCH_SIMD_LOOP
        for (std::size_t basis = 0; basis < state.re.size(); ++basis) {
            const Complex coefficient = symft::detail::compact_rotation_coefficient(kernel, basis, sign);
            const double fr = c + coefficient.real();
            const double fi = coefficient.imag();
            const double r = state.re[basis];
            const double i = state.im[basis];
            state.re[basis] = fr * r - fi * i;
            state.im[basis] = fr * i + fi * r;
        }
        return;
    }
    SYMFT_BENCH_SIMD_LOOP
    for (std::size_t pair = 0; pair < kernel.pair_count; ++pair) {
        const std::size_t left = symft::detail::insert_zero_bit(pair, static_cast<int>(kernel.pair_bit));
        const std::size_t right = left ^ static_cast<std::size_t>(kernel.action.xmask);
        const Complex left_coefficient = symft::detail::compact_rotation_coefficient(kernel, left, sign);
        const Complex right_coefficient = symft::detail::compact_rotation_coefficient(kernel, right, sign);
        const Complex a0(state.re[left], state.im[left]);
        const Complex a1(state.re[right], state.im[right]);
        const Complex out0 = c * a0 + right_coefficient * a1;
        const Complex out1 = c * a1 + left_coefficient * a0;
        state.re[left] = out0.real();
        state.im[left] = out0.imag();
        state.re[right] = out1.real();
        state.im[right] = out1.imag();
    }
}

void apply_soa_dispatch(SoAState& state, const symft::PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (!kernel.uniform_imag_pairs) {
        apply_soa_block(state, kernel, sign);
        return;
    }
    const Complex coefficient = symft::detail::compact_rotation_coefficient(kernel, 0, sign);
    symft::simd::dispatch_table().rotate_uniform_imag_pairs_soa(
        state.re.data(),
        state.im.data(),
        state.re.size(),
        kernel.action.xmask,
        kernel.pair_bit,
        kernel.cos_kernel_angle,
        coefficient.imag());
}

BenchResult bench_aos(
    const std::vector<Complex>& initial,
    const symft::PrecomputedActivePauliRotationKernel& kernel,
    int k,
    int iterations) {
    symft::ActiveState state(k, initial);
    const auto start = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        symft::rotate_pauli(state, kernel, (iter & 1) != 0);
    }
    const auto stop = Clock::now();
    const double seconds = seconds_between(start, stop);
    return {
        seconds,
        1.0e9 * seconds / static_cast<double>(iterations),
        1.0e9 * seconds / static_cast<double>(iterations * static_cast<int>(initial.size())),
        checksum_aos(state.alpha),
    };
}

template <typename Apply>
BenchResult bench_soa(const std::vector<Complex>& initial, const symft::PrecomputedActivePauliRotationKernel& kernel, int iterations, Apply apply) {
    SoAState state = to_soa(initial);
    const auto start = Clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        apply(state, kernel, (iter & 1) != 0);
    }
    const auto stop = Clock::now();
    const double seconds = seconds_between(start, stop);
    return {
        seconds,
        1.0e9 * seconds / static_cast<double>(iterations),
        1.0e9 * seconds / static_cast<double>(iterations * static_cast<int>(initial.size())),
        checksum_soa(state),
    };
}

std::vector<BenchCase> make_cases(int k) {
    std::vector<BenchCase> cases;
    cases.push_back({"diag_z0", symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(symft::pauli_z(k, 0)), kPi / 8.0)});
    cases.push_back({"uniform_x0", symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(symft::pauli_x(k, 0)), kPi / 8.0)});

    std::vector<int> pair_bits = {1, std::min(3, k - 1), k - 1};
    std::sort(pair_bits.begin(), pair_bits.end());
    pair_bits.erase(std::unique(pair_bits.begin(), pair_bits.end()), pair_bits.end());
    for (const int pair_bit : pair_bits) {
        cases.push_back(
            {"uniform_p" + std::to_string(pair_bit) + "_contig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(uniform_pauli(k, pair_bit, false)), kPi / 8.0)});
        cases.push_back(
            {"uniform_p" + std::to_string(pair_bit) + "_noncontig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(uniform_pauli(k, pair_bit, true)), kPi / 8.0)});
        cases.push_back(
            {"real_p" + std::to_string(pair_bit) + "_contig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(real_pauli(k, pair_bit, false)), kPi / 8.0)});
        cases.push_back(
            {"real_p" + std::to_string(pair_bit) + "_noncontig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(real_pauli(k, pair_bit, true)), kPi / 8.0)});
        cases.push_back(
            {"general_p" + std::to_string(pair_bit) + "_contig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(general_pauli(k, pair_bit, false)), kPi / 8.0)});
        cases.push_back(
            {"general_p" + std::to_string(pair_bit) + "_noncontig",
             symft::PrecomputedActivePauliRotationKernel(symft::ActivePauliAction(general_pauli(k, pair_bit, true)), kPi / 8.0)});
    }
    if (k >= 8) {
        std::vector<int> stress_pair_bits = {std::min(5, k - 1), k - 1};
        std::sort(stress_pair_bits.begin(), stress_pair_bits.end());
        stress_pair_bits.erase(std::unique(stress_pair_bits.begin(), stress_pair_bits.end()), stress_pair_bits.end());
        for (const int pair_bit : stress_pair_bits) {
            for (std::uint64_t lower_mask = 1; lower_mask <= 3; ++lower_mask) {
                cases.push_back(
                    {"uniform_p" + std::to_string(pair_bit) + "_low" + std::to_string(lower_mask),
                     symft::PrecomputedActivePauliRotationKernel(
                         symft::ActivePauliAction(uniform_pauli_with_lower_mask(k, pair_bit, lower_mask)),
                         kPi / 8.0)});
                cases.push_back(
                    {"real_p" + std::to_string(pair_bit) + "_low" + std::to_string(lower_mask),
                     symft::PrecomputedActivePauliRotationKernel(
                         symft::ActivePauliAction(real_pauli_with_lower_mask(k, pair_bit, lower_mask)),
                         kPi / 8.0)});
                cases.push_back(
                    {"general_p" + std::to_string(pair_bit) + "_low" + std::to_string(lower_mask),
                     symft::PrecomputedActivePauliRotationKernel(
                         symft::ActivePauliAction(general_pauli_with_lower_mask(k, pair_bit, lower_mask)),
                         kPi / 8.0)});
            }
        }
    }
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
    const BenchCase& bench_case,
    const std::string& layout,
    int iterations,
    const BenchResult& result) {
    std::cout << "k " << k << " dim " << active_dim(k) << " kernel " << bench_case.name << " layout " << layout
              << " pair_bit " << bench_case.kernel.pair_bit
              << " xmask 0x" << std::hex << bench_case.kernel.action.xmask << std::dec
              << " pairing " << pairing_name(bench_case.kernel)
              << " iterations " << iterations << " seconds " << result.seconds << " ns_per_apply " << result.ns_per_apply
              << " ns_per_amplitude " << result.ns_per_amplitude << " checksum " << std::setprecision(17) << result.checksum
              << std::setprecision(6) << "\n";
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
            std::cerr << "usage: symft_active_layout_bench [target_amplitude_updates] [repeats] [max_k]\n";
            return 2;
        }
        const int target_updates = argc >= 2 ? parse_int(argv[1], "target_amplitude_updates") : (1 << 22);
        const int repeats = argc >= 3 ? parse_int(argv[2], "repeats") : 3;
        const int max_k = argc >= 4 ? parse_int(argv[3], "max_k") : 10;
        if (target_updates <= 0 || repeats <= 0 || max_k < 4) {
            throw std::runtime_error("target updates and repeats must be positive; max_k must be at least 4");
        }

        std::cout << "SymFT C++ active-state layout microbenchmark\n";
        std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
        std::cout << "target_amplitude_updates " << target_updates << "\n";
        std::cout << "repeats " << repeats << "\n";

        for (int k = 4; k <= max_k; k += 2) {
            const auto initial = initial_alpha(k);
            const int iterations = std::max(16, target_updates / static_cast<int>(active_dim(k)));
            const auto cases = make_cases(k);
            for (const auto& bench_case : cases) {
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    print_result(k, bench_case, "aos_complex", iterations, bench_aos(initial, bench_case.kernel, k, iterations));
                    print_result(
                        k,
                        bench_case,
                        "soa_block",
                        iterations,
                        bench_soa(initial, bench_case.kernel, iterations, apply_soa_block));
                    print_result(
                        k,
                        bench_case,
                        "soa_dispatch",
                        iterations,
                        bench_soa(initial, bench_case.kernel, iterations, apply_soa_dispatch));
                    std::cout << "repeat_done " << repeat + 1 << "\n";
                }
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_active_layout_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
