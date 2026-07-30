#include "frontend/stim.hpp"
#include "sampler/exogenous.hpp"
#include "sampler/single_shot.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

int popcount64(std::uint64_t value) {
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

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: symft_bench <circuit.stim> <shots> [repeats]\n";
        return 2;
    }

    try {
        const std::string path = argv[1];
        const int shots = std::stoi(argv[2]);
        const int repeats = argc == 4 ? std::stoi(argv[3]) : 3;
        if (shots < 0 || repeats <= 0) {
            throw std::runtime_error("shots must be nonnegative and repeats must be positive");
        }

        double parse_total = 0.0;
        double plan_total = 0.0;
        double presample_total = 0.0;
        double execute_total = 0.0;
        double sample_total = 0.0;
        int n = 0;
        int records = 0;
        int max_k = 0;
        std::uint64_t checksum = 0;

        for (int repeat = 0; repeat < repeats; ++repeat) {
            const auto parse_start = Clock::now();
            auto parsed = symft::parse_stim_file(path);
            const auto parse_stop = Clock::now();

            const auto plan_start = Clock::now();
            symft::PendingFactoredState pending(parsed.state);
            auto program = symft::plan_factored_updates(std::move(pending));
            const auto plan_stop = Clock::now();

            n = parsed.state.n;
            records = program.nrecords;
            max_k = program.max_k;

            const std::uint64_t seed = 0x5eedULL + static_cast<std::uint64_t>(repeat);
            symft::FactoredExecutorState runtime(program, seed);
            const auto sample_start = Clock::now();
            const auto samples = symft::presample_exogenous(program, shots, seed);
            const auto presample_stop = Clock::now();
            runtime.rng_state = samples.next_rng_state;
            for (int shot = 0; shot < shots; ++shot) {
                symft::reset_executor(runtime, program);
                symft::execute_in_place(runtime, program, samples, shot);
                for (std::uint64_t word : runtime.measurement_words) {
                    checksum += static_cast<std::uint64_t>(popcount64(word));
                }
            }
            const auto sample_stop = Clock::now();

            parse_total += seconds_between(parse_start, parse_stop);
            plan_total += seconds_between(plan_start, plan_stop);
            presample_total += seconds_between(sample_start, presample_stop);
            execute_total += seconds_between(presample_stop, sample_stop);
            sample_total += seconds_between(sample_start, sample_stop);
        }

        const double inv_repeats = 1.0 / static_cast<double>(repeats);
        const double parse_s = parse_total * inv_repeats;
        const double plan_s = plan_total * inv_repeats;
        const double presample_s = presample_total * inv_repeats;
        const double execute_s = execute_total * inv_repeats;
        const double sample_s = sample_total * inv_repeats;
        const double shots_per_s = sample_s > 0.0 ? static_cast<double>(shots) / sample_s : 0.0;

        std::cout << "file " << path << "\n";
        std::cout << "qubits " << n << "\n";
        std::cout << "records " << records << "\n";
        std::cout << "max_active_qubits " << max_k << "\n";
        std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
        std::cout << "shots " << shots << "\n";
        std::cout << "repeats " << repeats << "\n";
        std::cout << "exogenous_mode presampled\n";
        std::cout << "parse_s_avg " << parse_s << "\n";
        std::cout << "plan_s_avg " << plan_s << "\n";
        std::cout << "presample_s_avg " << presample_s << "\n";
        std::cout << "execute_s_avg " << execute_s << "\n";
        std::cout << "sample_s_avg " << sample_s << "\n";
        std::cout << "sample_shots_per_s " << shots_per_s << "\n";
        std::cout << "sample_ms_per_shot " << (shots > 0 ? 1000.0 * sample_s / static_cast<double>(shots) : 0.0) << "\n";
        std::cout << "checksum " << checksum << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "symft_bench: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
