// Thin persistent process wrapper around the existing compiled sampler (LTO is a build option).
// Protocol: stdin lines "index stream_id shots"; stdout JSONL ready/result.
#include "frontend/stim_prepared_sampler.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

double monotonic_s() {
    timespec t{};
    if (clock_gettime(CLOCK_MONOTONIC, &t)) throw std::runtime_error("clock_gettime failed");
    return double(t.tv_sec) + double(t.tv_nsec) * 1e-9;
}
double process_cpu_s() {
    rusage r{};
    if (getrusage(RUSAGE_SELF, &r)) throw std::runtime_error("getrusage failed");
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec +
        (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: cpu_stream_worker CIRCUIT CPU");
        char* end = nullptr;
        errno = 0;
        const auto cpu = std::strtol(argv[2], &end, 10);
        if (errno || end == argv[2] || *end || cpu < 0 || cpu >= CPU_SETSIZE)
            throw std::runtime_error("invalid CPU");
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu, &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity))
            throw std::runtime_error("sched_setaffinity failed");
        const double prepare_start = monotonic_s();
        symft::CircuitSamplingOptions options;
        options.threads = 1;
        options.postselect_detectors = true;
        options.cpu_compiled = true;
        options.cpu_real_gauge = true;
        options.cpu_hoist_detectors = true;
        options.sample_chunk_shots = 2048;
        auto sampler = symft::PreparedCircuitBatchSampler(
            symft::make_stim_circuit_sampling_input_from_file(argv[1], options), options);
        const auto& info = sampler.info();
        if (!info.cpu_compiled || !info.cpu_real_gauge || info.threads != 1 ||
            !info.detector_postselection || !info.cpu_fallback_reason.empty())
            throw std::runtime_error("required compiled real-gauge single-worker backend unavailable");
        std::cout << std::setprecision(17)
            << "{\"event\":\"ready\",\"pid\":" << getpid()
            << ",\"cpu\":" << cpu << ",\"cpu_backend\":\"compiled\",\"cpu_rng\":\"cpu-shot-v1\""
            << ",\"cpu_real_gauge\":true,\"threads\":1,\"sample_chunk_shots\":" << info.sample_chunk_shots
            << ",\"n\":" << info.n << ",\"records\":" << info.records
            << ",\"detectors\":" << info.detectors << ",\"max_k\":" << info.max_k
            << ",\"cpu_noise_only_detectors\":" << info.cpu_noise_only_detectors
            << ",\"cpu_initial_checks\":" << info.cpu_initial_checks
            << ",\"prepare_wall_s\":" << monotonic_s() - prepare_start << "}" << std::endl;
        std::uint64_t index, stream, shots;
        while (std::cin >> index >> stream >> shots) {
            if (!shots) throw std::runtime_error("shots must be positive");
            const double start = monotonic_s(), cpu_start = process_cpu_s();
            const auto run = sampler.sample(shots, stream);
            const double finish = monotonic_s(), cpu_finish = process_cpu_s();
            const auto& c = run.counts;
            if (c.shots != shots || c.shots != c.discarded + c.accepted ||
                c.logical_errors > c.accepted || run.active_threads != 1)
                throw std::runtime_error("invalid sampler counts or thread count");
            std::cout << "{\"event\":\"result\",\"index\":" << index
                << ",\"stream_id\":" << stream << ",\"shots\":" << c.shots
                << ",\"discarded\":" << c.discarded << ",\"accepted\":" << c.accepted
                << ",\"logical_errors\":" << c.logical_errors
                << ",\"start_monotonic_s\":" << start << ",\"finish_monotonic_s\":" << finish
                << ",\"call_wall_s\":" << finish - start
                << ",\"process_cpu_s\":" << cpu_finish - cpu_start
                << ",\"sample_s\":" << run.timing.sample_s
                << ",\"presample_s\":" << run.timing.presample_s
                << ",\"execute_s\":" << run.timing.execute_s
                << ",\"actual_cpu\":" << sched_getcpu() << "}" << std::endl;
        }
        if (!std::cin.eof()) throw std::runtime_error("invalid input protocol");
    } catch (const std::exception& e) {
        std::cerr << "cpu_stream_worker: " << e.what() << std::endl;
        return 1;
    }
}
