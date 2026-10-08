#pragma once

#include <array>
#include <chrono>
#include <cstdint>

namespace symft {

enum class CpuPhase : unsigned {
    Noise, Expression, Reset, Classical, Rotation, Promotion, Measurement,
    Probability, Projection, Compaction, Count, Entrance, Size
};
inline constexpr unsigned cpu_phase_count = static_cast<unsigned>(CpuPhase::Size);
inline constexpr std::array<const char*, cpu_phase_count> cpu_phase_names{
    "noise", "expression", "reset", "classical", "rotation", "promotion",
    "measurement_inclusive", "probability", "projection", "compaction", "count", "entrance"};

struct CpuProfile {
    std::array<double, cpu_phase_count> seconds{};
    std::array<std::uint64_t, cpu_phase_count> calls{};
    std::array<std::array<std::uint64_t, 17>, cpu_phase_count> shot_visits{};
};

#if defined(SYMFT_CPU_DIAGNOSTICS)
inline constexpr bool cpu_diagnostics_available = true;
inline thread_local CpuProfile* active_cpu_profile = nullptr;
struct ScopedCpuTimer {
    using Clock = std::chrono::steady_clock;
    CpuProfile* profile;
    unsigned phase;
    Clock::time_point start;
    explicit ScopedCpuTimer(CpuPhase p, int k = -1, std::uint64_t visits = 0)
        : profile(active_cpu_profile), phase(static_cast<unsigned>(p)) {
        if (profile) {
            start = Clock::now();
            ++profile->calls[phase];
            if (k >= 0 && k <= 16) profile->shot_visits[phase][k] += visits;
        }
    }
    ~ScopedCpuTimer() {
        if (profile) profile->seconds[phase] += std::chrono::duration<double>(Clock::now() - start).count();
    }
};
#else
inline constexpr bool cpu_diagnostics_available = false;
inline thread_local CpuProfile* active_cpu_profile = nullptr;
struct ScopedCpuTimer {
    explicit ScopedCpuTimer(CpuPhase, int = -1, std::uint64_t = 0) {}
};
#endif

// FactoredInstruction variant order: rotation, promotion, record, detector,
// active measurement, dormant branch. Record/detector/branch are classical.
inline CpuPhase cpu_instruction_phase(std::size_t variant_index) {
    switch (variant_index) {
        case 0: return CpuPhase::Rotation;
        case 1: return CpuPhase::Promotion;
        case 4: return CpuPhase::Measurement;
        default: return CpuPhase::Classical;
    }
}
} // namespace symft
