#include "frontend/stim.hpp"
#include "sampler/batch_sampler.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct BenchResult {
    std::string fixture;
    int n = 0;
    int initial_k = 0;
    int max_k = 0;
    std::size_t instructions = 0;
    int records = 0;
    int symbols = 0;
    int batch_size = 0;
    int blocks = 0;
    double parse_s = 0.0;
    double plan_s = 0.0;
    double elapsed_s = 0.0;
    double shots_per_s = 0.0;
    std::uint64_t checksum = 0;
};

double seconds_between(Clock::time_point start, Clock::time_point stop) {
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t parse_u64(const char* raw, const char* name) {
    if (raw[0] == '-') {
        throw std::runtime_error(std::string(name) + " must be nonnegative");
    }
    errno = 0;
    char* end = nullptr;
    const auto value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || errno == ERANGE) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + raw);
    }
    return static_cast<std::uint64_t>(value);
}

int parse_positive_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value == 0 || value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must be in 1:Int32Max");
    }
    return static_cast<int>(value);
}

int parse_nonnegative_int(const char* raw, const char* name) {
    const std::uint64_t value = parse_u64(raw, name);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " must fit in Int32Max");
    }
    return static_cast<int>(value);
}

int parse_batch_size(const char* raw) {
    const std::string value(raw);
    if (value == "auto" || value == "default") {
        return 0;
    }
    return parse_positive_int(raw, "batch_size");
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

std::uint64_t low_bits_mask(int nbits) {
    if (nbits <= 0) {
        return 0;
    }
    if (nbits >= 64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return (std::uint64_t{1} << nbits) - 1;
}

std::uint64_t live_word_mask(int active_shots, std::size_t word) {
    return low_bits_mask(active_shots - static_cast<int>(word << 6));
}

std::size_t measurement_offset(const symft::BatchFactoredExecutorState& runtime, int record, std::size_t word) {
    return static_cast<std::size_t>(record - 1) * runtime.batch_words + word;
}

std::uint64_t consume_measurements(const symft::BatchFactoredExecutorState& runtime) {
    std::uint64_t checksum = 0;
    const std::size_t nwords = static_cast<std::size_t>((runtime.active_shots + 63) >> 6);
    for (int record = 1; record <= runtime.nrecords; ++record) {
        for (std::size_t word = 0; word < nwords; ++word) {
            checksum += static_cast<std::uint64_t>(popcount64(
                runtime.measurement_words[measurement_offset(runtime, record, word)] &
                live_word_mask(runtime.active_shots, word)));
        }
    }
    return checksum;
}

std::uint64_t run_blocks(
    symft::BatchFactoredExecutorState& runtime,
    const symft::FactoredInstructionProgram& program,
    int shots) {
    std::uint64_t checksum = 0;
    int offset = 0;
    while (offset < shots) {
        const int block = std::min(runtime.batches, shots - offset);
        symft::reset_batch_executor(runtime, program, block);
        symft::execute_batch_in_place(runtime, program);
        checksum += consume_measurements(runtime);
        offset += block;
    }
    return checksum;
}

BenchResult run_fixture(
    const std::string& fixture,
    int shots,
    int requested_batch_size,
    int warmup_blocks) {
    const auto parse_start = Clock::now();
    const auto parsed = symft::parse_stim_file(fixture);
    const auto parse_stop = Clock::now();

    const auto plan_start = Clock::now();
    symft::PendingFactoredState pending(parsed.state);
    const auto program = symft::plan_factored_updates(pending);
    const auto plan_stop = Clock::now();

    const int batch_size = requested_batch_size > 0 ? requested_batch_size : symft::default_batch_count(program.max_k);
    symft::BatchFactoredExecutorState runtime(program, batch_size, 0x5eedULL);
    runtime.dense_shot_major_active = true;

    for (int warmup = 0; warmup < warmup_blocks; ++warmup) {
        symft::reset_batch_executor(runtime, program, runtime.batches);
        symft::execute_batch_in_place(runtime, program);
        (void)consume_measurements(runtime);
    }

    const auto sample_start = Clock::now();
    const std::uint64_t checksum = run_blocks(runtime, program, shots);
    const auto sample_stop = Clock::now();

    BenchResult result;
    result.fixture = fixture;
    result.n = program.n;
    result.initial_k = program.initial_k;
    result.max_k = program.max_k;
    result.instructions = program.instructions.size();
    result.records = program.nrecords;
    result.symbols = program.nsymbols;
    result.batch_size = batch_size;
    result.blocks = (shots + batch_size - 1) / batch_size;
    result.parse_s = seconds_between(parse_start, parse_stop);
    result.plan_s = seconds_between(plan_start, plan_stop);
    result.elapsed_s = seconds_between(sample_start, sample_stop);
    result.shots_per_s = result.elapsed_s > 0.0 ? static_cast<double>(shots) / result.elapsed_s : 0.0;
    result.checksum = checksum;
    return result;
}

void print_header(int shots, int batch_size, int warmup_blocks) {
    std::cout << "SymFT C++ batched FactoredInstructionProgram sampler\n";
    std::cout << "shots=" << shots
              << " batch_size=" << (batch_size > 0 ? std::to_string(batch_size) : std::string("auto"))
              << " warmup_blocks=" << warmup_blocks
              << " dense_backend=" << symft::active_batch_backend()
              << "\n";
    std::cout << std::left << std::setw(12) << "fixture"
              << std::right << std::setw(5) << "n"
              << std::setw(5) << "k0"
              << std::setw(7) << "max_k"
              << std::setw(8) << "instr"
              << std::setw(8) << "records"
              << std::setw(8) << "symbols"
              << std::setw(8) << "blocks"
              << std::setw(10) << "batch"
              << std::setw(12) << "parse_s"
              << std::setw(12) << "plan_s"
              << std::setw(12) << "elapsed_s"
              << std::setw(18) << "shots/s"
              << std::setw(12) << "checksum"
              << "\n";
}

void print_result(const BenchResult& result) {
    std::cout << std::left << std::setw(12) << result.fixture
              << std::right << std::setw(5) << result.n
              << std::setw(5) << result.initial_k
              << std::setw(7) << result.max_k
              << std::setw(8) << result.instructions
              << std::setw(8) << result.records
              << std::setw(8) << result.symbols
              << std::setw(8) << result.blocks
              << std::setw(10) << result.batch_size
              << std::setw(12) << std::fixed << std::setprecision(4) << result.parse_s
              << std::setw(12) << std::fixed << std::setprecision(4) << result.plan_s
              << std::setw(12) << std::fixed << std::setprecision(4) << result.elapsed_s
              << std::setw(18) << std::fixed << std::setprecision(2) << result.shots_per_s
              << std::setw(12) << result.checksum
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 5 || (argc >= 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help"))) {
        std::cerr << "usage: symft_batch_bench [shots=10000] [batch_size=auto] [warmup_blocks=1] [fixture=all]\n";
        return argc > 5 ? 2 : 0;
    }

    try {
        const int shots = argc >= 2 ? parse_positive_int(argv[1], "shots") : 10000;
        const int batch_size = argc >= 3 ? parse_batch_size(argv[2]) : 0;
        const int warmup_blocks = argc >= 4 ? parse_nonnegative_int(argv[3], "warmup_blocks") : 1;
        const std::string fixture_arg = argc >= 5 ? argv[4] : "all";

        std::vector<std::string> fixtures;
        if (fixture_arg == "all") {
            fixtures = {
                "benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim",
                "benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim",
            };
        } else {
            fixtures = {fixture_arg};
        }

        print_header(shots, batch_size, warmup_blocks);
        for (const auto& fixture : fixtures) {
            print_result(run_fixture(fixture, shots, batch_size, warmup_blocks));
        }
    } catch (const std::exception& ex) {
        std::cerr << "symft_batch_bench: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
