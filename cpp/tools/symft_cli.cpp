#include "frontend/stim.hpp"
#include "sampler/single_shot.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "usage: symft_cli <circuit.stim> [shots]\n";
        return 2;
    }
    try {
        const std::string path = argv[1];
        const int shots = argc == 3 ? std::stoi(argv[2]) : 1000;
        const auto parsed = symft::parse_stim_file(path);
        symft::PendingFactoredState pending(parsed.state);
        const auto program = symft::plan_factored_updates(std::move(pending));
        const auto summary = symft::estimate_stim_logical_error_rate(parsed, shots);
        std::cout << "qubits " << parsed.state.n << "\n";
        std::cout << "records " << program.nrecords << "\n";
        std::cout << "max_active_qubits " << program.max_k << "\n";
        std::cout << "simd_backend " << symft::active_simd_backend() << "\n";
        std::cout << "shots " << summary.shots << "\n";
        std::cout << "discarded " << summary.discarded << "\n";
        std::cout << "accepted " << summary.accepted << "\n";
        std::cout << "logical_errors " << summary.logical_errors << "\n";
        std::cout << "discard_rate " << symft::discard_rate(summary) << "\n";
        std::cout << "logical_error_rate " << symft::logical_error_rate(summary) << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "symft_cli: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
