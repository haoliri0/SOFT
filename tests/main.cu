#include "cstdio"
#include "../source/decompose.hpp"
#include "../source/simulator.hpp"
#include "./utils/print.hpp"

using namespace StnCuda;

void test_simulator() {
    printf("\n\n### test_simulator ###\n");

    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(2, 2, 4);
        if (err != cudaSuccess) break;

        err = simulator.apply_x(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_y(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_z(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_h(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_s(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_sdg(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_cx(0, 1);
        if (err != cudaSuccess) break;

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        print_cuda_error(err);

    simulator.destroy();
}

void test_decompose() {
    printf("\n\n### test_decompose ###\n");

    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(2, 2, 4);
        if (err != cudaSuccess) break;

        constexpr Qid target = 0;
        decompose_gate_z(
            simulator.shots_n,
            simulator.qubits_n,
            simulator.table,
            simulator.dest_bits,
            target);

        err = cuda_compute_decomposed_phase(
            simulator.shots_n,
            simulator.qubits_n,
            simulator.table,
            simulator.dest_bits,
            simulator.decomp_pauli,
            simulator.decomp_phase,
            simulator.stream);
        if (err != cudaSuccess) break;

        cudaDeviceSynchronize();

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        print_cuda_error(err);

    simulator.destroy();
}

int main() {
    test_simulator();
    test_decompose();
}
