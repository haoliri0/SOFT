#include "cstdio"
#include "./print.hpp"
#include "../source/decompose.hpp"

using namespace StnCuda;

/** 测试 simulator 的创建和销毁 */
void test_simulator() {
    printf("\n\n### test_simulator ###\n");

    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(1, 2, 4);
        if (err != cudaSuccess) break;

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        print_cuda_error(err);

    simulator.destroy();
}

/** 测试 clifford gates */
void test_gates() {
    printf("\n\n### test_gates ###\n");

    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(1, 2, 4);
        if (err != cudaSuccess) break;

        simulator.apply_x(0);
        simulator.apply_y(0);
        simulator.apply_z(0);
        simulator.apply_h(0);
        simulator.apply_s(0);
        simulator.apply_sdg(0);
        simulator.apply_cx(0, 1);

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        print_cuda_error(err);

    simulator.destroy();
}

/** 测试 Z 门的分解 */
void test_decompose() {
    printf("\n\n### test_decompose ###\n");

    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(1, 2, 4);
        if (err != cudaSuccess) break;

        constexpr Qid target = 0;
        cuda_compute_decomposed_bits(
            simulator.stream,
            simulator.shots_state_ptr,
            target);

        cuda_compute_decomposed_phase(
            simulator.stream,
            simulator.shots_state_ptr);

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        print_cuda_error(err);

    simulator.destroy();
}

/** 测试 T 门 */
void test_gates_t() {
    printf("\n\n### test_gates_t ###\n");

    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(1, 2, 4);
        if (err != cudaSuccess) break;

        simulator.apply_h(0);
        simulator.apply_t(0);
        simulator.apply_tdg(0);

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        print_cuda_error(err);

    simulator.destroy();
}


int main() {
    test_simulator();
    test_gates();
    test_decompose();
    test_gates_t();
}
