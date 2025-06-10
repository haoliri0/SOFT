#include "cstdio"
#include "../source/decompose.hpp"
#include "../source/simulator.hpp"

using namespace StnCuda;

void printCudaError(const cudaError_t error) {
    printf("%s\n%s", cudaGetErrorName(error), cudaGetErrorString(error));
}

void print_bit(const bool bit) {
    if (bit) printf("1");
    else printf("0");
}

void print_table(const Bit *table, const Qid rows_n, const Qid cols_n) {
    printf("\ttable=\n");
    for (int row_i = 0; row_i < rows_n; ++row_i) {
        printf("\t\t");
        for (int col_i = 0; col_i < cols_n; ++col_i) {
            print_bit(table[row_i * cols_n + col_i]);
            printf(" ");
        }
        printf("\n");
    }
}

void print_stde_bits(const Bit *stde_bits, const Qid qubits_n) {
    printf("\tstabilizer bits:\n");
    printf("\t\t");
    for (int qubit_i = 0; qubit_i < qubits_n; ++qubit_i) {
        print_bit(stde_bits[qubit_i]);
        printf(" ");
    }
    printf("\n");
    printf("\tdestabilizer bits:\n");
    printf("\t\t");
    for (int qubit_i = 0; qubit_i < qubits_n; ++qubit_i) {
        print_bit(stde_bits[qubits_n + qubit_i]);
        printf(" ");
    }
    printf("\n");
}

void print_simulator(const Simulator &simulator) {
    const Sid shots_n = simulator.shots_n;
    const Qid qubits_n = simulator.qubits_n;
    const Aid map_limit = simulator.map_limit;
    printf("\ntotal:\n");
    printf("\tshots_n=%u\n", shots_n);
    printf("\tqubits_n=%u\n", qubits_n);
    printf("\tmap_limit=%u\n", map_limit);

    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;
    const auto table = new Bit[rows_n * cols_n];
    const auto stde_bits = new Bit[rows_n];
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        printf("\nshot_i=%u\n", shot_i);
        cudaMemcpy(
            table,
            simulator.table + (shot_i * rows_n * cols_n),
            rows_n * cols_n * sizeof(Bit),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            stde_bits,
            simulator.stde_bits + (shot_i * rows_n),
            rows_n * sizeof(Bit),
            cudaMemcpyDeviceToHost);
        print_table(table, rows_n, cols_n);
        print_stde_bits(stde_bits, qubits_n);
    }
    delete[] table;
    delete[] stde_bits;
}

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
        printCudaError(err);

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
            simulator.stde_bits,
            target);

        cudaDeviceSynchronize();

        print_simulator(simulator);
    } while (false);

    if (err != cudaSuccess)
        printCudaError(err);

    simulator.destroy();
}

int main() {
    test_simulator();
    test_decompose();
}
