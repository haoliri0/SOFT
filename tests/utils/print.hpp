#ifndef STN_CUDA_UTILS_PRINT_CUH
#define STN_CUDA_UTILS_PRINT_CUH

#include "../../source/simulator.hpp"

using namespace StnCuda;

static
void print_cuda_error(const cudaError_t error) {
    printf("%s\n%s", cudaGetErrorName(error), cudaGetErrorString(error));
}

static
void print_bit(const Bit bit) {
    if (bit) printf("1");
    else printf("0");
}

static
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

static
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

static
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

#endif
