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
void print_phase2(const Phs phase) {
    const Phs phase2 = phase % 2;
    if (phase2 == 0) printf("+1");
    if (phase2 == 1) printf("-1");
}

static
void print_phase4(const Phs phase) {
    const Phs phase4 = phase % 4;
    if (phase4 == 0) printf("+1");
    if (phase4 == 1) printf("+i");
    if (phase4 == 2) printf("-1");
    if (phase4 == 3) printf("-i");
}

static
void print_pauli(const Bit x, const Bit z) {
    if (!x && !z) printf("I");
    if (x && !z) printf("X");
    if (!x && z) printf("Z");
    if (x && z) printf("Y");
}

static
void print_pauli_row(const Qid qubits_n, const Bit *row) {
    for (int qubit_i = 0; qubit_i < qubits_n; ++qubit_i) {
        const Bit x = row[qubit_i];
        const Bit z = row[qubits_n + qubit_i];
        print_pauli(x, z);
        printf(" ");
    }
}

static
void print_table(const Bit *table, const Qid qubits_n) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    printf("\ttable:\n");
    for (int row_i = 0; row_i < rows_n; ++row_i) {
        printf("\t\t");
        const Bit *row = table + (row_i * cols_n);
        print_pauli_row(qubits_n, row);
        print_phase2(row[qubits_n * 2]);
        printf("\n");
    }
}

static
void print_decomp_bits(const Bit *decomp_bits, const Qid qubits_n) {
    printf("\tdecomposed bits:\n");
    printf("\t\t");
    for (int i = 0; i < 2 * qubits_n; ++i) {
        print_bit(decomp_bits[i]);
        printf(" ");
    }
    printf("\n");
}

static
void print_decomp_pauli_row(const Qid qubits_n, const Bit *decomp_pauli, const Phs decomp_phase) {
    printf("\tdecomposed pauli row:\n");
    printf("\t\t");
    print_pauli_row(qubits_n, decomp_pauli);
    print_phase4(decomp_phase);
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
    const auto decomp_bits = new Bit[rows_n];
    const auto decomp_pauli = new Bit[rows_n];
    auto decomp_phase = Phs();
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        printf("\nshot_i=%u\n", shot_i);
        cudaMemcpy(
            table,
            simulator.table + (shot_i * rows_n * cols_n),
            rows_n * cols_n * sizeof(Bit),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            decomp_bits,
            simulator.decomp_bits + (shot_i * rows_n),
            rows_n * sizeof(Bit),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            decomp_pauli,
            simulator.decomp_pauli + (shot_i * rows_n),
            rows_n * sizeof(Bit),
            cudaMemcpyDeviceToHost);
        cudaMemcpy(
            &decomp_phase,
            simulator.decomp_phase + shot_i,
            sizeof(Phs),
            cudaMemcpyDeviceToHost);
        print_table(table, qubits_n);
        print_decomp_bits(decomp_bits, qubits_n);
        print_decomp_pauli_row(qubits_n, decomp_pauli, decomp_phase);
    }
    delete[] table;
    delete[] decomp_bits;
    delete[] decomp_pauli;
}

#endif
