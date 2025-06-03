#include "cstdio"
#include "../source/simulator.hpp"

using namespace StnCuda;

void printCudaError(const cudaError_t error) {
    printf("%s\n%s", cudaGetErrorName(error), cudaGetErrorString(error));
}

void print_item(const bool item) {
    if (item) printf("1");
    else printf("0");
}

void print_table(const Sti *table, const Qid rows_n, const Qid cols_n) {
    printf("table=\n");
    for (int row_i = 0; row_i < rows_n; ++row_i) {
        printf("  ");
        for (int col_i = 0; col_i < cols_n; ++col_i) {
            print_item(table[row_i * cols_n + col_i]);
            printf(" ");
        }
        printf("\n");
    }
}

void print_simulator(const Simulator &simulator) {
    const Sid shots_n = simulator.shots_n;
    const Qid qubits_n = simulator.qubits_n;
    const Aid map_limit = simulator.map_limit;
    printf("shots_n: %u\n", shots_n);
    printf("qubits_n: %u\n", qubits_n);
    printf("map_limit: %u\n", map_limit);

    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;
    const auto table = new Sti[rows_n * cols_n];
    for (int shot_i = 0; shot_i < shots_n; ++shot_i) {
        cudaMemcpy(table, simulator.table, rows_n * cols_n * sizeof(bool), cudaMemcpyDeviceToHost);
        print_table(table, rows_n, cols_n);
    }
    delete[] table;
}

void test_simulator() {
    Simulator simulator;
    cudaError_t err;

    do {
        err = simulator.create(1, 2, 4);
        if (err != cudaSuccess) break;

        err = simulator.apply_x(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_y(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_z(0);
        if (err != cudaSuccess) break;

        err = simulator.apply_h(0);
        if (err != cudaSuccess) break;

        print_simulator(simulator);

    } while (false);

    if (err != cudaSuccess)
        printCudaError(err);

    simulator.destroy();
}

int main() {
    test_simulator();
}
