#ifndef STN_CUDA_PRINT_CUH
#define STN_CUDA_PRINT_CUH

#include "../source/simulator.hpp"

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
void print_bits(const Bit *bits, const unsigned int n) {
    for (int i = 0; i < n; ++i) {
        print_bit(bits[i]);
        printf(" ");
    }
}

static
void print_sign(Bit sign) {
    sign %= 2;
    if (sign == 0) printf("+1");
    if (sign == 1) printf("-1");
}

static
void print_phase(Phs phase) {
    phase %= 4;
    if (phase == 0) printf("+1");
    if (phase == 1) printf("+i");
    if (phase == 2) printf("-1");
    if (phase == 3) printf("-i");
}

static
void print_pauli(const Bit x, const Bit z) {
    if (!x && !z) printf("I");
    if (x && !z) printf("X");
    if (!x && z) printf("Z");
    if (x && z) printf("Y");
}

static
void print_pauli_row(const PauliRowPtr ptr) {
    for (int qubit_i = 0; qubit_i < ptr.qubits_n; ++qubit_i) {
        const Bit x = *ptr.get_x_ptr(qubit_i);
        const Bit z = *ptr.get_z_ptr(qubit_i);
        print_pauli(x, z);
        printf(" ");
    }
}


static
void print_table_row(const TableRowPtr ptr) {
    print_pauli_row(ptr.get_pauli_ptr());
    print_sign(*ptr.get_sign_ptr());
}

static
void print_table(const TablePtr ptr) {
    printf("\t\ttable:\n");
    for (int row_i = 0; row_i < ptr.get_rows_n(); ++row_i) {
        printf("\t\t\t");
        print_table_row(ptr.get_row_ptr(row_i));
        printf("\n");
    }
}

static
void print_decomp(const DecompPtr ptr) {
    printf("\t\tdecomposed:\n");

    printf("\t\t\t");
    print_bits(ptr.get_bits_ptr(), 2 * ptr.qubits_n);
    printf("\n");

    printf("\t\t\t");
    print_pauli_row(ptr.get_pauli_ptr());
    print_phase(*ptr.get_phase_ptr());
    printf("\n");
}

static
void print_shot_state(const ShotStatePtr ptr) {
    print_table(ptr.get_table_ptr());
    print_decomp(ptr.get_decomp_ptr());
}

static
void print_shots_state(const ShotsStatePtr ptr) {
    for (Sid shot_i = 0; shot_i < ptr.shots_n; ++shot_i) {
        printf("\tshot %u:\n", shot_i);
        print_shot_state(ptr.get_shot_state_ptr(shot_i));
        printf("\n");
    }
}

static
void print_simulator(const Simulator &simulator) {
    const auto [shots_n, qubits_n, ptr] = simulator.shots_state_ptr;

    printf("\nSimulator:\n");
    printf("\tshots_n:%u\n", shots_n);
    printf("\tqubits_n:%u\n", qubits_n);
    printf("\n");

    const size_t state_bytes_n = ShotsStatePtr::compute_bytes_n(shots_n, qubits_n);
    const auto buffer_ptr = static_cast<char*>(malloc(state_bytes_n));
    cudaMemcpy(buffer_ptr, ptr, state_bytes_n, cudaMemcpyDeviceToHost);

    print_shots_state({shots_n, qubits_n, buffer_ptr});

    free(buffer_ptr);
}

#endif
