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
void print_int_bits(Aid integer, const unsigned int n) {
    for (int i = 0; i < n; ++i) {
        const Bit bit = integer % 2;
        integer >>= 1;
        print_bit(bit);
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
void print_amplitude(const Amp amp) {
    const auto real = amp.real();
    const auto imag = amp.imag();
    const char *real_sign = real >= 0 ? "+" : "-";
    const char *imag_sign = imag >= 0 ? "+" : "-";
    printf("%s %f %s %f i", real_sign, abs(real), imag_sign, abs(imag));
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
void print_pivot(const Qid pivot) {
    if (pivot == NullPivot) {
        printf("pivot=null");
    } else {
        printf("pivot=%u", pivot);
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
    const Qid qubits_n = ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    for (int row_i = 0; row_i < rows_n; ++row_i) {
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
    print_pivot(*ptr.get_pivot_ptr());
    printf("\n");

    printf("\t\t\t");
    print_pauli_row(ptr.get_pauli_ptr());
    print_phase(*ptr.get_phase_ptr());
    printf("\n");
}

static
void print_amps(const AmpsMapPtr ptr) {
    printf("\t\tamplitudes:\n");
    const Kid amps_n = *ptr.get_amps_n_ptr();
    printf("\t\t\t(amps_n = %u)\n", amps_n);
    for (int amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = *(ptr.get_aids_ptr() + amp_i);
        const Amp amp = *(ptr.get_amps_ptr() + amp_i);
        printf("\t\t\t");
        print_int_bits(aid, ptr.qubits_n);
        printf(": ");
        print_amplitude(amp);
        printf("\n");
    }
    printf("\n");
}

static
void print_amps_halves(const AmpsMapPtr ptr) {
    printf("\t\tamplitudes:\n");
    const Kid amps_m = ptr.amps_m;
    const Kid amps_n = *ptr.get_amps_n_ptr();
    printf("\t\t\t(amps_n = %u)\n", amps_n);
    if (amps_n > amps_m / 2) {
        printf("\t\t\tToo Large!");
        return;
    }

    for (int amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = *(ptr.get_aids_ptr() + amp_i);
        const Amp amp = *(ptr.get_amps_ptr() + amp_i);
        printf("\t\t\t");
        print_int_bits(aid, ptr.qubits_n);
        printf(": ");
        print_amplitude(amp);
        printf("\n");
    }

    printf("\t\t\t----------------\n");

    for (int amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = *ptr.get_aid_ptr(amps_m / 2 + amp_i);
        const Amp amp = *ptr.get_amp_ptr( amps_m / 2 + amp_i);
        printf("\t\t\t");
        print_int_bits(aid, ptr.qubits_n);
        printf(": ");
        print_amplitude(amp);
        printf("\n");
    }

    printf("\n");
}


static
void print_shot_state(const ShotStatePtr ptr) {
    print_table(ptr.get_table_ptr());
    print_decomp(ptr.get_decomp_ptr());
    print_amps_halves(ptr.get_amps_ptr());
}

static
void print_shots_state(const ShotsStatePtr ptr) {
    for (Sid shot_i = 0; shot_i < ptr.shots_n; ++shot_i) {
        printf("\tshot %u:\n", shot_i);
        print_shot_state(ptr.get_shot_ptr(shot_i));
        printf("\n");
    }
}

static
void print_simulator(const Simulator &simulator) {
    const Sid shots_n = simulator.shots_state_ptr.shots_n;
    const Qid qubits_n = simulator.shots_state_ptr.qubits_n;
    const Kid amps_m = simulator.shots_state_ptr.amps_m;
    const char *ptr = simulator.shots_state_ptr.ptr;

    printf("\nSimulator:\n");
    printf("\tshots_n:%u\n", shots_n);
    printf("\tqubits_n:%u\n", qubits_n);
    printf("\tamps_m:%u\n", amps_m);
    printf("\n");

    cudaDeviceSynchronize();

    const size_t state_bytes_n = simulator.shots_state_ptr.get_size_bytes_n();
    const auto buffer_ptr = static_cast<char *>(malloc(state_bytes_n));
    cudaMemcpy(buffer_ptr, ptr, state_bytes_n, cudaMemcpyDeviceToHost);

    print_shots_state({shots_n, qubits_n, amps_m, buffer_ptr});

    free(buffer_ptr);
}

#endif
