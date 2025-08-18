#ifndef STN_CUDA_PRINT_HPP
#define STN_CUDA_PRINT_HPP

#include "./simulator.hpp"
#include "./exceptions.hpp"

using namespace StnCuda;

static
void print_cuda_error(const cudaError error) {
    printf("%s\n%s", cudaGetErrorName(error), cudaGetErrorString(error));
}

static
void print_indent(const unsigned int indent) {
    for (int i = 0; i < indent; ++i) {
        printf("  ");
    }
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
    }
}

static
void print_sign(Bit sign) {
    !(sign %= 2)
        ? printf("+")
        : printf("-");
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
    printf("%s%f %s%f i", real_sign, abs(real), imag_sign, abs(imag));
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
    for (Qid qubit_i = 0; qubit_i < ptr.qubits_n; ++qubit_i) {
        const Bit x = *ptr.get_x_ptr(qubit_i);
        const Bit z = *ptr.get_z_ptr(qubit_i);
        print_pauli(x, z);
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
void print_table_row(const TableRowPtr ptr, const unsigned int indent) {
    print_indent(indent);
    print_sign(*ptr.get_sign_ptr());
    print_pauli_row(ptr.get_pauli_ptr());
    printf("\n");
}

static
void print_table(const TablePtr ptr, const unsigned int indent) {
    print_indent(indent);
    printf("table:\n");
    const Qid qubits_n = ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    for (Qid row_i = 0; row_i < rows_n; ++row_i)
        print_table_row(ptr.get_row_ptr(row_i), indent + 1);
}

static
void print_decomp(const DecompPtr ptr, const unsigned int indent) {
    print_indent(indent);
    printf("decomposed:\n");

    print_indent(indent + 1);
    print_bits(ptr.get_bits_ptr(), 2 * ptr.qubits_n);
    print_pivot(*ptr.get_pivot_ptr());
    printf("\n");

    print_indent(indent + 1);
    print_pauli_row(ptr.get_pauli_ptr());
    printf(" ");
    print_phase(*ptr.get_phase_ptr());
    printf("\n");
}

static
void print_amp_entry(const AmpsMapPtr ptr, const Kid amp_i, const unsigned int indent) {
    const Aid aid = *ptr.get_aid_ptr(amp_i);
    const Amp amp = *ptr.get_amp_ptr(amp_i);
    print_indent(indent);
    print_int_bits(aid, ptr.qubits_n);
    printf(" : ");
    print_amplitude(amp);
    printf("\n");
}

static
void print_amps(const AmpsMapPtr ptr, const bool full, const unsigned int indent) {
    const Kid amps_m = ptr.amps_m;
    const Kid amps_n = *ptr.get_amps_n_ptr();
    print_indent(indent);
    printf("amplitudes:\n");
    print_indent(indent + 1);
    printf("amps_n=%u\n", amps_n);
    for (Kid amp_i = 0; amp_i < (full ? amps_m : amps_n); ++amp_i)
        print_amp_entry(ptr, amp_i, indent + 1);
}

static
void print_result_item(const ResultsPtr ptr, const Rid result_idx, const unsigned int indent) {
    const Rid results_m = ptr.results_m;
    const Rid result_i = result_idx % results_m;
    const Rvl result_value = *ptr.get_value_ptr(result_i);
    const Flt result_prob = *ptr.get_prob_ptr(result_i);
    print_indent(indent + 1);
    printf("idx=%04u,value=%u,prob=%f\n", result_idx, result_value, result_prob);
}

static
void print_results(const ResultsPtr ptr, const unsigned int indent) {
    const Rid results_m = ptr.results_m;
    const Rid results_n = *ptr.get_results_n_ptr();
    const Rid results_idx0 = results_n > results_m ? results_n - results_m : 0;
    if (results_n == 0) return;

    print_indent(indent);
    printf("results:\n");
    for (Rid result_idx = results_idx0; result_idx < results_n; ++result_idx)
        print_result_item(ptr, result_idx, indent + 1);
}


static
void print_shot_state(const ShotStatePtr &ptr, const unsigned int indent) {
    print_table(ptr.get_table_ptr(), indent + 1);
    print_decomp(ptr.get_decomp_ptr(), indent + 1);
    print_amps(ptr.get_amps_ptr(), true, indent + 1);
    print_results(ptr.get_results_ptr(), indent + 1);
}

static
void print_shots_state(const ShotsStatePtr &ptr, const unsigned int indent) {
    for (Sid shot_i = 0; shot_i < ptr.shots_n; ++shot_i) {
        print_indent(indent);
        printf("shot %u:\n", shot_i);
        print_shot_state(ptr.get_shot_ptr(shot_i), indent + 1);
        printf("\n");
    }
}


static
void print_simulator_args(const Simulator &simulator, const unsigned int indent) {
    const Sid shots_n = simulator.shots_state_ptr.shots_n;
    const Qid qubits_n = simulator.shots_state_ptr.qubits_n;
    const Kid amps_m = simulator.shots_state_ptr.amps_m;
    const Rid results_m = simulator.shots_state_ptr.results_m;

    print_indent(indent);
    printf("shots_n=%u\n", shots_n);
    print_indent(indent);
    printf("qubits_n=%u\n", qubits_n);
    print_indent(indent);
    printf("amps_m=%u\n", amps_m);
    print_indent(indent);
    printf("results_m=%u\n", results_m);
}

static
void sync_and_print_simulator(const Simulator &simulator, const unsigned int indent) {
    cuda_check(cudaStreamSynchronize(simulator.stream));

    char buffer[simulator.shots_state_ptr.get_size_bytes_n()];
    cuda_check(cudaMemcpy(buffer, simulator.shots_state_ptr.ptr,
        simulator.shots_state_ptr.get_size_bytes_n(), cudaMemcpyDeviceToHost));
    ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    shots_state_ptr.ptr = buffer;

    printf("Simulator:\n");
    print_simulator_args(simulator, indent + 1);
    printf("\n");
    print_shots_state(shots_state_ptr, indent + 1);
    printf("\n");
}

static
void sync_and_print_simulator(const Simulator &simulator) {
    sync_and_print_simulator(simulator, 0);
}

#endif
