#ifndef SOFT_CUDA_DECOMPOSE_CUH
#define SOFT_CUDA_DECOMPOSE_CUH

#include "./datatype.cuh"

using namespace SoftCuda;

static __device__
Bit compute_sign(Bst key, const Bit *stab_bits, const Qid qubits_n) {
    Bit sign = false;
    for (Qid qubit_i = 0; qubit_i < qubits_n; ++qubit_i) {
        const Bit key_bit = key % 2;
        const Bit stab_bit = stab_bits[qubit_i];
        sign ^= stab_bit & key_bit;
        key >>= 1;
    }
    return sign;
}


static __device__
void compute_multiply_pauli_phase(
    const Bit gate0_x,
    const Bit gate0_z,
    const Bit gate1_x,
    const Bit gate1_z,
    Phs &phase
) {
    const Phs pauli_phase[] = {
        0, // 00 00, I I = I
        0, // 00 01, I Z = Z
        0, // 00 10, I X = X
        0, // 00 11, I Y = Y
        0, // 01 00, Z I = Z
        0, // 01 01, Z Z = I
        1, // 01 10, Z X = iY
        3, // 01 11, Z Y = -iX
        0, // 10 00, X I = X
        3, // 10 01, X Z = -iY
        0, // 10 10, X X = I
        1, // 10 11, X Y = iZ
        0, // 11 00, Y I = Y
        1, // 11 01, Y Z = iX
        3, // 11 10, Y X = -iZ
        0, // 11 11, Y Y = I
    };

    const int gate0_x_int = gate0_x ? 1 : 0;
    const int gate0_z_int = gate0_z ? 1 : 0;
    const int gate1_x_int = gate1_x ? 1 : 0;
    const int gate1_z_int = gate1_z ? 1 : 0;
    const int index = 0 | gate0_x_int << 3 | gate0_z_int << 2 | gate1_x_int << 1 | gate1_z_int << 0;
    phase += pauli_phase[index];
}

static __device__
void compute_multiply_pauli_rows(
    const PauliRowPtr pauli_row_ptr0,
    const PauliRowPtr pauli_row_ptr1,
    PauliRowPtr const pauli_row_ptr,
    Phs &phase
) {
    const Qid qubits_n = pauli_row_ptr.qubits_n;
    for (Qid qubit_i = 0; qubit_i < qubits_n; qubit_i++) {
        const Bit gate0_x = *pauli_row_ptr0.get_x_ptr(qubit_i);
        const Bit gate0_z = *pauli_row_ptr0.get_z_ptr(qubit_i);
        const Bit gate1_x = *pauli_row_ptr1.get_x_ptr(qubit_i);
        const Bit gate1_z = *pauli_row_ptr1.get_z_ptr(qubit_i);
        Bit &gate_x = *pauli_row_ptr.get_x_ptr(qubit_i);
        Bit &gate_z = *pauli_row_ptr.get_z_ptr(qubit_i);
        compute_multiply_pauli_phase(gate0_x, gate0_z, gate1_x, gate1_z, phase);
        gate_x = gate0_x ^ gate1_x;
        gate_z = gate0_z ^ gate1_z;
    }
}

#endif
