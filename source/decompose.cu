#include "./decompose.hpp"
#include "./simulator.hpp"
#include "./threads.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;


struct ComputeDecomposedBitsArgs {
    ShotsStatePtr ptr;
    Qid target;
};

static __device__
void op_compute_decomposed_bits(const ComputeDecomposedBitsArgs args, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Qid const row_i = dims_idx.get<1>();
    Qid const qubits_n = args.ptr.qubits_n;
    Qid const rows_n = 2 * qubits_n;
    Qid const target = args.target;

    const PauliRowPtr pauli_row_ptr = args.ptr
        .get_shot_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i)
        .get_pauli_ptr();

    // 在分解 Z 门时，根据 target 位置的门的类型
    // 判断某个 (de)stabilizer 是否应该存在
    const Bit x = *pauli_row_ptr.get_x_ptr(target);
    const Bit z = *pauli_row_ptr.get_z_ptr(target);

    Bit *bits_ptr = args.ptr
        .get_shot_ptr(shot_i)
        .get_decomp_ptr()
        .get_bits_ptr();

    // 注意，这里计算结果是反的
    // stabilizer 算出的结果放到 destabilizer bit
    // destabilizer 算出的结果放到 stabilizer bit
    Bit &bit = *(bits_ptr + (row_i + qubits_n) % rows_n);

    const Bit isX = x && !z;
    const Bit isY = x && z;
    const Bit isAntiComm = isX || isY;
    bit = isAntiComm;
}

void cuda_compute_decomposed_bits(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    const Qid target
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ComputeDecomposedBitsArgs, 2, op_compute_decomposed_bits>
        (stream, {shots_state_ptr, target}, dimsof(shots_n, rows_n));
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
void compute_multiply_pauli_string(
    const PauliRowPtr pauli_row_ptr0,
    const PauliRowPtr pauli_row_ptr1,
    PauliRowPtr const pauli_row_ptr,
    Phs &phase
) {
    const Qid qubits_n = pauli_row_ptr.qubits_n;
    for (int qubit_i = 0; qubit_i < qubits_n; qubit_i++) {
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

struct ComputeDecomposedPhaseArgs {
    ShotsStatePtr shots_state_ptr;
};

static __device__
void op_compute_decomposed_phase(const ComputeDecomposedPhaseArgs args, const DimsIdx<1> dims_idx) {
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    Qid const qubits_n = shots_state_ptr.qubits_n;
    Qid const rows_n = 2 * qubits_n;
    Qid const cols_n = 2 * qubits_n;

    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

    const TablePtr table_ptr = shot_state_ptr.get_table_ptr();
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *decomp_bits_ptr = decomp_ptr.get_destab_bits_ptr();
    const PauliRowPtr decomp_pauli_row = decomp_ptr.get_pauli_ptr();
    Phs &decomp_phase = *decomp_ptr.get_phase_ptr();

    // clear decomp pauli row
    for (Qid col_i = 0; col_i < cols_n; ++col_i)
        *decomp_pauli_row.get_bit_ptr(col_i) = false;

    // clear decomp phase
    decomp_phase = 0;

    for (Qid row_i = 0; row_i < rows_n; ++row_i) {
        if (decomp_bits_ptr[row_i]) {
            const TableRowPtr table_row_ptr = table_ptr.get_row_ptr(row_i);
            const PauliRowPtr table_pauli_row_ptr = table_row_ptr.get_pauli_ptr();

            // add multiply pauli phase
            compute_multiply_pauli_string(
                decomp_pauli_row,
                table_pauli_row_ptr,
                decomp_pauli_row,
                decomp_phase);

            // add table row phase
            const Bit table_row_sign = *table_row_ptr.get_sign_ptr();
            decomp_phase += static_cast<Phs>(table_row_sign) * 2;
        }
    }
}

void cuda_compute_decomposed_phase(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ComputeDecomposedPhaseArgs, 1, op_compute_decomposed_phase>
        (stream, {shots_state_ptr}, dimsof(shots_n));
}
