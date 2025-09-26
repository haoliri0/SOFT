#include "./decompose.hpp"
#include "./decompose.cuh"
#include "./simulator.hpp"
#include "./threads.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;


struct ArgsComputeDecomposedBits {
    const ShotsStatePtr shots_state_ptr;
    const Qid target;
};

static __device__
void op_compute_decomposed_bits(const ArgsComputeDecomposedBits args, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Qid const row_i = dims_idx.get<1>();
    ShotsStatePtr const shots_state_ptr = args.shots_state_ptr;
    ShotStatePtr const shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    Qid const qubits_n = shot_state_ptr.qubits_n;
    Qid const rows_n = 2 * qubits_n;
    Qid const target = args.target;

    const PauliRowPtr pauli_row_ptr = shot_state_ptr
        .get_table_ptr()
        .get_row_ptr(row_i)
        .get_pauli_ptr();
    const DecompPtr decomp_ptr = shot_state_ptr
        .get_decomp_ptr();

    // 在分解 Z 门时，根据 target 位置的门的类型
    // 判断某个 (de)stabilizer 是否应该存在
    const Bit x = *pauli_row_ptr.get_x_ptr(target);
    const Bit z = *pauli_row_ptr.get_z_ptr(target);

    // 注意，这里计算结果是反的
    // stabilizer 算出的结果放到 destabilizer bit
    // destabilizer 算出的结果放到 stabilizer bit
    Bit &bit = *decomp_ptr.get_bit_ptr((row_i + qubits_n) % rows_n);

    const Bit isX = x && !z;
    const Bit isY = x && z;
    const Bit isAntiComm = isX || isY;
    bit = isAntiComm;
}

void cuda_compute_decomposed_bits(
    cudaStream_t const &stream,
    ShotsStatePtr const shots_state_ptr,
    const Qid target
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsComputeDecomposedBits, 2, op_compute_decomposed_bits>
        (stream, {shots_state_ptr, target}, dimsof(shots_n, rows_n));
}


static __device__
void op_compute_decomposed_phase(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    Qid const qubits_n = shot_state_ptr.qubits_n;
    Qid const rows_n = 2 * qubits_n;
    Qid const cols_n = 2 * qubits_n;

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

    // compute
    for (Qid row_i = 0; row_i < rows_n; ++row_i) {
        if (decomp_bits_ptr[row_i]) {
            const TableRowPtr table_row_ptr = table_ptr.get_row_ptr(row_i);
            const PauliRowPtr table_pauli_row_ptr = table_row_ptr.get_pauli_ptr();

            // add multiply pauli phase
            compute_multiply_pauli_rows(
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
    cudaStream_t const &stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_compute_decomposed_phase>
        (stream, shots_state_ptr, dimsof(shots_n));
}


static __device__
void op_compute_decomp_pivot(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Qid qubits_n = decomp_ptr.qubits_n;
    const Bit *decomp_bits = decomp_ptr.get_destab_bits_ptr();
    Qid &decomp_pivot = *decomp_ptr.get_pivot_ptr();

    decomp_pivot = NullPivot;
    for (Qid row_i = 0; row_i < qubits_n; ++row_i) {
        if (decomp_bits[row_i]) {
            decomp_pivot = row_i;
            break;
        }
    }
}

void cuda_compute_decomp_pivot(
    cudaStream_t const &stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_compute_decomp_pivot>
        (stream, shots_state_ptr, dimsof(shots_n));
}
