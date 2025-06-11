#include "./decompose.hpp"
#include "./simulator.hpp"
#include "./threads.cuh"
#include "dimsop.cuh"

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
        .get_shot_state_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i)
        .get_pauli_ptr();

    // 在分解 Z 门时，根据 target 位置的门的类型
    // 判断某个 (de)stabilizer 是否应该存在
    const Bit x = *pauli_row_ptr.get_x_ptr(target);
    const Bit z = *pauli_row_ptr.get_z_ptr(target);

    Bit *bits_ptr = args.ptr
        .get_shot_state_ptr(shot_i)
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
    const CudaBit gate0_x,
    const CudaBit gate0_z,
    const CudaBit gate1_x,
    const CudaBit gate1_z,
    CudaPhs *const phase
) {
    const CudaPhs pauli_phase[] = {
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
    *phase += pauli_phase[index];
}

static __device__
void compute_multiply_pauli_string(
    const CudaQid qubits_n,
    const CudaBit *gate0_bits, // [2 * qubits_n]
    const CudaBit *gate1_bits, // [2 * qubits_n]
    CudaBit *const gate_bits, // [2 * qubits_n]
    CudaPhs *const phase // []
) {
    for (int i = 0; i < qubits_n; i++) {
        const CudaBit gate0_x = gate0_bits[i];
        const CudaBit gate0_z = gate0_bits[qubits_n + i];
        const CudaBit gate1_x = gate1_bits[i];
        const CudaBit gate1_z = gate1_bits[qubits_n + i];
        CudaBit *const gate_x = gate_bits + (i);
        CudaBit *const gate_z = gate_bits + (qubits_n + i);
        compute_multiply_pauli_phase(gate0_x, gate0_z, gate1_x, gate1_z, phase);
        *gate_x = gate0_x ^ gate1_x;
        *gate_z = gate0_z ^ gate1_z;
    }
}

static __device__
void shot_compute_decomposed_phase(
    const Qid qubits_n,
    const CudaBit *table, // [rows_n, cols_n]
    const CudaBit *decomp_bits, //  [rows_n]
    CudaBit *const decomp_pauli, // [rows_n]
    CudaPhs *const decomp_phase // []
) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    for (Qid row_i = 0; row_i < rows_n; row_i++) {
        if (decomp_bits[row_i]) {
            const CudaBit *row = table + row_i * cols_n;
            compute_multiply_pauli_string(qubits_n, decomp_pauli, row, decomp_pauli, decomp_phase);
            *decomp_phase += static_cast<CudaPhs>(row[qubits_n * 2]) * 2; // add row phase
        }
    }
}

static __global__
void kernel_compute_decomposed_phase(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table, // [shots_n, rows_n, cols_n]
    const CudaBit *decomp_bits, // [shots_n, rows_n]
    CudaBit *const decomp_pauli, // [shots_n, rows_n]
    CudaPhs *const decomp_phase // [shots_n]
) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const unsigned int global_threads_n = shots_n;
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n) return;

    const unsigned int shot_i = global_thread_i;
    const CudaBit *shot_table = table + (shot_i * rows_n * cols_n);
    const CudaBit *shot_decomp_bits = decomp_bits + (shot_i * rows_n);
    CudaBit *const shot_decomp_pauli = decomp_pauli + (shot_i * rows_n);
    CudaPhs *const shot_decomp_phase = decomp_phase + (shot_i);

    shot_compute_decomposed_phase(
        qubits_n, shot_table, shot_decomp_bits, shot_decomp_pauli, shot_decomp_phase);
}

cudaError_t cuda_compute_decomposed_phase(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table,
    const CudaBit *decomp_bits,
    CudaBit *const decomp_pauli,
    CudaPhs *const decomp_phase,
    cudaStream_t stream
) {
    const Qid rows_n = 2 * qubits_n;

    cudaError_t err = cudaSuccess;

    // initialize decomp_pauli
    const size_t decomp_pauli_bytes_n = shots_n * rows_n * sizeof(CudaBit);
    err = cudaMemsetAsync(decomp_pauli, 0, decomp_pauli_bytes_n, stream);
    if (err != cudaSuccess) return err;

    // initialize decomp_phase
    const size_t decomp_phase_bytes_n = shots_n * sizeof(CudaPhs);
    err = cudaMemsetAsync(decomp_phase, 0, decomp_phase_bytes_n, stream);
    if (err != cudaSuccess) return err;

    const unsigned int global_threads_n = shots_n;
    const unsigned int block_threads_n = default_block_threads_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_compute_decomposed_phase<<<blocks_n,block_threads_n,0,stream>>>
        (shots_n, qubits_n, table, decomp_bits, decomp_pauli, decomp_phase);

    return cudaSuccess;
}
