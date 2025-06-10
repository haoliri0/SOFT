#include "./decompose.hpp"
#include "./simulator.hpp"
#include "./utils/thread.cuh"

using namespace StnCuda;

static __global__
void kernel_compute_gate_z_dest_bits(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table,
    CudaBit *const dest_bits,
    const Qid target
) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n) return;

    const unsigned int shot_i = global_thread_i / rows_n;
    const unsigned int row_i = global_thread_i % rows_n;

    // 在分解 Z 门时，判断某个 (de)stabilizer 是否应该存在
    const CudaBit *row = table + (shot_i * rows_n * cols_n + row_i * cols_n);
    const CudaBit x = row[target];
    const CudaBit z = row[qubits_n + target];
    const CudaBit isX = x && !z;
    const CudaBit isY = x && z;
    const CudaBit isAntiComm = isX || isY;

    // 注意，这里计算结果是反的
    // stabilizer 算出的结果放到 destabilizer bit
    // destabilizer 算出的结果放到 stabilizer bit
    CudaBit *bit = dest_bits + (shot_i * rows_n + (row_i + qubits_n) % rows_n);
    *bit = isAntiComm;
}

void decompose_gate_z(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table,
    CudaBit *const dest_bits,
    const Qid target
) {
    const Qid rows_n = 2 * qubits_n;
    const unsigned int block_threads_n = default_block_threads_n;
    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_compute_gate_z_dest_bits<<<blocks_n, block_threads_n>>>
        (shots_n, qubits_n, table, dest_bits, target);
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
    const CudaBit *gate0_bits, // 2 * qubits_n
    const CudaBit *gate1_bits, // 2 * qubits_n
    CudaBit *const gate_bits, // 2 * qubits_n
    CudaPhs *const phase // 2 bits
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
    const CudaBit *table, // rows_n * cols_n
    const CudaBit *dest_bits, //  rows_n
    CudaBit *const decomp_pauli, // rows_n
    CudaPhs *const decomp_phase // 1
) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    for (Qid row_i = 0; row_i < rows_n; row_i++) {
        if (dest_bits[row_i]) {
            const CudaBit *row = table + row_i * cols_n;
            compute_multiply_pauli_string(qubits_n, decomp_pauli, row, decomp_pauli, decomp_phase);
        }
    }
}

static __global__
void kernel_compute_decomposed_phase(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table, // shots_n * rows_n * cols_n
    const CudaBit *dest_bits, // shots_n * rows_n
    CudaBit *const decomp_pauli, // shots_n * rows_n
    CudaPhs *const decomp_phase // shots_n
) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const unsigned int global_threads_n = shots_n;
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n) return;

    const unsigned int shot_i = global_thread_i;
    const CudaBit *shot_table = table + (shot_i * rows_n * cols_n);
    const CudaBit *shot_dest_bits = dest_bits + (shot_i * rows_n);
    CudaBit *const shot_decomp_pauli = decomp_pauli + (shot_i * rows_n);
    CudaPhs *const shot_decomp_phase = decomp_phase + (shot_i);

    shot_compute_decomposed_phase(
        qubits_n, shot_table, shot_dest_bits, shot_decomp_pauli, shot_decomp_phase);
}

cudaError_t cuda_compute_decomposed_phase(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table,
    const CudaBit *dest_bits,
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
        (shots_n, qubits_n, table, dest_bits, decomp_pauli, decomp_phase);

    return cudaSuccess;
}
