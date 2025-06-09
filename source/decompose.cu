#include "./decompose.hpp"
#include "./simulator.hpp"
#include "./utils/thread.cuh"

using namespace StnCuda;

static __global__
void kernel_compute_gate_z_stde_bits(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table,
    CudaBit *const stde_bits,
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

    CudaBit *stde_bit = stde_bits + (shot_i * rows_n + row_i);
    *stde_bit = isAntiComm;
}

void decompose_gate_z(
    const Qid shots_n,
    const Qid qubits_n,
    const CudaBit *table,
    CudaBit *const stde_bits,
    const Qid target
) {
    const Qid rows_n = 2 * qubits_n;
    const unsigned int block_threads_n = 1024u;
    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_compute_gate_z_stde_bits<<<blocks_n, block_threads_n>>>
        (shots_n, qubits_n, table, stde_bits, target);
}
