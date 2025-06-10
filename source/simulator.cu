#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./utils/dimsop.cuh"

using namespace StnCuda;

static __global__
void kernel_initialize_table(
    CudaBit *const table,
    const CudaSid shots_n,
    const CudaQid qubits_n
) {
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n) return;

    const unsigned int shot_i = global_thread_i / rows_n;
    const unsigned int row_i = global_thread_i % rows_n;
    CudaBit *row = table + (shot_i * rows_n * cols_n + row_i * cols_n);

    row[row_i] = true;
}

static
cudaError_t cuda_initialize_table(
    cudaStream_t stream,
    CudaBit *const table,
    const CudaSid shots_n,
    const CudaQid qubits_n
) {
    const Qid rows_n = 2 * qubits_n;
    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int block_threads_n = default_block_threads_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_initialize_table<<<blocks_n,block_threads_n,0,stream>>>
        (table, shots_n, qubits_n);
    return cudaSuccess;
}

cudaError_t Simulator::create(Sid const shots_n, Qid const qubits_n, Aid const map_limit) noexcept {
    const CudaQid rows_n = 2 * qubits_n;
    const CudaQid cols_n = 2 * qubits_n + 1;

    this->shots_n = shots_n;
    this->qubits_n = qubits_n;
    this->map_limit = map_limit;

    cudaError_t err = cudaSuccess;
    do {
        // create stream
        err = cudaStreamCreate(&this->stream);
        if (err != cudaSuccess) break;

        // allocate table
        const size_t table_bytes_n = shots_n * rows_n * cols_n * sizeof(CudaBit);
        err = cudaMallocAsync(&this->table, table_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate mpa_n
        const size_t map_n_bytes_n = shots_n * sizeof(CudaKid);
        err = cudaMallocAsync(&this->map_n, map_n_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate map_keys
        const size_t map_keys_bytes_n = shots_n * map_limit * sizeof(CudaAid);
        err = cudaMallocAsync(&this->map_keys, map_keys_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate map_values
        const size_t map_values_bytes_n = shots_n * map_limit * sizeof(CudaAmp);
        err = cudaMallocAsync(&this->map_values, map_values_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate dest_bits
        const size_t dest_bits_bytes_n = shots_n * rows_n * sizeof(CudaBit);
        err = cudaMallocAsync(&this->dest_bits, dest_bits_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate dest_pauli
        const size_t decomp_pauli_bytes_n = shots_n * rows_n * sizeof(CudaBit);
        err = cudaMallocAsync(&this->decomp_pauli, decomp_pauli_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate decomp_phase
        const size_t decomp_phase_bytes_n = shots_n * sizeof(CudaPhs);
        err = cudaMallocAsync(&this->decomp_phase, decomp_phase_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize table
        err = cudaMemsetAsync(this->table, 0, table_bytes_n, this->stream);
        if (err != cudaSuccess) break;
        err = cuda_initialize_table(this->stream, this->table, this->shots_n, this->qubits_n);
        if (err != cudaSuccess) break;

        // initialize map_n
        constexpr CudaKid kid_one = 1;
        cuda_dims_fill(this->stream, this->map_n, kid_one, Dim{shots_n});

        // initialize map_keys
        constexpr CudaAid aid_zero = 0;
        cuda_dims_fill(this->stream, this->map_keys, aid_zero, Dim{shots_n}, Dim{map_limit, 0, 1});

        // initialize map_values
        constexpr CudaAmp amp_one = 1;
        cuda_dims_fill(this->stream, this->map_values, amp_one, Dim{shots_n}, Dim{map_limit, 0, 1});

        // initialize dest_bits
        err = cudaMemsetAsync(this->dest_bits, 0, dest_bits_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize decomp_pauli
        err = cudaMemsetAsync(this->decomp_pauli, 0, decomp_pauli_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize decomp_phase
        err = cudaMemsetAsync(this->decomp_phase, 0, decomp_phase_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // wait for async operations to complete
        err = cudaStreamSynchronize(this->stream);
        if (err != cudaSuccess) break;
    } while (false);

    if (err != cudaSuccess)
        this->destroy();

    return err;
}

cudaError_t Simulator::destroy() noexcept {
    if (this->table != nullptr) cudaFree(this->table);
    if (this->map_n != nullptr) cudaFree(this->map_n);
    if (this->map_keys != nullptr) cudaFree(this->map_keys);
    if (this->map_values != nullptr) cudaFree(this->map_values);
    if (this->stream != nullptr) cudaStreamDestroy(this->stream);
    this->table = nullptr;
    this->map_n = nullptr;
    this->map_keys = nullptr;
    this->map_values = nullptr;
    this->stream = nullptr;
    return cudaSuccess;
}
