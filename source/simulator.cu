#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./utils/thread.cuh"

using namespace StnCuda;

static __global__
void kernel_initialize_shots_table(
    ShotsStatePtr const shots_state_ptr
) {
    const CudaSid shots_n = shots_state_ptr.shots_n;
    const CudaQid qubits_n = shots_state_ptr.qubits_n;
    const CudaQid rows_n = 2 * qubits_n;

    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n) return;

    const Sid shot_i = global_thread_i / rows_n;
    const Qid row_i = global_thread_i % rows_n;

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_state_ptr(shot_i);
    const TablePtr table_ptr = shot_state_ptr.get_table_ptr();
    const TableRowPtr table_row_ptr = table_ptr.get_row_ptr(row_i);
    const PauliRowPtr pauli_row_ptr = table_row_ptr.get_pauli_ptr();

    *pauli_row_ptr.get_ptr(row_i) = true;
}

static
cudaError_t cuda_initialize_shots_table(
    ShotsStatePtr const shots_state_ptr,
    cudaStream_t const stream
) {
    const CudaSid shots_n = shots_state_ptr.shots_n;
    const CudaQid qubits_n = shots_state_ptr.qubits_n;
    const CudaQid rows_n = 2 * qubits_n;

    const unsigned int global_threads_n = shots_n * rows_n;
    const unsigned int block_threads_n = default_block_threads_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_initialize_shots_table<<<blocks_n,block_threads_n,0,stream>>>(shots_state_ptr);
    return cudaSuccess;
}

cudaError_t Simulator::create(Sid const shots_n, Qid const qubits_n, Aid const map_limit) noexcept {
    cudaError_t err = cudaSuccess;
    do {
        // create stream
        err = cudaStreamCreate(&this->stream);
        if (err != cudaSuccess) break;

        // allocate state
        this->shots_state_ptr.shots_n = shots_n;
        this->shots_state_ptr.qubits_n = qubits_n;
        const size_t state_bytes_n = ShotsStatePtr::compute_bytes_n(shots_n, qubits_n);
        err = cudaMallocAsync(&this->shots_state_ptr.ptr, state_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize state
        err = cudaMemsetAsync(this->shots_state_ptr.ptr, 0, state_bytes_n, this->stream);
        if (err != cudaSuccess) break;
        err = cuda_initialize_shots_table(this->shots_state_ptr, this->stream);
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
    if (this->shots_state_ptr.ptr != nullptr) cudaFree(this->shots_state_ptr.ptr);
    if (this->stream != nullptr) cudaStreamDestroy(this->stream);
    this->shots_state_ptr = {0, 0, nullptr};
    this->stream = nullptr;
    return cudaSuccess;
}
