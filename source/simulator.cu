#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./dataops.cuh"

using namespace StnCuda;

struct InitTableArgs {};

static __device__
void op_init_table(const CudaQid row_i, const TableRowPtr table_row_ptr, InitTableArgs) {
    Qid const col_i = row_i;
    Bit *const ptr = table_row_ptr.get_pauli_ptr().get_ptr(col_i);
    *ptr = true;
}

static __host__
void cuda_init_table(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    cuda_shots_table_rows_op<InitTableArgs, op_init_table>
        (stream, shots_state_ptr, InitTableArgs{});
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
        cuda_init_table(this->stream, this->shots_state_ptr);

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
