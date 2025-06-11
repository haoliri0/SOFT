#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./utils/dimsop.cuh"

using namespace StnCuda;

static __device__
void initialize_table_op(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Qid row_i = dims_idx.get<1>();

    Bit *ptr = shots_state_ptr
        .get_shot_state_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i)
        .get_pauli_ptr()
        .get_ptr(row_i);

    *ptr = true;
}

static
void cuda_initialize_shots_table(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const CudaSid shots_n = shots_state_ptr.shots_n;
    const CudaQid qubits_n = shots_state_ptr.qubits_n;
    const CudaQid rows_n = 2 * qubits_n;

    cuda_dims_op<ShotsStatePtr, 2, initialize_table_op>
        (stream, shots_state_ptr, Dims<2>::of(shots_n, rows_n));
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
        cuda_initialize_shots_table(this->stream, this->shots_state_ptr);

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
