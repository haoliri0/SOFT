#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./dimsop.cuh"

using namespace StnCuda;


static __device__
void op_init_table(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Qid const row_i = dims_idx.get<1>();
    Qid const col_i = row_i;
    Bit &bit = *shots_state_ptr
        .get_shot_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i)
        .get_pauli_ptr()
        .get_bit_ptr(col_i);
    bit = true;
}

static __host__
void cuda_init_table(cudaStream_t const stream, const ShotsStatePtr shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid rows_n = 2 * shots_state_ptr.qubits_n;
    cuda_dims_op<ShotsStatePtr, 2, op_init_table>
        (stream, shots_state_ptr, dimsof(shots_n, rows_n));
}


static __device__
void op_init_amps(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const AmpsMapPtr amps_map_ptr = shots_state_ptr
        .get_shot_ptr(shot_i)
        .get_amps_ptr();
    Aid &aid0 = *amps_map_ptr.get_aid_ptr(0);
    Amp &amp0 = *amps_map_ptr.get_amp_ptr(0);
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    aid0 = 0;
    amp0 = 1;
    amps_n = 1;
}

static __host__
void cuda_init_amps(cudaStream_t const stream, const ShotsStatePtr shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_init_amps>
        (stream, shots_state_ptr, dimsof(shots_n));
}


cudaError_t Simulator::create(Sid const shots_n, Qid const qubits_n, Kid const amps_m, Rid const results_m) noexcept {
    cudaError_t err = cudaSuccess;
    do {
        // create stream
        err = cudaStreamCreate(&this->stream);
        if (err != cudaSuccess) break;

        // allocate state
        this->shots_state_ptr = {shots_n, qubits_n, amps_m, results_m};
        const size_t state_bytes_n = this->shots_state_ptr.get_size_bytes_n();
        err = cudaMallocAsync(&this->shots_state_ptr.ptr, state_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize state
        err = cudaMemsetAsync(this->shots_state_ptr.ptr, 0, state_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        cuda_init_table(this->stream, this->shots_state_ptr);
        cuda_init_amps(this->stream, this->shots_state_ptr);

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
    this->shots_state_ptr = {0, 0, 0, 0, nullptr};
    this->stream = nullptr;
    return cudaSuccess;
}
