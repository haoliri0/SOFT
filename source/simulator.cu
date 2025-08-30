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
void op_init_entries(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    Bst &bst0 = *entries_ptr.get_bst_ptr(0);
    Amp &amp0 = *entries_ptr.get_amp_ptr(0);
    Eid &entries_n = *entries_ptr.get_entries_n_ptr();
    bst0 = 0;
    amp0 = 1;
    entries_n = 1;
}

static __host__
void cuda_init_entries(cudaStream_t const stream, const ShotsStatePtr shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_init_entries>
        (stream, shots_state_ptr, dimsof(shots_n));
}


struct ArgsInitRand {
    const ShotsStatePtr shots_state_ptr;
    const unsigned long long seed;
};

static __device__
void op_init_rand(const ArgsInitRand args, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    curandState *rand_state_ptr = args.shots_state_ptr
        .get_shot_ptr(shot_i).get_work_ptr().get_rand_state_ptr();
    curand_init(args.seed, shot_i, 0, rand_state_ptr);
}

static __host__
void cuda_init_rand(cudaStream_t const stream, const ShotsStatePtr shots_state_ptr, const unsigned long long seed) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsInitRand, 1, op_init_rand>
        (stream, {shots_state_ptr, seed}, dimsof(shots_n));
}


cudaError_t Simulator::create(
    Sid const shots_n,
    Qid const qubits_n,
    Eid const entries_m,
    Mid const mem_ints_m,
    Mid const mem_flts_m,
    unsigned long long const seed
) noexcept {
    cudaError_t err = cudaSuccess;
    do {
        // create stream
        err = cudaStreamCreate(&this->stream);
        if (err != cudaSuccess) break;

        // allocate state
        this->shots_state_ptr = {shots_n, qubits_n, entries_m, mem_ints_m, mem_flts_m};
        const size_t state_bytes_n = this->shots_state_ptr.get_size_bytes_n();
        err = cudaMallocAsync(&this->shots_state_ptr.ptr, state_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize state
        err = cudaMemsetAsync(this->shots_state_ptr.ptr, 0, state_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        cuda_init_table(this->stream, this->shots_state_ptr);
        cuda_init_entries(this->stream, this->shots_state_ptr);
        cuda_init_rand(this->stream, this->shots_state_ptr, seed);

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
    this->shots_state_ptr = {0, 0, 0, 0, 0, nullptr};
    this->stream = nullptr;
    return cudaSuccess;
}
