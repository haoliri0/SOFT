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
    const Sid shot_i;
};

static __device__
void op_init_rand(const ArgsInitRand args, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    curandState *rand_state_ptr = args.shots_state_ptr
        .get_shot_ptr(shot_i).get_work_ptr().get_rand_state_ptr();
    curand_init(args.seed, args.shot_i + shot_i, 0, rand_state_ptr);
}

static __host__
void cuda_init_rand(
    cudaStream_t const stream,
    const ShotsStatePtr shots_state_ptr,
    const unsigned long long seed,
    const Sid shot_i
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsInitRand, 1, op_init_rand>
        (stream, {shots_state_ptr, seed, shot_i}, dimsof(shots_n));
}


cudaError_t Simulator::create(SimulatorArgs const &args) noexcept {
    epsilon = args.epsilon;
    cudaError_t err = cudaSuccess;
    do {
        // create stream
        err = cudaStreamCreate(&stream);
        if (err != cudaSuccess) break;

        // allocate state
        shots_state_ptr = {args.shots_n, args.qubits_n, args.entries_m, args.mem_ints_m, args.mem_flts_m};
        const size_t state_bytes_n = shots_state_ptr.get_size_bytes_n();
        err = cudaMallocAsync(&shots_state_ptr.ptr, state_bytes_n, stream);
        if (err != cudaSuccess) break;

        // initialize state
        err = cudaMemsetAsync(shots_state_ptr.ptr, 0, state_bytes_n, stream);
        if (err != cudaSuccess) break;

        cuda_init_table(stream, shots_state_ptr);
        cuda_init_entries(stream, shots_state_ptr);
        cuda_init_rand(stream, shots_state_ptr, args.seed, args.shot_i);

        // wait for async operations to complete
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) break;
    } while (false);

    if (err != cudaSuccess)
        destroy();

    return err;
}

cudaError_t Simulator::destroy() noexcept {
    if (shots_state_ptr.ptr != nullptr) cudaFree(shots_state_ptr.ptr);
    if (stream != nullptr) cudaStreamDestroy(stream);
    shots_state_ptr = {0, 0, 0, 0, 0, nullptr};
    stream = nullptr;
    return cudaSuccess;
}
