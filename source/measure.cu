#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./datatype.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;


static __device__
void op_compute_decomp_pivot(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_state_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *decomp_bits = decomp_ptr.get_bits_ptr();
    Qid &decomp_pivot = *decomp_ptr.get_pivot_ptr();

    decomp_pivot = 0;
    decomp_pivot -= 1;
    Qid const qubits_n = shot_state_ptr.qubits_n;
    for (int row_i = 0; row_i < qubits_n; ++row_i) {
        if (decomp_bits[row_i]) {
            decomp_pivot = row_i;
            break;
        }
    }
}

static __host__
void cuda_compute_decomp_pivot(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_compute_decomp_pivot>
        (stream, shots_state_ptr, dimsof(shots_n));
}

void Simulator::measure(const Qid target, Bit *res, Flt *prob) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    // TODO
}
