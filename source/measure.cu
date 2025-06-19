#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./datatype.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;

struct ArgsComputeMeasureCondition {
    ShotsStatePtr shots_state_ptr;
    Qid target;
};

static __device__
void op_compute_measure_condition(const ArgsComputeMeasureCondition args, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = args.shots_state_ptr.get_shot_state_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *decomp_bits = decomp_ptr.get_bits_ptr();
    Bit &measure_condition = *decomp_ptr.get_condition_ptr();

    measure_condition = false;
    Qid const qubits_n = shot_state_ptr.qubits_n;
    for (int row_i = 0; row_i < qubits_n; ++row_i) {
        if (decomp_bits[row_i]) {
            measure_condition = true;
            break;
        }
    }
}

static __host__
void cuda_compute_measure_condition(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    const Qid target
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsComputeMeasureCondition, 1, op_compute_measure_condition>
        (stream, {shots_state_ptr, target}, dimsof(shots_n));
}

void Simulator::measure(const Qid target, Bit *res, Flt *prob) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_measure_condition(stream, shots_state_ptr, target);

    // TODO
}
