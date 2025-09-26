#ifndef STN_CUDA_SHOTSOP_CUH
#define STN_CUDA_SHOTSOP_CUH

#include "./dimsop.cuh"
#include "./datastruct.cuh"

using namespace StnCuda;

template<typename Args>
struct ArgsShotsOp {
    const ShotsStatePtr shots_state_ptr;
    const Args args;
};

template<typename Args, void (*op)(ShotStatePtr shot_state_ptr, Args args)>
static __device__
void op_shots_op(const ArgsShotsOp<Args> args, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    op(shot_state_ptr, args.args);
}

template<typename Args, void (*op)(ShotStatePtr shot_state_ptr, Args args)>
static __host__
void cuda_shots_op(cudaStream_t const &stream, ShotsStatePtr shots_state_ptr, Args args) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsShotsOp<Args>, 1, op_shots_op<Args, op>>
        (stream, {shots_state_ptr, args}, dimsof(shots_n));
}

struct EmptyArgs {};

template<void (*op)(ShotStatePtr shot_state_ptr)>
static __device__
void op_shots_op(const ArgsShotsOp<EmptyArgs> args, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    op(shot_state_ptr);
}

template<void (*op)(ShotStatePtr shot_state_ptr)>
static __host__
void cuda_shots_op(cudaStream_t const &stream, ShotsStatePtr shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsShotsOp<EmptyArgs>, 1, op_shots_op<op>>
        (stream, {shots_state_ptr, {}}, dimsof(shots_n));
}

#endif
