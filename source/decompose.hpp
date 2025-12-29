#ifndef SOFT_CUDA_DECOMPOSE_HPP
#define SOFT_CUDA_DECOMPOSE_HPP

#include "./simulator.hpp"

using namespace SoftCuda;

void cuda_compute_decomposed_bits(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr,
    Qid target
);

void cuda_compute_decomposed_phase(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr
);

void cuda_compute_decomp_pivot(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr
);

#endif
