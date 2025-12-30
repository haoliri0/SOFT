#ifndef SOFT_CUDA_THREADS_CUH
#define SOFT_CUDA_THREADS_CUH

#include <cuda_runtime.h>

static __device__
unsigned int get_blocks_n() {
    return gridDim.x;
}

static __device__
unsigned int get_block_i() {
    return blockIdx.x;
}

static __device__
unsigned int get_block_thread_i() {
    return threadIdx.x;
}

static __device__
unsigned int get_block_threads_n() {
    return blockDim.x;
}

static __device__
unsigned int get_global_thread_i() {
    return get_block_threads_n() * get_block_i() + get_block_thread_i();
}

static __device__
unsigned int get_global_threads_n() {
    return get_block_threads_n() * get_blocks_n();
}

template<typename Integer>
static __device__ __host__
Integer ceiling_divide(Integer x, Integer y) {
    return (x + y - 1) / y;
}

constexpr
unsigned int default_block_threads_n = 1024;

#endif
