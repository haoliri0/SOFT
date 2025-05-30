#ifndef STN_CUDA_UTILS_FILLING_CUH
#define STN_CUDA_UTILS_FILLING_CUH

#include "./thread.cuh"

template<typename Value>
static __global__ void kernel_fill(Value value, Value *values, size_t values_n) {
    const size_t global_thread_i = get_global_thread_i();
    if (global_thread_i >= values_n)
        return;
    values[global_thread_i] = value;
}

template<typename Value>
static __host__ void cuda_fill(cudaStream_t stream, Value value, Value *values, size_t values_n) {
    const size_t global_threads_n = values_n;
    const size_t block_threads_n = std::min(global_threads_n, 1024ul);
    const size_t blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_fill<<<blocks_n, block_threads_n, 0, stream>>>(value, values, values_n);
}

#endif
