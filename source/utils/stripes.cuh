#ifndef STN_CUDA_UTILS_STRIPES_CUH
#define STN_CUDA_UTILS_STRIPES_CUH

#include "./thread.cuh"

struct Stripe {
    unsigned int repeat = 1;
    unsigned int interval = 1;
};

static __device__ __host__
unsigned int stripes_threads_n(const Stripe stripe) {
    return stripe.repeat;
}

template<typename... Stripes>
static __device__ __host__
unsigned int stripes_threads_n(const Stripe stripe, Stripes... stripes) {
    return stripe.repeat * stripes_threads_n(stripes...);
}

static __device__ __host__
unsigned int stripes_offset(const unsigned int thread_i, const Stripe stripe) {
    return thread_i * stripe.interval;
}

template<typename... Stripes>
static __device__ __host__
unsigned int stripes_offset(const unsigned int thread_i, const Stripe stripe, Stripes... stripes) {
    const unsigned int sub_threads_n = stripes_threads_n(stripes...);
    const unsigned int this_thread_i = thread_i / sub_threads_n;
    const unsigned int sub_thread_i = thread_i % sub_threads_n;
    return this_thread_i * stripe.interval + stripes_offset(sub_thread_i, stripes...);
}

template<typename Value, typename... Stripes>
static __global__
void kernel_stripes_fill(Value value, Value *values, Stripes... stripes) {
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= stripes_threads_n(stripes...)) return;
    const unsigned int offset = stripes_offset(global_thread_i, stripes...);
    values[offset] = value;
}

template<typename Value, typename... Stripes>
static __host__
void cuda_stripes_fill(cudaStream_t stream, Value value, Value *values, Stripes... stripes) {
    const unsigned int global_threads_n = stripes_threads_n(stripes...);
    const unsigned int block_threads_n = std::min(global_threads_n, 1024u);
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_stripes_fill<<<blocks_n, block_threads_n, 0 ,stream>>>(value, values, stripes...);
}

#endif
