#ifndef STN_CUDA_UTILS_STRIPES_CUH
#define STN_CUDA_UTILS_STRIPES_CUH

#include "./utils_thread.cuh"

struct Stripe {
    size_t repeat = 1;
    size_t interval = 1;
};

static __device__ __host__
size_t stripes_threads_n(const Stripe stripe) {
    return stripe.repeat;
}

template<typename... Stripes>
static __device__ __host__
size_t stripes_threads_n(const Stripe stripe, Stripes... stripes) {
    return stripe.repeat * stripes_threads_n(stripes...);
}

static __device__ __host__
size_t stripes_offset(const size_t thread_i, const Stripe stripe) {
    return thread_i * stripe.interval;
}

template<typename... Stripes>
static __device__ __host__
size_t stripes_offset(const size_t thread_i, const Stripe stripe, Stripes... stripes) {
    const size_t sub_threads_n = stripes_threads_n(stripes...);
    const size_t this_thread_i = thread_i / sub_threads_n;
    const size_t sub_thread_i = thread_i % sub_threads_n;
    return this_thread_i * stripe.interval + stripes_offset(sub_thread_i, stripes...);
}

template<typename Value, typename... Stripes>
static __global__
void kernel_stripes_set(Value value, Value *values, Stripes... stripes) {
    const size_t global_thread_i = get_global_thread_i();
    if (global_thread_i >= stripes_threads_n(stripes...)) return;
    const size_t offset = stripes_offset(global_thread_i, stripes...);
    values[offset] = value;
}

template<typename Value, typename... Stripes>
static __host__
void cuda_stripes_set(Value value, Value *values, Stripes... stripes) {
    const size_t global_threads_n = stripes_threads_n(stripes...);
    const size_t block_threads_n = std::min(global_threads_n, 1024ul);
    const size_t blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_stripes_set<<<blocks_n, block_threads_n>>>(value, values, stripes...);
}

#endif
