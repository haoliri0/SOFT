#ifndef STN_CUDA_UTILS_DIMSOP_CUH
#define STN_CUDA_UTILS_DIMSOP_CUH

#include "./thread.cuh"

struct Dim {
    unsigned int size;
    unsigned int head = 0;
    unsigned int tail = size;
    unsigned int stride = 1;
};


static __device__ __host__
unsigned int compute_dims_items_n() {
    return 1;
}

template<typename... Dims>
static __device__ __host__
unsigned int compute_dims_items_n(const Dim dim, Dims... dims) {
    return dim.size * compute_dims_items_n(dims...);
}


static __device__ __host__
unsigned int compute_dims_threads_n() {
    return 1;
}

static __device__ __host__
unsigned int compute_dim_threads_n(const Dim dim) {
    return (dim.stride - 1 + dim.tail - dim.head) / dim.stride;
}

template<typename... Dims>
static __device__ __host__
unsigned int compute_dims_threads_n(const Dim dim, Dims... dims) {
    return compute_dim_threads_n(dim) * compute_dims_threads_n(dims...);
}


static __device__ __host__
unsigned int compute_dims_offset(const unsigned int) {
    return 0;
}

template<typename... Dims>
static __device__ __host__
unsigned int compute_dims_offset(const unsigned int thread_i, const Dim dim, Dims... dims) {
    const unsigned int sub_items_n = compute_dims_items_n(dims...);
    const unsigned int sub_threads_n = compute_dims_threads_n(dims...);
    const unsigned int cur_thread_i = thread_i / sub_threads_n;
    const unsigned int sub_thread_i = thread_i % sub_threads_n;
    const unsigned int cur_offset = (dim.head + cur_thread_i * dim.stride) * sub_items_n;
    const unsigned int sub_offset = compute_dims_offset(sub_thread_i, dims...);
    return cur_offset + sub_offset;
}


template<typename Value, void (*op)(Value *v0, Value *v1), typename... Dims>
static __global__
void kernel_dims_op2(
    Value *values,
    const unsigned int p0,
    const unsigned int p1,
    Dims... dims
) {
    const unsigned int global_thread_i = get_global_thread_i();
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    if (global_thread_i >= global_threads_n) return;
    const unsigned int offset = compute_dims_offset(global_thread_i, dims...);
    op(values + offset + p0, values + offset + p1);
}

template<typename Value, void (*op)(Value *v0, Value *v1), typename... Dims>
static __host__
void cuda_dims_op2(
    cudaStream_t stream,
    Value *values,
    const unsigned int p0,
    const unsigned int p1,
    Dims... dims
) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    const unsigned int block_threads_n = std::min(global_threads_n, 1024u);
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_dims_op2<Value, op, Dims...>
        <<<blocks_n, block_threads_n, 0 ,stream>>>(values, p0, p1, dims...);
}


template<typename Value, void (*op)(Value *v0, Value *v1, Value *v2), typename... Dims>
static __global__
void kernel_dims_op3(
    Value *values,
    const unsigned int p0,
    const unsigned int p1,
    const unsigned int p2,
    Dims... dims
) {
    const unsigned int global_thread_i = get_global_thread_i();
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    if (global_thread_i >= global_threads_n) return;
    const unsigned int offset = compute_dims_offset(global_thread_i, dims...);
    op(values + offset + p0, values + offset + p1, values + offset + p2);
}

template<typename Value, void (*op)(Value *v0, Value *v1, Value *v2), typename... Dims>
static __host__
void cuda_dims_op3(
    cudaStream_t stream,
    Value *values,
    const unsigned int p0,
    const unsigned int p1,
    const unsigned int p2,
    Dims... dims
) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    const unsigned int block_threads_n = std::min(global_threads_n, 1024u);
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_dims_op3<Value, op, Dims...>
        <<<blocks_n, block_threads_n, 0 ,stream>>>(values, p0, p1, p2, dims...);
}


template<typename Value, typename... Dims>
static __global__
void kernel_dims_fill(Value *values, Value value, Dims... dims) {
    const unsigned int global_thread_i = get_global_thread_i();
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    if (global_thread_i >= global_threads_n) return;
    const unsigned int offset = compute_dims_offset(global_thread_i, dims...);
    values[offset] = value;
}

template<typename Value, typename... Dims>
static __host__
void cuda_dims_fill(cudaStream_t stream, Value *values, Value value, Dims... dims) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    const unsigned int block_threads_n = std::min(global_threads_n, 1024u);
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_dims_fill<<<blocks_n, block_threads_n, 0, stream>>>(values, value, dims...);
}


template<typename Value>
static __device__ __host__
void op_xor(Value *v0, Value *v1) {
    *v1 ^= *v0;
}


template<typename Value, typename... Dims>
static __global__
void kernel_dims_xor_xor(
    Value *values,
    const unsigned int src0,
    const unsigned int src1,
    const unsigned int dst,
    Dims... dims
) {
    const unsigned int global_thread_i = get_global_thread_i();
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    if (global_thread_i >= global_threads_n) return;
    const unsigned int offset = compute_dims_offset(global_thread_i, dims...);
    values[offset + dst] ^= values[offset + src0] ^ values[offset + src1];
}

template<typename Value, typename... Dims>
static __host__
void cuda_dims_xor_xor(
    cudaStream_t stream,
    Value *values,
    const unsigned int src0,
    const unsigned int src1,
    const unsigned int dst,
    Dims... dims
) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    const unsigned int block_threads_n = std::min(global_threads_n, 1024u);
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_dims_xor_xor<<<blocks_n, block_threads_n, 0 ,stream>>>(values, src0, src1, dst, dims...);
}

#endif
