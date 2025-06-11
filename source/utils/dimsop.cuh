#ifndef STN_CUDA_UTILS_DIMSOP_CUH
#define STN_CUDA_UTILS_DIMSOP_CUH

#include "./thread.cuh"

template<typename Item, unsigned int _n>
struct Array {
    Item item;
    Array<Item, _n - 1> tail;

    static __device__ __host__
    unsigned int n() {
        return _n;
    }

    template<unsigned int _i>
    __device__ __host__
    Item get() const {
        if constexpr (_i == 0)
            return item;
        else
            return tail.template get<_i - 1>();
    }

    template<typename... Items>
    static __device__ __host__
    Array of(Item _item, Items... _items) {
        return {_item, Array<Item, _n - 1>::of(_items...)};
    }
};

template<typename Item>
struct Array<Item, 0u> {
    static __device__ __host__
    unsigned int n() {
        return 0;
    }

    static __device__ __host__
    Array of() {
        return {};
    }
};


template<unsigned int dims_n>
using DimsIdx = Array<unsigned int, dims_n>;


static __device__ __host__
unsigned int compute_dims_threads_n() {
    return 1;
}

template<typename... Dims>
static __device__ __host__
unsigned int compute_dims_threads_n(const unsigned int dim, Dims... dims) {
    return dim * compute_dims_threads_n(dims...);
}


static __device__ __host__
DimsIdx<0> compute_dims_idx(const unsigned int) {
    return DimsIdx<0>{};
}

template<typename... Dims>
static __device__ __host__
DimsIdx<1 + sizeof...(Dims)> compute_dims_idx(const unsigned int thread_i, const unsigned int, Dims... dims) {
    const unsigned int sub_threads_n = compute_dims_threads_n(dims...);
    const unsigned int cur_thread_i = thread_i / sub_threads_n;
    const unsigned int sub_thread_i = thread_i % sub_threads_n;
    const auto sub_dims_idx = compute_dims_idx(sub_thread_i, dims...);
    return DimsIdx<1 + sizeof...(Dims)>{cur_thread_i, sub_dims_idx};
}


template<typename Args, unsigned int dims_n, void (*op)(Args args, DimsIdx<dims_n> dims_idx), typename... Dims>
static __global__
void kernel_dims_op(Args args, Dims... dims) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n)return;
    op(args, compute_dims_idx(global_thread_i, dims...));
}

template<typename Args, unsigned int dims_n, void (*op)(Args args, DimsIdx<dims_n> dims_idx), typename... Dims>
static __host__
void cuda_dims_op(cudaStream_t stream, Args args, Dims... dims) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims...);
    const unsigned int block_threads_n = default_block_threads_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_dims_op<Args, dims_n, op, Dims...>
        <<<blocks_n, block_threads_n, 0 ,stream>>>(args, dims...);
}

#endif
