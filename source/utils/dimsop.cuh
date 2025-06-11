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
using Dims = Array<unsigned int, dims_n>;

template<unsigned int dims_n>
using DimsIdx = Array<unsigned int, dims_n>;


static __device__ __host__
unsigned int compute_dims_threads_n(const Dims<0>) {
    return 1;
}

template<unsigned int dims_n>
static __device__ __host__
unsigned int compute_dims_threads_n(const Dims<dims_n> dims) {
    return dims.item * compute_dims_threads_n(dims.tail);
}


static __device__ __host__
DimsIdx<0> compute_dims_idx(const unsigned int, const Dims<0>) {
    return DimsIdx<0>{};
}

template<unsigned int dims_n>
static __device__ __host__
DimsIdx<dims_n> compute_dims_idx(const unsigned int thread_i, const Dims<dims_n> dims) {
    const unsigned int sub_threads_n = compute_dims_threads_n(dims.tail);
    const unsigned int cur_thread_i = thread_i / sub_threads_n;
    const unsigned int sub_thread_i = thread_i % sub_threads_n;
    return {cur_thread_i, compute_dims_idx(sub_thread_i, dims.tail)};
}


template<typename Args, unsigned int dims_n, void (*op)(Args args, DimsIdx<dims_n> dims_idx)>
static __global__
void kernel_dims_op(Args args, Dims<dims_n> dims) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims);
    const unsigned int global_thread_i = get_global_thread_i();
    if (global_thread_i >= global_threads_n)return;
    op(args, compute_dims_idx(global_thread_i, dims));
}

template<typename Args, unsigned int dims_n, void (*op)(Args args, DimsIdx<dims_n> dims_idx)>
static __host__
void cuda_dims_op(cudaStream_t stream, Args args, Dims<dims_n> dims) {
    const unsigned int global_threads_n = compute_dims_threads_n(dims);
    const unsigned int block_threads_n = default_block_threads_n;
    const unsigned int blocks_n = ceiling_divide(global_threads_n, block_threads_n);
    kernel_dims_op<Args, dims_n, op>
        <<<blocks_n, block_threads_n, 0 ,stream>>>(args, dims);
}

#endif
