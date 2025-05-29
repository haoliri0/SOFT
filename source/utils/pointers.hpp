#ifndef STN_CUDA_UTILS_POINTERS_CUH
#define STN_CUDA_UTILS_POINTERS_CUH

#include <memory>
#include <cuda_runtime.h>
#include "./exception.hpp"

template<typename T>
struct CudaMemDeleter {
    void operator()(T *ptr) const {
        if(ptr != nullptr) cudaCheck(cudaFree(ptr));
    }
};

template<typename T>
using CudaMemUptr = std::unique_ptr<T, CudaMemDeleter<T>>;

template<typename T>
CudaMemUptr<T> cuda_memory_uptr(const size_t size) {
    T *ptr;
    cudaCheck(cudaMalloc(&ptr, size));
    return CudaMemUptr<T>(ptr);
}


struct CudaStreamDeleter {
    void operator()(const cudaStream_t stream) const {
        if(stream != nullptr) cudaCheck(cudaStreamDestroy(stream));
    }
};

using CudaStreamUptr = std::unique_ptr<CUstream_st, CudaStreamDeleter>;

inline CudaStreamUptr cuda_stream_uptr() {
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));
    return CudaStreamUptr(stream);
}

#endif
