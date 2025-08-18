#ifndef STN_CUDA_EXCEPTIONS_HPP
#define STN_CUDA_EXCEPTIONS_HPP

#include<cuda_runtime.h>

class CudaException final : public std::exception {
public:
    const cudaError error;
    explicit CudaException(const cudaError error) : error(error) {}
};

static
void cuda_check(const cudaError error) {
    if (error == cudaSuccess) return;
    throw CudaException(error);
}

#endif
