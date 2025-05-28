#ifndef STN_CUDA_UTILS_EXCEPTION_CUH
#define STN_CUDA_UTILS_EXCEPTION_CUH

#include<exception>
#include<cuda_runtime.h>

class CudaException final : public std::exception {
public:
    const cudaError_t cudaError;

    explicit CudaException(const cudaError_t cudaError) : cudaError(cudaError) {}

    [[nodiscard]]
    const char *what() const noexcept override {
        return cudaGetErrorString(cudaError);
    }
};

static
void cudaCheck(const cudaError_t error) {
    if (error != cudaSuccess) throw CudaException(error);
}

static
void cudaCheck() {
    cudaCheck(cudaPeekAtLastError());
}

#endif
