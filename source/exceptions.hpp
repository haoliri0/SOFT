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


enum class CliArgsError {
    Success,
    IllegalArg,
    IllegalKey,
    IllegalValue,
};

class CliArgsException final : public std::exception {
public:
    const CliArgsError error;
    explicit CliArgsException(const CliArgsError error) : error(error) {}
};


enum class ExecError {
    Success,
    IOError,
    IllegalOp,
    IllegalArg,
};

class ExecException final : public std::exception {
public:
    const ExecError error;
    explicit ExecException(const ExecError error) : error(error) {}
};

#endif
