#ifndef STN_CUDA_SIMULATOR_HPP
#define STN_CUDA_SIMULATOR_HPP

#include<complex>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda/std/complex>

namespace StnCuda {

using Sid = unsigned int;
using Qid = unsigned int;
using Kid = unsigned int;
using Aid = unsigned int;
using Sti = bool;
using Amp = std::complex<float>;

using CudaSid = unsigned int;
using CudaQid = unsigned int;
using CudaKid = unsigned int;
using CudaAid = unsigned int;
using CudaSti = bool;
using CudaAmp = cuda::std::complex<float>;

struct Simulator {
    Sid shots_n;
    Qid qubits_n;
    Kid map_limit;
    cudaStream_t stream = nullptr;
    CudaSti *table = nullptr; // shape=[shots_n, 2*qubits_n, 2*qubits_n+1], dtype=bool
    CudaKid *map_n = nullptr; // shape=[shots_n], dtype=index
    CudaAid *map_keys = nullptr; // shape=[shots_n, map_limit], dtype=index
    CudaAmp *map_values = nullptr; // shape=[shots_n, map_limit], dtype=complex

    cudaError_t create(Sid shots_n, Qid qubits_n, Aid map_limit) noexcept;

    cudaError_t destroy() noexcept;


    cudaError_t apply_x(int qubit) noexcept;

    // void apply_reset(int qubit);
    // void apply_y(int qubit);
    // void apply_z(int qubit);
    // void apply_h(int qubit);
    // void apply_s(int qubit);
    // void apply_sdg(int qubit);
    // void apply_t(int qubit);
    // void apply_tdg(int qubit);
    // void apply_cnot(int qubit);
};

}

#endif
