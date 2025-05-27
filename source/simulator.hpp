#ifndef STN_CUDA_SIMULATOR_HPP
#define STN_CUDA_SIMULATOR_HPP

#include<complex>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cuda/std/complex>

namespace StnCuda {

using Sid = uint64_t;
using Qid = uint64_t;
using Aid = uint64_t;
using Sti = bool;
using Amp = std::complex<float>;

using CudaSid = uint64_t;
using CudaQid = uint64_t;
using CudaAid = uint64_t;
using CudaSti = bool;
using CudaAmp = cuda::std::complex<float>;

class Simulator {
public:
    Sid const shots_n;
    Sid const qubits_n;
    Aid const map_limit;
    cudaStream_t const stream;
    CudaSti *const table; // shape=[shots_n, 2*qubits_n, 2*qubits_n+1], dtype=bool
    CudaAid *const map_n; // shape=[shots_n], dtype=index
    CudaAid *const map_keys; // shape=[shots_n, map_limit], dtype=index
    CudaAmp *const map_values; // shape=[shots_n, map_limit], dtype=complex

    Simulator(Sid shots_n, Qid qubits_n, Aid map_limit);

    ~Simulator();

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