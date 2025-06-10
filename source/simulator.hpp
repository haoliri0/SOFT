#ifndef STN_CUDA_SIMULATOR_HPP
#define STN_CUDA_SIMULATOR_HPP

#include<complex>
#include<cuda_runtime.h>
#include<cuda/std/complex>

namespace StnCuda {

using Sid = unsigned int;
using Qid = unsigned int;
using Kid = unsigned int;
using Aid = unsigned int;
using Bit = bool;
using Phs = unsigned char;
using Amp = std::complex<float>;

using CudaSid = unsigned int;
using CudaQid = unsigned int;
using CudaKid = unsigned int;
using CudaAid = unsigned int;
using CudaBit = bool;
using CudaPhs = unsigned char;
using CudaAmp = cuda::std::complex<float>;

struct Simulator {
    Sid shots_n;
    Qid qubits_n;
    Kid map_limit;
    cudaStream_t stream = nullptr;
    CudaBit *table = nullptr; // shape=[shots_n, 2*qubits_n, 2*qubits_n+1], dtype=bool
    CudaKid *map_n = nullptr; // shape=[shots_n], dtype=index
    CudaAid *map_keys = nullptr; // shape=[shots_n, map_limit], dtype=index
    CudaAmp *map_values = nullptr; // shape=[shots_n, map_limit], dtype=complex

    // work memory for decomposition: destabilizer, stabilizer bits
    // shape=[shots_n, 2*qubits_n], dtype=bool
    CudaBit *decomp_bits = nullptr;

    // work memory for decomposition: pauli operators
    // shape=[shots_n, 2*qubits_n], dtype=bit
    CudaBit *decomp_pauli = nullptr;

    // work memory for decomposition: phase
    // shape=[shots_n], dtype=byte (only use 2 bits)
    CudaPhs *decomp_phase = nullptr;


    cudaError_t create(Sid shots_n, Qid qubits_n, Aid map_limit) noexcept;

    cudaError_t destroy() noexcept;


    cudaError_t apply_x(Qid target) const noexcept;

    cudaError_t apply_y(Qid target) const noexcept;

    cudaError_t apply_z(Qid target) const noexcept;

    cudaError_t apply_h(Qid target) const noexcept;

    cudaError_t apply_s(Qid target) const noexcept;

    cudaError_t apply_sdg(Qid target) const noexcept;

    cudaError_t apply_cx(Qid control, Qid target) const noexcept;

    // void apply_reset(int qubit);
    // void apply_t(int qubit);
    // void apply_tdg(int qubit);
    // void apply_cnot(int qubit);
};

}

#endif
