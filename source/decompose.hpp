#ifndef STN_CUDA_DECOMPOSE_HPP
#define STN_CUDA_DECOMPOSE_HPP

#include "./simulator.hpp"

using namespace StnCuda;

void decompose_gate_z(
    Qid shots_n,
    Qid qubits_n,
    const CudaBit *table,
    CudaBit *decomp_bits,
    Qid target
);

cudaError_t cuda_compute_decomposed_phase(
    Qid shots_n,
    Qid qubits_n,
    const CudaBit *table,
    const CudaBit *decomp_bits,
    CudaBit *decomp_pauli,
    CudaPhs *decomp_phase,
    cudaStream_t stream
);

#endif
