#ifndef STN_CUDA_DECOMPOSE_HPP
#define STN_CUDA_DECOMPOSE_HPP

#include "./simulator.hpp"

using namespace StnCuda;

void decompose_gate_z(
    Qid shots_n,
    Qid qubits_n,
    const CudaBit *table,
    CudaBit *stde_bits,
    Qid target
);

#endif
