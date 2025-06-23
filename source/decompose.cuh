#ifndef STN_CUDA_DECOMPOSE_CUH
#define STN_CUDA_DECOMPOSE_CUH

#include "./datatype.cuh"

using namespace StnCuda;

static __device__
Aid bits_to_int(const Bit *bits, const unsigned int n) {
    Aid v = 0;
    for (unsigned int i = 0; i < n; i++) {
        const Bit bit = bits[i];
        v <<= 1;
        v |= bit;
    }
    return v;
}

static __device__
Bit compute_sign(Aid key, const Bit *stab_bits, const Qid qubits_n) {
    Bit sign = false;
    for (int qubit_i = 0; qubit_i < qubits_n; ++qubit_i) {
        const Bit key_bit = key % 2;
        const Bit stab_bit = stab_bits[qubit_i];
        sign ^= stab_bit & key_bit;
        key >>= 1;
    }
    return sign;
}

#endif
