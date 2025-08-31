#ifndef STN_CUDA_DATATYPE_CUH
#define STN_CUDA_DATATYPE_CUH

#include<complex>
#include<curand_kernel.h>
#include<cuda/std/complex>

namespace StnCuda {

using Sid = unsigned int;
using Qid = unsigned int;
using Eid = unsigned int;
using Mid = unsigned int;

using Int = int;
using Bit = bool;
using Flt = float;
using Amp = cuda::std::complex<Flt>;
using Bst = unsigned long long;
using Phs = unsigned char;

using Err = int;
constexpr Err err_ok = 0;
constexpr Err err_entries_overflow = -1;
constexpr Err err_memory_overflow = -2;

constexpr Qid NullPivot = static_cast<Qid>(-1);

static __device__ __host__
Flt sign_to_flt(const Bit sign) {
    return !sign ? 1 : -1;
}

static __device__ __host__
Amp phase_to_amp(const Phs phase) {
    if (!(phase / 2 % 2)) {
        if (!(phase % 2))
            return Amp{+1, 0}; // 0: +1
        else
            return Amp{0, +1}; // 1: +i
    } else {
        if (!(phase % 2))
            return Amp{-1, 0}; // 2: -1
        else
            return Amp{0, -1}; // 3: -i
    }
}

static __device__ __host__
Bst bits_to_bst(const Bit *bits, const unsigned int n) {
    Bst bst = 0;
    for (unsigned int i = 0; i < n; i++) {
        const Bst bit = !bits[i] ? 0 : 1;
        bst |= bit << i;
    }
    return bst;
}

}

#endif
