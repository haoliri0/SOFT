#ifndef STN_CUDA_DATATYPE_CUH
#define STN_CUDA_DATATYPE_CUH

#include<complex>
#include<cuda/std/complex>
#include<curand_kernel.h>

namespace StnCuda {

using Sid = unsigned int;
using Qid = unsigned int;
using Kid = unsigned int;
using Aid = unsigned int;
using Rid = unsigned int;
using Bit = bool;
using Phs = unsigned char;
using Flt = float;
using Amp = cuda::std::complex<Flt>;

constexpr Qid NullPivot = static_cast<Qid>(-1);

}

#endif
