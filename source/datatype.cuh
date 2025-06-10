#ifndef STN_CUDA_DATATYPE_CUH
#define STN_CUDA_DATATYPE_CUH

#include<complex>
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

}

#endif
