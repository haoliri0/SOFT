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

using CudaSid = Sid;
using CudaQid = Qid;
using CudaKid = Kid;
using CudaAid = Aid;
using CudaBit = Bit;
using CudaPhs = Phs;
using CudaAmp = cuda::std::complex<float>;

}

#endif
