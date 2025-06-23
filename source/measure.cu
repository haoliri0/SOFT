#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./datatype.cuh"

using namespace StnCuda;


void Simulator::measure(const Qid target, Bit *res, Flt *prob) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    // TODO
}
