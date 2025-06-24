#include "./gates.cuh"

using namespace StnCuda;

static __device__ __host__
void op_apply_x(Bit &s, Bit &x, Bit &z) {
    s ^= z;
}

static __device__ __host__
void op_apply_y(Bit &s, Bit &x, Bit &z) {
    s ^= x ^ z;
}

static __device__ __host__
void op_apply_z(Bit &s, Bit &x, Bit &z) {
    s ^= x;
}

static __device__ __host__
void op_apply_h(Bit &s, Bit &x, Bit &z) {
    s ^= x & z;
    cuda::std::swap(x, z);
}

static __device__ __host__
void op_apply_s(Bit &s, Bit &x, Bit &z) {
    s ^= x & z;
    z ^= x;
}

static __device__ __host__
void op_apply_sdg(Bit &s, Bit &x, Bit &z) {
    s ^= x & ~z;
    z ^= x;
}

static __device__ __host__
void op_apply_cx(Bit &s, Bit &cx, Bit &cz, Bit &tx, Bit &tz) {
    s ^= (tx ^ cz ^ true) & tz & cx;
    tx ^= cx;
    cz ^= tz;
}


void Simulator::apply_x(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_x>(stream, shots_state_ptr, target);
}

void Simulator::apply_y(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_y>(stream, shots_state_ptr, target);
}

void Simulator::apply_z(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_z>(stream, shots_state_ptr, target);
}

void Simulator::apply_h(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_h>(stream, shots_state_ptr, target);
}

void Simulator::apply_s(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_s>(stream, shots_state_ptr, target);
}

void Simulator::apply_sdg(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_sdg>(stream, shots_state_ptr, target);
}

void Simulator::apply_cx(const Qid control, const Qid target) const noexcept {
    cuda_apply_gate2_op<op_apply_cx>(stream, shots_state_ptr, control, target);
}
