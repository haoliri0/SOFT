#include <cuda_runtime.h>

#include "dataops.cuh"
#include "simulator.hpp"

using namespace StnCuda;


struct ApplyGate1Args {
    Qid target;
};

template<void (*op)(Bit &s, Bit &x, Bit &z)>
static __device__
void op_apply_gate1(const CudaQid, const TableRowPtr ptr, const ApplyGate1Args args) {
    op(*ptr.get_sign_ptr(),
        *ptr.get_pauli_ptr().get_x_ptr(args.target),
        *ptr.get_pauli_ptr().get_z_ptr(args.target));
}

template<void (*op)(Bit &s, Bit &x, Bit &z)>
static __host__
void cuda_apply_gate1_op(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Qid const target
) {
    cuda_shots_table_rows_op<ApplyGate1Args, op_apply_gate1<op>>
        (stream, shots_state_ptr, {target});
}


struct ApplyGate2Args {
    Qid control;
    Qid target;
};

template<void (*op)(Bit &s, Bit &cx, Bit &cz, Bit &tx, Bit &tz)>
static __device__
void op_apply_gate2(const CudaQid, const TableRowPtr ptr, const ApplyGate2Args args) {
    op(*ptr.get_sign_ptr(),
        *ptr.get_pauli_ptr().get_x_ptr(args.control),
        *ptr.get_pauli_ptr().get_z_ptr(args.control),
        *ptr.get_pauli_ptr().get_x_ptr(args.target),
        *ptr.get_pauli_ptr().get_z_ptr(args.target));
}

template<void (*op)(Bit &s, Bit &cx, Bit &cz, Bit &tx, Bit &tz)>
static __host__
void cuda_apply_gate2_op(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Qid const control,
    Qid const target
) {
    cuda_shots_table_rows_op<ApplyGate2Args, op_apply_gate2<op>>
        (stream, shots_state_ptr, {control, target});
}


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
