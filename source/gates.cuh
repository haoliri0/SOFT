#ifndef STN_CUDA_GATES_CUH
#define STN_CUDA_GATES_CUH

#include "./simulator.hpp"
#include "./dimsop.cuh"

using namespace StnCuda;

struct ArgsApplyGate1 {
    const ShotsStatePtr shots_state_ptr;
    const Qid target;
};

template<void (*op)(Bit &s, Bit &x, Bit &z)>
static __device__
void op_apply_gate1(const ArgsApplyGate1 args, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Qid const row_i = dims_idx.get<1>();
    TableRowPtr const ptr = args.shots_state_ptr
        .get_shot_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i);
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
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsApplyGate1, 2, op_apply_gate1<op>>
        (stream, {shots_state_ptr, target}, dimsof(shots_n, rows_n));
}


struct ArgsApplyGate2 {
    const ShotsStatePtr shots_state_ptr;
    const Qid control;
    const Qid target;
};

template<void (*op)(Bit &s, Bit &cx, Bit &cz, Bit &tx, Bit &tz)>
static __device__
void op_apply_gate2(const ArgsApplyGate2 args, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Qid const row_i = dims_idx.get<1>();
    TableRowPtr const ptr = args.shots_state_ptr
        .get_shot_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i);
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
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsApplyGate2, 2, op_apply_gate2<op>>
        (stream, {shots_state_ptr, control, target}, dimsof(shots_n, rows_n));
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

#endif
