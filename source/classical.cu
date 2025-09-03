#include "./simulator.hpp"
#include "./shotsop.cuh"
#include "./gates.cuh"

using namespace StnCuda;

static __device__
void op_classical_flip(const ShotStatePtr shot_state_ptr) {
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Bit src = *work_ptr.get_int_ptr();
    Int &dst = *work_ptr.get_int_ptr();
    dst = !src;
}

static __device__
void op_classical_check(const ShotStatePtr shot_state_ptr, const Err error) {
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Int cond = *work_ptr.get_int_ptr();
    Err &dst = *work_ptr.get_err_ptr();
    if (!dst && cond) dst = error;
}

void Simulator::apply_classical_flip() const noexcept {
    cuda_shots_op<op_classical_flip>(stream, shots_state_ptr);
}

void Simulator::apply_classical_check(const Err error) const noexcept {
    cuda_shots_op<Err, op_classical_check>(stream, shots_state_ptr, error);
}


static __device__
void op_classical_load_int(const ShotStatePtr shot_state_ptr, const Mid pointer) {
    const MemoryPtr memory_ptr = shot_state_ptr.get_memory_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    if (pointer >= memory_ptr.mem_ints_m) {
        Err &err = *work_ptr.get_err_ptr();
        err = err_memory_overflow;
        return;
    }

    const Int src = *memory_ptr.get_int_ptr(pointer);
    Int &dst = *work_ptr.get_int_ptr();
    dst = src;
}

static __device__
void op_classical_load_flt(const ShotStatePtr shot_state_ptr, const Mid pointer) {
    const MemoryPtr memory_ptr = shot_state_ptr.get_memory_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    if (pointer >= memory_ptr.mem_flts_m) {
        Err &err = *work_ptr.get_err_ptr();
        err = err_memory_overflow;
        return;
    }

    const Flt src = *memory_ptr.get_flt_ptr(pointer);
    Flt &dst = *work_ptr.get_flt_ptr();
    dst = src;
}

static __device__
void op_classical_save_int(const ShotStatePtr shot_state_ptr, const Mid pointer) {
    const MemoryPtr memory_ptr = shot_state_ptr.get_memory_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    if (pointer >= memory_ptr.mem_ints_m) {
        Err &err = *work_ptr.get_err_ptr();
        err = err_memory_overflow;
        return;
    }

    const Int src = *work_ptr.get_int_ptr();
    Int &dst = *memory_ptr.get_int_ptr(pointer);
    dst = src;
}

static __device__
void op_classical_save_flt(const ShotStatePtr shot_state_ptr, const Mid pointer) {
    const MemoryPtr memory_ptr = shot_state_ptr.get_memory_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    if (pointer >= memory_ptr.mem_flts_m) {
        Err &err = *work_ptr.get_err_ptr();
        err = err_memory_overflow;
        return;
    }

    const Flt src = *work_ptr.get_flt_ptr();
    Flt &dst = *memory_ptr.get_flt_ptr(pointer);
    dst = src;
}

void Simulator::apply_classical_load_int(const Mid pointer) const noexcept {
    cuda_shots_op<Mid, op_classical_load_int>(stream, shots_state_ptr, pointer);
}

void Simulator::apply_classical_load_flt(const Mid pointer) const noexcept {
    cuda_shots_op<Mid, op_classical_load_flt>(stream, shots_state_ptr, pointer);
}

void Simulator::apply_classical_save_int(const Mid pointer) const noexcept {
    cuda_shots_op<Mid, op_classical_save_int>(stream, shots_state_ptr, pointer);
}

void Simulator::apply_classical_save_flt(const Mid pointer) const noexcept {
    cuda_shots_op<Mid, op_classical_save_flt>(stream, shots_state_ptr, pointer);
}


using ArgsClassicalReduce = ClassicalReduceArgs<>;

template<Int (*reduction)(Int, Int), Int value0 = 0>
static __device__
void op_classical_reduce_int(const ShotStatePtr shot_state_ptr, const ArgsClassicalReduce args) {
    Int value = value0;
    const MemoryPtr memory_ptr = shot_state_ptr.get_memory_ptr();
    for (Mid i = 0; i < args.n; ++i) {
        const Mid pointer = args.pointers.get(i);
        const Int value1 = *memory_ptr.get_int_ptr(pointer);
        value = reduction(value, value1);
    }

    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    Int &dst = *work_ptr.get_int_ptr();
    dst = value;
}

static __device__ __host__
Int reduction_logical_or(const Int value0, const Int value1) {
    const Bit bit0 = value0;
    const Bit bit1 = value1;
    const Bit bit = bit0 || bit1;
    const Int value = bit;
    return value;
}

static __device__ __host__
Int reduction_logical_xor(const Int value0, const Int value1) {
    const Bit bit0 = value0;
    const Bit bit1 = value1;
    const Bit bit = bit0 ^ bit1;
    const Int value = bit;
    return value;
}

static __device__ __host__
Int reduction_logical_and(const Int value0, const Int value1) {
    const Bit bit0 = value0;
    const Bit bit1 = value1;
    const Bit bit = bit0 && bit1;
    const Int value = bit;
    return value;
}

void Simulator::apply_classical_or(const ArgsClassicalReduce args) const noexcept {
    cuda_shots_op<ArgsClassicalReduce, op_classical_reduce_int<reduction_logical_or>>
        (stream, shots_state_ptr, args);
}

void Simulator::apply_classical_xor(const ArgsClassicalReduce args) const noexcept {
    cuda_shots_op<ArgsClassicalReduce, op_classical_reduce_int<reduction_logical_xor>>
        (stream, shots_state_ptr, args);
}

void Simulator::apply_classical_and(const ArgsClassicalReduce args) const noexcept {
    cuda_shots_op<ArgsClassicalReduce, op_classical_reduce_int<reduction_logical_and>>
        (stream, shots_state_ptr, args);
}


using ArgsClassicalLut = ClassicalLutArgs<>;

static __device__
void op_classical_lut(const ShotStatePtr shot_state_ptr, const ArgsClassicalLut args) {
    unsigned int index = 0;
    const MemoryPtr memory_ptr = shot_state_ptr.get_memory_ptr();
    for (unsigned int i = 0; i < args.n; i++) {
        const Mid pointer = args.pointers.get(i);
        const Bit bit = *memory_ptr.get_int_ptr(pointer);
        if (bit) index |= 1 << i;
    }

    const Int src = args.table.get(index);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    Int &dst = *work_ptr.get_int_ptr();
    dst = src;
}

void Simulator::apply_classical_lut(const ArgsClassicalLut args) const noexcept {
    cuda_shots_op<ArgsClassicalLut, op_classical_lut>(stream, shots_state_ptr, args);
}


struct ArgsClassicalControlledGate1 {
    ShotsStatePtr shots_state_ptr;
    Qid target;
};

template<void (*op)(Bit &s, Bit &x, Bit &z)>
static __device__
void op_classical_controlled_gate1(const ArgsClassicalControlledGate1 args, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Bit cond = *work_ptr.get_int_ptr();
    if (cond) op_apply_gate1<op>({shots_state_ptr, args.target}, dims_idx);
}

template<void (*op)(Bit &s, Bit &x, Bit &z)>
static __host__
void cuda_classical_controlled_gate1(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    const Qid target
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsClassicalControlledGate1, 2, op_classical_controlled_gate1<op>>
        (stream, {shots_state_ptr, target}, dimsof(shots_n, rows_n));
}

void Simulator::apply_classical_controlled_x(const Qid target) const noexcept {
    cuda_classical_controlled_gate1<op_apply_x>(stream, shots_state_ptr, target);
}

void Simulator::apply_classical_controlled_y(const Qid target) const noexcept {
    cuda_classical_controlled_gate1<op_apply_y>(stream, shots_state_ptr, target);
}

void Simulator::apply_classical_controlled_z(const Qid target) const noexcept {
    cuda_classical_controlled_gate1<op_apply_z>(stream, shots_state_ptr, target);
}
