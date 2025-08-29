#include "./simulator.hpp"
#include "./shotsop.cuh"
#include "./gates.cuh"

using namespace StnCuda;

static __device__
void op_classical_not(const ShotStatePtr shot_state_ptr) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Bit src = *results_ptr.get_work_value_ptr();
    Rvl &dst = *results_ptr.get_work_value_ptr();
    dst = ~src;
}

static __device__
void op_classical_set(const ShotStatePtr shot_state_ptr, const Rvl value) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    Rvl &dst = *results_ptr.get_work_value_ptr();
    dst = value;
}

static __device__
void op_classical_read(const ShotStatePtr shot_state_ptr, const Rid pointer) {
    if (pointer >= shot_state_ptr.results_m) {
        Err &err = *shot_state_ptr.get_error_ptr();
        err = err_results_overflow;
        return;
    }

    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rvl src = *results_ptr.get_value_ptr(pointer);
    Rvl &dst = *results_ptr.get_work_value_ptr();
    dst = src;
}

static __device__
void op_classical_write(const ShotStatePtr shot_state_ptr, const Rid pointer) {
    if (pointer >= shot_state_ptr.results_m) {
        Err &err = *shot_state_ptr.get_error_ptr();
        err = err_results_overflow;
        return;
    }

    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rvl src = *results_ptr.get_work_value_ptr();
    Rvl &dst = *results_ptr.get_value_ptr(pointer);
    dst = src;
}

static __device__
void op_classical_check(const ShotStatePtr shot_state_ptr) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rvl val = *results_ptr.get_work_value_ptr();
    Err &err = *shot_state_ptr.get_error_ptr();
    if (!err && val) err = val;
}

void Simulator::apply_classical_not() const noexcept {
    cuda_shots_op<op_classical_not>(stream, shots_state_ptr);
}

void Simulator::apply_classical_set(const Rvl value) const noexcept {
    cuda_shots_op<Rvl, op_classical_set>(stream, shots_state_ptr, value);
}

void Simulator::apply_classical_read(const Rid pointer) const noexcept {
    cuda_shots_op<Rid, op_classical_read>(stream, shots_state_ptr, pointer);
}

void Simulator::apply_classical_write(const Rid pointer) const noexcept {
    cuda_shots_op<Rid, op_classical_write>(stream, shots_state_ptr, pointer);
}

void Simulator::apply_classical_check() const noexcept {
    cuda_shots_op<op_classical_check>(stream, shots_state_ptr);
}


using ArgsClassicalReduce = ClassicalReduceArgs<>;

template<Rvl (*reduction)(Rvl, Rvl), Rvl value0 = 0>
static __device__
void op_classical_reduce(const ShotStatePtr shot_state_ptr, const ArgsClassicalReduce args) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    Rvl &value = *results_ptr.get_work_value_ptr();

    value = value0;
    for (Rid i = 0; i < args.n; ++i) {
        const Rid pointer = args.pointers.get(i);
        const Rvl value1 = *results_ptr.get_value_ptr(pointer);
        value = reduction(value, value1);
    }
}

static __device__ __host__
Rvl reduction_logical_or(const Rvl value0, const Rvl value1) {
    const Bit bit0 = value0;
    const Bit bit1 = value1;
    const Bit bit = bit0 || bit1;
    const Rvl value = bit;
    return value;
}

static __device__ __host__
Rvl reduction_logical_xor(const Rvl value0, const Rvl value1) {
    const Bit bit0 = value0;
    const Bit bit1 = value1;
    const Bit bit = bit0 ^ bit1;
    const Rvl value = bit;
    return value;
}

static __device__ __host__
Rvl reduction_logical_and(const Rvl value0, const Rvl value1) {
    const Bit bit0 = value0;
    const Bit bit1 = value1;
    const Bit bit = bit0 && bit1;
    const Rvl value = bit;
    return value;
}

void Simulator::apply_classical_or(const ArgsClassicalReduce args) const noexcept {
    cuda_shots_op<ArgsClassicalReduce, op_classical_reduce<reduction_logical_or>>
        (stream, shots_state_ptr, args);
}

void Simulator::apply_classical_xor(const ArgsClassicalReduce args) const noexcept {
    cuda_shots_op<ArgsClassicalReduce, op_classical_reduce<reduction_logical_xor>>
        (stream, shots_state_ptr, args);
}

void Simulator::apply_classical_and(const ArgsClassicalReduce args) const noexcept {
    cuda_shots_op<ArgsClassicalReduce, op_classical_reduce<reduction_logical_and>>
        (stream, shots_state_ptr, args);
}


using ArgsClassicalLut = ClassicalLutArgs<>;

static __device__
void op_classical_lut(const ShotStatePtr shot_state_ptr, const ArgsClassicalLut args) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();

    unsigned int index = 0;
    for (unsigned int i = 0; i < args.n; i++) {
        const Rid pointer = args.pointers.get(i);
        const Rvl value_i = *results_ptr.get_value_ptr(pointer);
        if (value_i) index |= 1 << i;
    }

    Rvl &value = *results_ptr.get_work_value_ptr();
    value = args.table.get(index);
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
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rvl value = *results_ptr.get_value_ptr(args.target);
    if (value) op_apply_gate1<op>({shots_state_ptr, args.target}, dims_idx);
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
