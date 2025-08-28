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

void Simulator::apply_classical_not() const noexcept {
    cuda_shots_op<op_classical_not>(stream, shots_state_ptr);
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

void Simulator::apply_classical_read(const Rid pointer) const noexcept {
    cuda_shots_op<Rid, op_classical_read>(stream, shots_state_ptr, pointer);
}

void Simulator::apply_classical_write(const Rid pointer) const noexcept {
    cuda_shots_op<Rid, op_classical_write>(stream, shots_state_ptr, pointer);
}


template<Rid n, Rvl (*reduction)(Rvl, Rvl)>
static __device__
void op_classical_reduce(const ShotStatePtr shot_state_ptr, const Array<Rid, n> pointers, const Bit value0) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    Rvl &value = *results_ptr.get_work_value_ptr();
    if constexpr (n > 0) {
        const Rvl value1 = *results_ptr.get_value_ptr(pointers.template get<0>());
        value = reduction(value0, value1);
    } else {
        value = value0;
    }
}

template<Rid n, Rvl (*reduction)(Rvl, Rvl), Rvl value0>
static __device__
void op_classical_reduce(const ShotStatePtr shot_state_ptr, const Array<Rid, n> pointers) {
    op_classical_reduce<n, reduction>(shot_state_ptr, pointers, value0);
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

template<Rid n>
void Simulator::apply_classical_or(Array<Rid, n> pointers) const noexcept {
    cuda_shots_op<Array<Rid, n>, op_classical_reduce<n, reduction_logical_or, 0>>(stream, shots_state_ptr, pointers);
}

template<Rid n>
void Simulator::apply_classical_xor(Array<Rid, n> pointers) const noexcept {
    cuda_shots_op<Array<Rid, n>, op_classical_reduce<n, reduction_logical_xor, 0>>(stream, shots_state_ptr, pointers);
}

template<Rid n>
void Simulator::apply_classical_and(Array<Rid, n> pointers) const noexcept {
    cuda_shots_op<Array<Rid, n>, op_classical_reduce<n, reduction_logical_and, 0>>(stream, shots_state_ptr, pointers);
}


template<Rid n>
struct ArgsClassicalLut {
    Array<Rid, n> pointers;
    Array<Bit, 1 << n> table;
};

template<Rid n>
Array<Bit, n> read_bits_array(const Rvl *values, Array<Rid, n> pointers) {
    if constexpr (n > 0) {
        auto [head_pointer,tail_pointers] = pointers;
        return {values[head_pointer], read_bits_array<n - 1>(values, tail_pointers)};
    } else {
        return {};
    }
}

template<Rid n>
static __device__
void op_classical_lut(const ShotStatePtr shot_state_ptr, const ArgsClassicalLut<n> args) {
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    auto [pointers,table] = args;
    auto values = read_bits_array<n>(results_ptr.get_values_ptr(), pointers);
    Rvl &value = *results_ptr.get_work_value_ptr();

    unsigned int index = 0;
    for (unsigned int i = 0; i < n; i++)
        if (values[i]) index |= 1 << i;
    value = table.get(index);
}

template<Rid n>
void Simulator::apply_classical_lut(Array<Rid, n> pointers, Array<Bit, 1 << n> table) const noexcept {
    cuda_shots_op<ArgsClassicalLut<n>, op_classical_lut<n>>(stream, shots_state_ptr, {pointers, table});
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
    const Bit cond = *results_ptr.get_work_value_ptr();
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
