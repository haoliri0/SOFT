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
