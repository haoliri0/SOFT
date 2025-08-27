#include "./simulator.hpp"
#include "./shotsop.cuh"

using namespace StnCuda;

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
