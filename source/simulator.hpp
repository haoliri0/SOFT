#ifndef STN_CUDA_SIMULATOR_HPP
#define STN_CUDA_SIMULATOR_HPP

#include<cuda_runtime.h>
#include "./array.cuh"
#include "./datatype.cuh"
#include "./datastruct.cuh"

namespace StnCuda {

template<Rid m = 16>
struct ClassicalReduceArgs {
    Rid n;
    Array<Rid, m> pointers;
};

template<Rid m = 6>
struct ClassicalLutArgs {
    Rid n;
    Array<Rid, m> pointers;
    Array<Bit, 1 << m> table;
};


struct Simulator {
    cudaStream_t stream = nullptr;
    ShotsStatePtr shots_state_ptr = {0, 0, 0, 0, nullptr};


    cudaError_t create(Sid shots_n, Qid qubits_n, Eid entries_m, Rid results_m, unsigned long long seed) noexcept;

    cudaError_t destroy() noexcept;


    void apply_x(Qid target) const noexcept;

    void apply_y(Qid target) const noexcept;

    void apply_z(Qid target) const noexcept;

    void apply_h(Qid target) const noexcept;

    void apply_s(Qid target) const noexcept;

    void apply_sdg(Qid target) const noexcept;

    void apply_cx(Qid control, Qid target) const noexcept;


    void apply_t(Qid target) const noexcept;

    void apply_tdg(Qid target) const noexcept;


    void apply_measure(Qid target) const noexcept;

    void apply_desire(Qid target, Bit value) const noexcept;

    void apply_reset(Qid target, Bit value) const noexcept;


    void apply_noise_x(Flt prob, Qid target) const noexcept;

    void apply_noise_z(Flt prob, Qid target) const noexcept;

    void apply_noise_depo1(Flt prob, Qid target) const noexcept;

    void apply_noise_depo2(Flt prob, Qid target0, Qid target1) const noexcept;


    void apply_classical_not() const noexcept;

    void apply_classical_set(Rvl value) const noexcept;

    void apply_classical_read(Rid pointer) const noexcept;

    void apply_classical_write(Rid pointer) const noexcept;

    void apply_classical_check() const noexcept;


    void apply_classical_or(ClassicalReduceArgs<> args) const noexcept;

    void apply_classical_xor(ClassicalReduceArgs<> args) const noexcept;

    void apply_classical_and(ClassicalReduceArgs<> args) const noexcept;

    void apply_classical_lut(ClassicalLutArgs<> args) const noexcept;


    void apply_classical_controlled_x(Qid target) const noexcept;

    void apply_classical_controlled_y(Qid target) const noexcept;

    void apply_classical_controlled_z(Qid target) const noexcept;

};

}

#endif
