#ifndef STN_CUDA_SIMULATOR_HPP
#define STN_CUDA_SIMULATOR_HPP

#include<cuda_runtime.h>
#include "./array.cuh"
#include "./datatype.cuh"
#include "./datastruct.cuh"

namespace StnCuda {

struct SimulatorArgs {
    Sid shots_n = 1;
    Qid qubits_n = 4;
    Eid entries_m = 16;
    Mid mem_ints_m = 16;
    Mid mem_flts_m = 16;
    unsigned long long seed = 0;
};

template<Mid m = 64>
struct ClassicalReduceArgs {
    Mid n;
    Array<Mid, m> pointers;
};

template<Mid m = 6>
struct ClassicalLutArgs {
    Mid n;
    Array<Mid, m> pointers;
    Array<Bit, 1 << m> table;
};


struct Simulator {
    cudaStream_t stream = nullptr;
    ShotsStatePtr shots_state_ptr = {0, 0, 0, 0, 0, nullptr};

    cudaError_t create(SimulatorArgs const &args) noexcept;

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


    void apply_classical_flip() const noexcept;

    void apply_classical_check(Err error) const noexcept;


    void apply_classical_load_int(Mid pointer) const noexcept;

    void apply_classical_load_flt(Mid pointer) const noexcept;

    void apply_classical_save_int(Mid pointer) const noexcept;

    void apply_classical_save_flt(Mid pointer) const noexcept;


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
