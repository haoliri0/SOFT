#ifndef STN_CUDA_SIMULATOR_HPP
#define STN_CUDA_SIMULATOR_HPP

#include<cuda_runtime.h>
#include "./datatype.cuh"
#include "./datastruct.cuh"

namespace StnCuda {

struct Simulator {
    cudaStream_t stream = nullptr;
    ShotsStatePtr shots_state_ptr = {0, 0, 0, 0, nullptr};


    cudaError_t create(Sid shots_n, Qid qubits_n, Aid amps_m, Rid results_m, unsigned long long seed) noexcept;

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

    void apply_desire(Qid target, Bit result) const noexcept;

    void apply_assign(Qid target, Bit value) const noexcept;


    void apply_noise_x(Flt prob, Qid target) const noexcept;

    void apply_noise_z(Flt prob, Qid target) const noexcept;

    void apply_noise_depo1(Flt prob, Qid target) const noexcept;

    void apply_noise_depo2(Flt prob, Qid target0, Qid target1) const noexcept;

};

}

#endif
