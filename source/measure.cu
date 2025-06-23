#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./decompose.cuh"
#include "./datatype.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;


static __device__
void op_compute_amps_branch0(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Aid const amp_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Kid amps_m = shots_state_ptr.amps_m;

    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *stab = decomp_ptr.get_stab_bits_ptr();
    const Phs phase = *decomp_ptr.get_phase_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Aid aid = *amps_map_ptr.get_aid_ptr(amp_i);
    const Amp amp = *amps_map_ptr.get_amp_ptr(amp_i);
    Aid &aid0 = *amps_map_ptr.get_aid_ptr(amp_i);
    Amp &amp0 = *amps_map_ptr.get_amp_ptr(amp_i);
    Aid &aid1 = *amps_map_ptr.get_aid_ptr(amps_m / 2 + amp_i);
    Amp &amp1 = *amps_map_ptr.get_amp_ptr(amps_m / 2 + amp_i);

    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amp_i >= amps_n)
        return; // index 超过了 amp 数，不用计算
    if (amps_n > amps_m / 2) {
        // 数量超过一半，无法计算，设置为 0 表示失败
        amps_n = 0;
        return;
    }

    const Bit sign_phase = phase / 2 % 2;
    const Bit sign_stab = compute_sign(aid, stab, qubits_n);
    const Bit sign = sign_phase ^ sign_stab;

    aid0 = aid;
    aid1 = aid;
    if (!sign) {
        amp0 = amp;
        amp1 = 0;
    } else {
        amp0 = 0;
        amp1 = amp;
    }
}

static __host__
void cuda_compute_measure_amps_branch0(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_amps_branch0>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m));
}


static __device__
void op_compute_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Kid amps_m = amps_map_ptr.amps_m;
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;

    const Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // compute probs
    Flt prob0 = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Amp amp = *amps_map_ptr.get_amp_ptr(amp_i);
        prob0 += cuda::std::norm(amp);
    }

    Flt prob1 = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Amp amp = *amps_map_ptr.get_amp_ptr(amps_m / 2 + amp_i);
        prob1 += cuda::std::norm(amp);
    }

    // normalize probs
    const Flt total = prob0 + prob1;
    prob0 /= total;
    prob1 /= total;

    // random choice
    curandState *rand_state_ptr = results_ptr.get_rand_state_ptr();
    const Bit result = curand_uniform(rand_state_ptr) > prob0;

    // save result
    Rid &results_n = *results_ptr.get_results_n_ptr();
    const Rid result_i = results_n % results_m;
    Flt &result_prob = *results_ptr.get_prob_ptr(result_i);
    Bit &result_bit = *results_ptr.get_bit_ptr(result_i);

    results_n += 1;
    if (!result) {
        result_prob = prob0;
        result_bit = false;
    } else {
        result_prob = prob1;
        result_bit = true;
    }
}

static __host__
void cuda_compute_measure_result(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_compute_measure_result>
        (stream, shots_state_ptr, dimsof(shots_n));
}


static __device__
void op_apply_measure_result_branch0(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Kid amps_m = amps_map_ptr.amps_m;
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;

    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // load result
    const Rid results_n = *results_ptr.get_results_n_ptr();
    const Rid result_i = (results_n - 1) % results_m;
    const Bit result_bit = *results_ptr.get_bit_ptr(result_i);
    const Flt result_prob = *results_ptr.get_prob_ptr(result_i);
    const Kid amp_i_offset = result_bit ? amps_m / 2 : 0;

    // compute probs
    Kid amps_n_new = 0;
    constexpr Amp amp_zero = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Amp amp = *amps_map_ptr.get_amp_ptr(amp_i + amp_i_offset);
        const Aid aid = *amps_map_ptr.get_aid_ptr(amp_i + amp_i_offset);
        if (amp == amp_zero) continue;

        const Kid amp_i_new = amps_n_new++;
        *amps_map_ptr.get_amp_ptr(amp_i_new) = amp / result_prob;
        *amps_map_ptr.get_aid_ptr(amp_i_new) = aid;
    }
    amps_n = amps_n_new;
}

static __host__
void cuda_apply_measure_result_branch0(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_apply_measure_result_branch0>
        (stream, shots_state_ptr, dimsof(shots_n));
}


void Simulator::measure(const Qid target) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_amps_branch0(stream, shots_state_ptr);
    cuda_compute_measure_result(stream, shots_state_ptr);
    cuda_apply_measure_result_branch0(stream, shots_state_ptr);
}
