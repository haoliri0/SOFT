#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./decompose.cuh"
#include "./datatype.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;

static __device__
void compute_measure_amps_situation0(const ShotStatePtr shot_state_ptr, const Aid amp_i) {
    const Qid qubits_n = shot_state_ptr.qubits_n;
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *stab = decomp_ptr.get_stab_bits_ptr();
    const Phs phase = *decomp_ptr.get_phase_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Aid aid = *amps_map_ptr.get_aid_ptr(amp_i);
    const Amp amp = *amps_map_ptr.get_amp_ptr(amp_i);
    Aid &aid0 = *amps_map_ptr.get_half0_aid_ptr(amp_i);
    Amp &amp0 = *amps_map_ptr.get_half0_amp_ptr(amp_i);
    Aid &aid1 = *amps_map_ptr.get_half1_aid_ptr(amp_i);
    Amp &amp1 = *amps_map_ptr.get_half1_amp_ptr(amp_i);

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

static __device__
void compute_measure_amps_situation1(const ShotStatePtr shot_state_ptr, const Aid amp_i) {
    const Qid qubits_n = shot_state_ptr.qubits_n;
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *stab = decomp_ptr.get_stab_bits_ptr();
    const Phs phase = *decomp_ptr.get_phase_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Aid aid = *amps_map_ptr.get_aid_ptr(amp_i);
    const Amp amp = *amps_map_ptr.get_amp_ptr(amp_i);
    Aid &aid0 = *amps_map_ptr.get_half0_aid_ptr(amp_i);
    Amp &amp0 = *amps_map_ptr.get_half0_amp_ptr(amp_i);
    Aid &aid1 = *amps_map_ptr.get_half1_aid_ptr(amp_i);
    Amp &amp1 = *amps_map_ptr.get_half1_amp_ptr(amp_i);

    // TODO
    // const Bit sign_phase = phase / 2 % 2;
    // const Bit sign_stab = compute_sign(aid, stab, qubits_n);
    // const Bit sign = sign_phase ^ sign_stab;

    // aid0 = aid;
    // aid1 = aid;
    // if (!sign) {
    //     amp0 = amp;
    //     amp1 = 0;
    // } else {
    //     amp0 = 0;
    //     amp1 = amp;
    // }
}

static __device__
void op_compute_measure_amps(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Aid const amp_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();

    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amp_i >= amps_n)
        return; // index 超过了 amp 数，不用计算
    if (amps_n > amps_m / 2) {
        // 数量超过一半，无法计算，设置为 0 表示失败
        amps_n = 0;
        return;
    }

    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot != NullPivot) {
        compute_measure_amps_situation0(shot_state_ptr, amp_i);
    } else {
        compute_measure_amps_situation1(shot_state_ptr, amp_i);
    }
}

static __host__
void cuda_compute_measure_amps(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_measure_amps>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m));
}


static __device__
void op_compute_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;

    const Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // compute probs
    Flt prob0 = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Amp amp = *amps_map_ptr.get_half0_amp_ptr(amp_i);
        prob0 += cuda::std::norm(amp);
    }

    Flt prob1 = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Amp amp = *amps_map_ptr.get_half1_amp_ptr(amp_i);
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
void op_apply_measure_result_situation0(const AmpsMapPtr amps_map_ptr, const Bit result_bit, const Flt result_prob) {
    Kid amps_n_new = 0;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    const Aid *aids = !result_bit ? amps_map_ptr.get_half0_aids_ptr() : amps_map_ptr.get_half1_aids_ptr();
    const Amp *amps = !result_bit ? amps_map_ptr.get_half0_amps_ptr() : amps_map_ptr.get_half1_amps_ptr();
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = aids[amp_i];
        const Amp amp = amps[amp_i];

        constexpr Amp amp_zero = 0;
        if (amp == amp_zero) continue;

        const Kid amp_j = amps_n_new++;
        Aid &aid_dst = *amps_map_ptr.get_aid_ptr(amp_j);
        Amp &amp_dst = *amps_map_ptr.get_amp_ptr(amp_j);

        aid_dst = aid;
        amp_dst = amp / result_prob;
    }
    amps_n = amps_n_new;
}

static __device__
void op_apply_measure_result_situation1(const AmpsMapPtr amps_map_ptr, const Bit result_bit, const Flt result_prob) {
    Kid amps_n_new = 0;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    const Aid *aids = !result_bit ? amps_map_ptr.get_half0_aids_ptr() : amps_map_ptr.get_half1_aids_ptr();
    const Amp *amps = !result_bit ? amps_map_ptr.get_half0_amps_ptr() : amps_map_ptr.get_half1_amps_ptr();
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = aids[amp_i];
        const Amp amp = amps[amp_i];

        Kid amp_j = 0;
        for (; amp_j < amps_n_new; ++amp_j) {
            const Aid aid_dst = *amps_map_ptr.get_aid_ptr(amp_j);
            Amp &amp_dst = *amps_map_ptr.get_amp_ptr(amp_j);
            if (aid_dst == aid) {
                amp_dst += amp / result_prob;
                break;
            }
        }

        if (amp_j == amps_n_new) {
            Aid &aid_dst = *amps_map_ptr.get_aid_ptr(amp_j);
            Amp &amp_dst = *amps_map_ptr.get_amp_ptr(amp_j);
            aid_dst = aid;
            amp_dst = amp / result_prob;
            amps_n_new++;
        }
    }
    amps_n = amps_n_new;
}

static __device__
void op_apply_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;

    Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // load result
    const Rid results_n = *results_ptr.get_results_n_ptr();
    const Rid result_i = (results_n - 1) % results_m;
    const Bit result_bit = *results_ptr.get_bit_ptr(result_i);
    const Flt result_prob = *results_ptr.get_prob_ptr(result_i);

    // sort amplitudes
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot != NullPivot) {
        op_apply_measure_result_situation0(amps_map_ptr, result_bit, result_prob);
    } else {
        op_apply_measure_result_situation1(amps_map_ptr, result_bit, result_prob);
    }
}

static __host__
void cuda_apply_measure_result(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_apply_measure_result>
        (stream, shots_state_ptr, dimsof(shots_n));
}


void Simulator::measure(const Qid target) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_amps(stream, shots_state_ptr);
    cuda_compute_measure_result(stream, shots_state_ptr);
    cuda_apply_measure_result(stream, shots_state_ptr);
}
