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
void cuda_compute_amps_branch0(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_amps_branch0>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m));
}


void Simulator::measure(const Qid target, Bit *res, Flt *prob) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    // TODO
}
