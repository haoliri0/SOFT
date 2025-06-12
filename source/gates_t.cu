#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./dimsop.cuh"

using namespace StnCuda;


static __device__
Aid bits_to_int(const Bit *bits, const unsigned int n) {
    Aid v = 0;
    for (unsigned int i = 0; i < n; i++) {
        const Bit bit = bits[i];
        v <<= 1;
        v |= bit;
    }
    return v;
}

static __device__
Bit compute_sign(Aid key, const Bit *stab_bits, const Qid qubits_n) {
    Bit sign = false;
    for (int qubit_i = 0; qubit_i < qubits_n; ++qubit_i) {
        const Bit key_bit = key % 2;
        const Bit stab_bit = stab_bits[qubit_i];
        sign ^= stab_bit & key_bit;
        key >>= 1;
    }
    return sign;
}

static __device__
void op_update_amps_half1(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Kid amp_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_state_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_map_ptr();

    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_entries_n_ptr();
    if (amps_n == 0)
        return; // 这个 shot 已经失败，不进行计算
    if (amps_n > amps_m / 2) {
        amps_n = 0;
        return; // 数量超过一半，无法计算，设置为失败
    }
    if (amp_i > amps_n)
        return; // index 超过了 amp 数，不用计算

    const Qid qubits_n = decomp_ptr.qubits_n;
    const Bit *decomp_bits = decomp_ptr.get_bits_ptr();
    const Bit *destab_bits = decomp_bits;
    const Bit *stab_bits = decomp_bits + qubits_n;

    const AmpEntry src = *amps_map_ptr.get_half0_entry_ptr(amp_i);
    AmpEntry &dst = *amps_map_ptr.get_half1_entry_ptr(amp_i);

    const Bit sign_bit = compute_sign(src.key, stab_bits, qubits_n);
    const Amp sign_amp = sign_bit ? 1 : -1;
    const Amp coef = {0, -sinf(M_PI / 8)};

    dst.key = src.key ^ bits_to_int(destab_bits, qubits_n);
    dst.value = src.value * coef * sign_amp;
}

static __device__
void op_update_amps_half0(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Kid amp_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_state_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_map_ptr();

    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_entries_n_ptr();
    if (amps_n == 0)
        return; // 这个 shot 已经失败，不进行计算
    if (amps_n > amps_m / 2) {
        amps_n = 0;
        return; // 数量超过一半，无法计算，设置为失败
    }
    if (amp_i > amps_n)
        return; // index 超过了 amp 数，不用计算

    AmpEntry &src = *amps_map_ptr.get_half0_entry_ptr(amp_i);
    const Amp coef = sinf(M_PI / 8);
    src.value *= coef;
}

static __host__
void cuda_update_amps_half1(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_update_amps_half1>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m / 2));
}

static __host__
void cuda_update_amps_half0(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_update_amps_half0>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m / 2));
}


static __device__
void op_merge_amps_halves(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_state_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_map_ptr();

    Kid &entries_n = *amps_map_ptr.get_entries_n_ptr();
    if (entries_n == 0) return; // 这个 shot 已经失败，不进行计算

    Kid entries_add_n = 0;
    for (Kid ai1 = 0; ai1 < entries_n; ++ai1) {
        const AmpEntry entry1 = *amps_map_ptr.get_half1_entry_ptr(ai1);
        Kid ai0 = 0;
        for (; ai0 < entries_n; ++ai0) {
            AmpEntry &entry0 = *amps_map_ptr.get_half0_entry_ptr(ai0);
            if (entry0.key == entry1.key) {
                entry0.value += entry1.value;
                break;
            }
        }
        if (ai0 == entries_n) {
            // 在 half0 找不到对应的 key 就加到 half0 的后面
            AmpEntry &entry_add = *amps_map_ptr.get_entry_ptr(entries_n + entries_add_n);
            entry_add = entry1;
            entries_add_n += 1;
        }
    }

    // 如果有新的 entry 就修改 entries_n
    if (entries_add_n > 0) entries_n += entries_add_n;
}

static __host__
void cuda_merge_amps_halves(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_merge_amps_halves>
        (stream, shots_state_ptr, dimsof(shots_n));
}


void Simulator::apply_t(const Qid target) const noexcept {
    // T = cos(pi/8) I - i sin(pi/8) Z

    // 先计算 Z 部分
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_update_amps_half1(stream, shots_state_ptr);

    // 再计算 I 部分
    cuda_update_amps_half0(stream, shots_state_ptr);

    // 将两部分合并
    cuda_merge_amps_halves(stream, shots_state_ptr);
}
