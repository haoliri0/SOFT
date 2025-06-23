#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./decompose.cuh"
#include "./dimsop.cuh"

using namespace StnCuda;


template<bool dagger>
static __device__
void op_update_amps_half1(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Kid amp_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();

    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0)
        return; // 这个 shot 已经失败，不进行计算
    if (amps_n > amps_m / 2) {
        amps_n = 0;
        return; // 数量超过一半，无法计算，设置为失败
    }
    if (amp_i > amps_n)
        return; // index 超过了 amp 数，不用计算

    const Qid qubits_n = decomp_ptr.qubits_n;
    const Bit *destab_bits = decomp_ptr.get_destab_bits_ptr();
    const Bit *stab_bits = decomp_ptr.get_stab_bits_ptr();

    // 将后半部分当作工作空间存放计算结果
    const Kid src_aid = *amps_map_ptr.get_aid_ptr(amp_i);
    const Amp src_amp = *amps_map_ptr.get_amp_ptr(amp_i);
    Kid &dst_aid = *amps_map_ptr.get_aid_ptr(amps_m / 2 + amp_i);
    Amp &dst_amp = *amps_map_ptr.get_amp_ptr(amps_m / 2 + amp_i);

    const Bit sign_bit = compute_sign(src_aid, stab_bits, qubits_n);
    const Amp sign_amp = sign_bit ? 1 : -1;
    const Amp coef = {0, -sinf(M_PI / 8) * (dagger ? -1 : 1)};

    dst_aid = src_aid ^ bits_to_int(destab_bits, qubits_n);
    dst_amp = src_amp * coef * sign_amp;
}

template<bool dagger>
static __host__
void cuda_update_amps_half1(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_update_amps_half1<dagger>>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m / 2));
}


static __device__
void op_update_amps_half0(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Kid amp_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();

    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0)
        return; // 这个 shot 已经失败，不进行计算
    if (amps_n > amps_m / 2) {
        amps_n = 0;
        return; // 数量超过一半，无法计算，设置为失败
    }
    if (amp_i > amps_n)
        return; // index 超过了 amp 数，不用计算

    // 直接原地修改前半部分
    Amp &src_amp = *amps_map_ptr.get_amp_ptr(amp_i);

    const Amp coef = cosf(M_PI / 8);
    src_amp *= coef;
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
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();

    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    Kid entries_add_n = 0;
    for (Kid src_amp_i = 0; src_amp_i < amps_n; ++src_amp_i) {
        const Kid src_aid = *amps_map_ptr.get_aid_ptr(amps_m / 2 + src_amp_i);
        const Amp src_amp = *amps_map_ptr.get_amp_ptr( amps_m / 2 + src_amp_i);

        // 在 half0 找 aid 对应的条目
        Kid dst_amp_i = 0;
        for (; dst_amp_i < amps_n; ++dst_amp_i) {
            const Kid dst_aid = *amps_map_ptr.get_aid_ptr(dst_amp_i);
            Amp &dst_amp = *amps_map_ptr.get_amp_ptr(dst_amp_i);
            if (dst_aid == src_aid) {
                dst_amp += src_amp;
                break;
            }
        }

        // 在 half0 找不到 aid 对应的条目，就加到的后面
        if (dst_amp_i == amps_n) {
            Kid &dst_aid = *amps_map_ptr.get_aid_ptr(amps_n + entries_add_n);
            Amp &dst_amp = *amps_map_ptr.get_amp_ptr(amps_n + entries_add_n);
            dst_aid = src_aid;
            dst_amp = src_amp;
            entries_add_n += 1;
        }
    }

    // 如果有新的 entry 就修改 entries_n
    if (entries_add_n > 0) amps_n += entries_add_n;
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


template<bool dagger>
static __host__
void cuda_apply_t(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Qid const target
) {
    // T = cos(pi/8) I ∓ i sin(pi/8) Z

    // 先计算 Z 部分
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_update_amps_half1<dagger>(stream, shots_state_ptr);

    // 再计算 I 部分
    cuda_update_amps_half0(stream, shots_state_ptr);

    // 将两部分合并
    cuda_merge_amps_halves(stream, shots_state_ptr);
}

void Simulator::apply_t(const Qid target) const noexcept {
    cuda_apply_t<false>(stream, shots_state_ptr, target);
}

void Simulator::apply_tdg(const Qid target) const noexcept {
    cuda_apply_t<true>(stream, shots_state_ptr, target);
}
