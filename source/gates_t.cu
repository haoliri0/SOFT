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

    // check amps_n & amp_i
    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算
    if (amp_i >= amps_n) return; // 线程超出了 amp_n 的范围，不进行计算
    if (amp_i == 0 && amps_n > amps_m / 2) {
        // 数量超过一半，无法计算，设置为 0 表示失败
        // 只有一条线程会执行该修改，避免写入冲突
        amps_n = 0;
        return;
    }

    const Qid qubits_n = decomp_ptr.qubits_n;
    const Bit *destab_bits = decomp_ptr.get_destab_bits_ptr();
    const Bit *stab_bits = decomp_ptr.get_stab_bits_ptr();

    // 将后半部分当作工作空间存放计算结果
    const Kid src_aid = *amps_map_ptr.get_half0_aid_ptr(amp_i);
    const Amp src_amp = *amps_map_ptr.get_half0_amp_ptr(amp_i);
    Kid &dst_aid = *amps_map_ptr.get_half1_aid_ptr(amp_i);
    Amp &dst_amp = *amps_map_ptr.get_half1_amp_ptr(amp_i);

    const Amp coef = {0, -sinf(M_PI / 8)};

    const Bit dagger_sign = dagger;
    const Flt dagger_sign_amp = sign_to_flt(dagger_sign);

    const Bit aid_sign = compute_sign(src_aid, stab_bits, qubits_n);
    const Flt aid_sign_amp = sign_to_flt(aid_sign);

    const Phs decomp_phase = *decomp_ptr.get_phase_ptr();
    const Phs decomp_phase_inv = -decomp_phase; // 这是 DDD 乘起来的系数，要取反得到 Z 分解的系数！
    const Amp decomp_phase_amp = phase_to_amp(decomp_phase_inv);

    dst_aid = src_aid ^ bits_to_int(destab_bits, qubits_n);
    dst_amp = src_amp * coef * dagger_sign_amp * aid_sign_amp * decomp_phase_amp;
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

    // check amps_n & amp_i
    const Kid amps_m = amps_map_ptr.amps_m;
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算
    if (amp_i >= amps_n) return; // 线程超出了 amp_n 的范围，不进行计算
    if (amp_i == 0 && amps_n > amps_m / 2) {
        // 数量超过一半，无法计算，设置为 0 表示失败
        // 只有一条线程会执行该修改，避免写入冲突
        amps_n = 0;
        return;
    }

    // 直接原地修改前半部分
    Amp &src_amp = *amps_map_ptr.get_half0_amp_ptr(amp_i);

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

    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    Kid entries_add_n = 0;
    for (Kid src_amp_i = 0; src_amp_i < amps_n; ++src_amp_i) {
        const Kid src_aid = *amps_map_ptr.get_half1_aid_ptr(src_amp_i);
        const Amp src_amp = *amps_map_ptr.get_half1_amp_ptr(src_amp_i);

        // 在 half0 找 aid 对应的条目
        Kid dst_amp_i = 0;
        for (; dst_amp_i < amps_n; ++dst_amp_i) {
            const Kid dst_aid = *amps_map_ptr.get_half0_aid_ptr(dst_amp_i);
            Amp &dst_amp = *amps_map_ptr.get_half0_amp_ptr(dst_amp_i);
            if (dst_aid == src_aid) {
                dst_amp += src_amp;
                break;
            }
        }

        // 在 half0 找不到 aid 对应的条目，就加到的后面
        if (dst_amp_i == amps_n) {
            Kid &dst_aid = *amps_map_ptr.get_half0_aid_ptr(amps_n + entries_add_n);
            Amp &dst_amp = *amps_map_ptr.get_half0_amp_ptr(amps_n + entries_add_n);
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
