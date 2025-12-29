#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./decompose.cuh"
#include "./shotsop.cuh"
#include "./dimsop.cuh"

using namespace SoftCuda;

template<bool dagger>
static __device__
void op_update_entries_half1(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Eid entry_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    // check error
    Err &err = *work_ptr.get_err_ptr();
    if (err) return; // 这个 shot 已经失败，不进行计算

    // check entries_m, entries_n, entry_i
    const Eid entries_m = entries_ptr.entries_m;
    const Eid entries_n = *entries_ptr.get_entries_n_ptr();
    if (entry_i >= entries_n) return; // 线程超出了 amp_n 的范围，不进行计算
    if (entry_i == 0 && entries_n > entries_m / 2) {
        // 数量超过一半，无法计算，设置 err 状态
        // 只有一条线程会执行该修改，避免写入冲突
        err = err_entries_overflow;
        return;
    }

    const Qid qubits_n = decomp_ptr.qubits_n;
    const Bit *destab_bits = decomp_ptr.get_destab_bits_ptr();
    const Bit *stab_bits = decomp_ptr.get_stab_bits_ptr();

    // 将后半部分当作工作空间存放计算结果
    const Bst src_bst = *entries_ptr.get_half0_bst_ptr(entry_i);
    const Amp src_amp = *entries_ptr.get_half0_amp_ptr(entry_i);
    Bst &dst_bst = *entries_ptr.get_half1_bst_ptr(entry_i);
    Amp &dst_amp = *entries_ptr.get_half1_amp_ptr(entry_i);

    const Amp coef = {0, -sinf(M_PI / 8)};

    const Bit dagger_sign = dagger;
    const Flt dagger_sign_amp = sign_to_flt(dagger_sign);

    const Bit bst_sign = compute_sign(src_bst, stab_bits, qubits_n);
    const Flt bst_sign_amp = sign_to_flt(bst_sign);

    const Phs decomp_phase = *decomp_ptr.get_phase_ptr();
    const Phs decomp_phase_inv = -decomp_phase; // 这是 DDD 乘起来的系数，要取反得到 Z 分解的系数！
    const Amp decomp_phase_amp = phase_to_amp(decomp_phase_inv);

    dst_bst = src_bst ^ bits_to_bst(destab_bits, qubits_n);
    dst_amp = src_amp * coef * dagger_sign_amp * bst_sign_amp * decomp_phase_amp;
}

template<bool dagger>
static __host__
void cuda_update_entries_half1(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Eid entries_m = shots_state_ptr.entries_m;
    cuda_dims_op<ShotsStatePtr, 2, op_update_entries_half1<dagger>>
        (stream, shots_state_ptr, dimsof(shots_n, entries_m / 2));
}


static __device__
void op_update_entries_half0(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    const Sid shot_i = dims_idx.get<0>();
    const Eid entry_i = dims_idx.get<1>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    // check error
    Err &err = *work_ptr.get_err_ptr();
    if (err) return; // 这个 shot 已经失败，不进行计算

    // check entries_m, entries_n, entry_i
    const Eid entries_m = entries_ptr.entries_m;
    const Eid entries_n = *entries_ptr.get_entries_n_ptr();
    if (entry_i >= entries_n) return; // 线程超出了 amp_n 的范围，不进行计算
    if (entry_i == 0 && entries_n > entries_m / 2) {
        // 数量超过一半，无法计算，设置 err 状态
        // 只有一条线程会执行该修改，避免写入冲突
        err = err_entries_overflow;
        return;
    }

    // 直接原地修改前半部分
    Amp &src_amp = *entries_ptr.get_half0_amp_ptr(entry_i);

    const Amp coef = cosf(M_PI / 8);
    src_amp *= coef;
}

static __host__
void cuda_update_entries_half0(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Eid entries_m = shots_state_ptr.entries_m;
    cuda_dims_op<ShotsStatePtr, 2, op_update_entries_half0>
        (stream, shots_state_ptr, dimsof(shots_n, entries_m / 2));
}


static __device__
void op_merge_entries_halves(const ShotStatePtr shot_state_ptr, const Flt epsilon) {
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    // check error
    const Err err = *work_ptr.get_err_ptr();
    if (err) return; // 这个 shot 已经失败，不进行计算

    // 合并 half1 的条目到 half0
    Eid entries_add_n = 0;
    Eid &entries_n = *entries_ptr.get_entries_n_ptr();
    for (Eid src_entry_i = 0; src_entry_i < entries_n; ++src_entry_i) {
        const Bst src_bst = *entries_ptr.get_half1_bst_ptr(src_entry_i);
        const Amp src_amp = *entries_ptr.get_half1_amp_ptr(src_entry_i);

        // 在 half0 找 bst 对应的条目
        Eid dst_entry_i = 0;
        for (; dst_entry_i < entries_n; ++dst_entry_i) {
            const Bst dst_bst = *entries_ptr.get_half0_bst_ptr(dst_entry_i);
            Amp &dst_amp = *entries_ptr.get_half0_amp_ptr(dst_entry_i);
            if (dst_bst == src_bst) {
                dst_amp += src_amp;
                break;
            }
        }

        // 在 half0 找不到 bst 对应的条目，就加到的后面
        if (dst_entry_i == entries_n) {
            Bst &dst_bst = *entries_ptr.get_half0_bst_ptr(entries_n + entries_add_n);
            Amp &dst_amp = *entries_ptr.get_half0_amp_ptr(entries_n + entries_add_n);
            dst_bst = src_bst;
            dst_amp = src_amp;
            entries_add_n += 1;
        }
    }

    // 如果有新的 entry 就修改 entries_n
    if (entries_add_n > 0) entries_n += entries_add_n;

    // 判断是否跳过条目清理
    if (epsilon < 0) return;

    // 清理接近 0 的条目
    Eid entries_del_n = 0;
    for (Eid entry_i = 0; entry_i < entries_n; ++entry_i) {
        const Bst src_bst = *entries_ptr.get_bst_ptr(entry_i);
        const Amp src_amp = *entries_ptr.get_amp_ptr(entry_i);

        if (cuda::std::abs(src_amp) <= epsilon) {
            entries_del_n += 1;
            continue;
        }

        if (entries_del_n == 0)
            continue;

        Bst &dst_bst = *entries_ptr.get_bst_ptr(entry_i - entries_del_n);
        Amp &dst_amp = *entries_ptr.get_amp_ptr(entry_i - entries_del_n);
        dst_bst = src_bst;
        dst_amp = src_amp;
    }

    // 如果有被清理的 entry 就修改 entries_n
    if (entries_del_n > 0) entries_n -= entries_del_n;
}

static __host__
void cuda_merge_entries_halves(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr,
    Flt const epsilon
) {
    cuda_shots_op<Flt, op_merge_entries_halves>(stream, shots_state_ptr, epsilon);
}


template<bool dagger>
static __host__
void cuda_apply_t(
    cudaStream_t const &stream,
    ShotsStatePtr const &shots_state_ptr,
    Flt const epsilon,
    Qid const target
) {
    // T = cos(pi/8) I ∓ i sin(pi/8) Z

    // 先计算 Z 部分
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_update_entries_half1<dagger>(stream, shots_state_ptr);

    // 再计算 I 部分
    cuda_update_entries_half0(stream, shots_state_ptr);

    // 将两部分合并
    cuda_merge_entries_halves(stream, shots_state_ptr, epsilon);
}

void Simulator::apply_t(const Qid target) const noexcept {
    cuda_apply_t<false>(stream, shots_state_ptr, epsilon, target);
}

void Simulator::apply_tdg(const Qid target) const noexcept {
    cuda_apply_t<true>(stream, shots_state_ptr, epsilon, target);
}
