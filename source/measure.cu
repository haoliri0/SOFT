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
void compute_measure_amps_situation1(const ShotStatePtr shot_state_ptr, const Aid amp_i, const Qid pivot) {
    const Qid qubits_n = shot_state_ptr.qubits_n;
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *stab = decomp_ptr.get_stab_bits_ptr();
    const Bit *destab = decomp_ptr.get_destab_bits_ptr();
    const Phs phase = *decomp_ptr.get_phase_ptr();
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Aid aid = *amps_map_ptr.get_aid_ptr(amp_i);
    const Amp amp = *amps_map_ptr.get_amp_ptr(amp_i);
    Aid &aid0 = *amps_map_ptr.get_half0_aid_ptr(amp_i);
    Amp &amp0 = *amps_map_ptr.get_half0_amp_ptr(amp_i);
    Aid &aid1 = *amps_map_ptr.get_half1_aid_ptr(amp_i);
    Amp &amp1 = *amps_map_ptr.get_half1_amp_ptr(amp_i);
    constexpr Flt coef = M_SQRT1_2; // sqrt(1/2);

    constexpr Aid aid_one = 1;
    const Aid aid_mask = aid_one << pivot;
    if (aid & aid_mask) {
        // 取 aid 中的第 pivot 那个 bit
        const Bit sign_phase = phase / 2 % 2;
        const Bit sign_stab = compute_sign(aid, stab, qubits_n);
        const Bit sign = sign_phase ^ sign_stab;
        const Flt phase0 = !sign ? +1. : -1.;
        const Flt phase1 = !sign ? -1. : +1.;

        amp0 = amp * coef * phase0;
        amp1 = amp * coef * phase1;
        aid0 = aid ^ bits_to_int(destab, qubits_n);
        aid1 = aid ^ bits_to_int(destab, qubits_n);
    } else {
        amp0 = amp * coef;
        amp1 = amp * coef;
        aid0 = aid;
        aid1 = aid;
    }
}

static __device__
void op_compute_measure_amps(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Aid const amp_i = dims_idx.get<1>();

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

    // check pivot
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot == NullPivot) {
        compute_measure_amps_situation0(shot_state_ptr, amp_i);
    } else {
        compute_measure_amps_situation1(shot_state_ptr, amp_i, pivot);
    }
}

static __host__
void cuda_compute_measure_amps(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_measure_amps>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m / 2));
}


static __device__
void op_compute_measure_probs_situation0(const AmpsMapPtr amps_map_ptr, const Bit result) {
    const Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    Kid &amps_n_new = *(!result ? amps_map_ptr.get_half0_amps_n_ptr() : amps_map_ptr.get_half1_amps_n_ptr());
    Flt &prob = *(!result ? amps_map_ptr.get_half0_prob_ptr() : amps_map_ptr.get_half1_prob_ptr());
    Aid *aids = !result ? amps_map_ptr.get_half0_aids_ptr() : amps_map_ptr.get_half1_aids_ptr();
    Amp *amps = !result ? amps_map_ptr.get_half0_amps_ptr() : amps_map_ptr.get_half1_amps_ptr();

    amps_n_new = 0;
    prob = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = aids[amp_i];
        const Amp amp = amps[amp_i];

        constexpr Amp amp_zero = 0;
        if (amp == amp_zero) continue;

        const Kid amp_j = amps_n_new;
        aids[amp_j] = aid;
        amps[amp_j] = amp;
        prob += norm(amp);
        amps_n_new++;
    }
}

static __device__
void op_compute_measure_probs_situation1(const AmpsMapPtr amps_map_ptr, const Bit result) {
    const Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    Kid &amps_n_new = *(!result ? amps_map_ptr.get_half0_amps_n_ptr() : amps_map_ptr.get_half1_amps_n_ptr());
    Flt &prob = *(!result ? amps_map_ptr.get_half0_prob_ptr() : amps_map_ptr.get_half1_prob_ptr());
    Aid *aids = !result ? amps_map_ptr.get_half0_aids_ptr() : amps_map_ptr.get_half1_aids_ptr();
    Amp *amps = !result ? amps_map_ptr.get_half0_amps_ptr() : amps_map_ptr.get_half1_amps_ptr();

    amps_n_new = 0;
    for (Kid amp_i = 0; amp_i < amps_n; ++amp_i) {
        const Aid aid = aids[amp_i];
        const Amp amp = amps[amp_i];

        Kid amp_j = 0;
        for (; amp_j < amps_n_new; ++amp_j) {
            if (aids[amp_j] == aid) {
                amps[amp_j] += amp;
                break;
            }
        }

        if (amp_j == amps_n_new) {
            aids[amp_j] = aid;
            amps[amp_j] = amp;
            amps_n_new++;
        }
    }

    prob = 0;
    for (int amp_i = 0; amp_i < amps_n_new; ++amp_i) {
        const Amp amp = amps[amp_i];
        prob += norm(amp);
    }
}

static __device__
void op_compute_measure_probs(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Bit const result = dims_idx.get<1>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

    // check amps_n
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // check pivot
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    pivot == NullPivot
        ? op_compute_measure_probs_situation0(amps_map_ptr, result)
        : op_compute_measure_probs_situation1(amps_map_ptr, result);
}

static __host__
void cuda_compute_measure_probs(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_measure_probs>
        (stream, shots_state_ptr, dimsof(shots_n, 2));
}


enum class SampleMode {
    Random = -1,
    Desire0 = 0,
    Desire1 = 1,
};

template<SampleMode mode = SampleMode::Random>
static __device__
void op_compute_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;

    // check amps_n
    const Kid amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // normalize probs
    Flt &prob0 = *amps_map_ptr.get_half0_prob_ptr();
    Flt &prob1 = *amps_map_ptr.get_half1_prob_ptr();
    const Flt total = prob0 + prob1;
    prob0 /= total;
    prob1 /= total;

    // random choice
    Bit result;
    switch (mode) {
        case SampleMode::Desire0:
            result = false;
            break;
        case SampleMode::Desire1:
            result = true;
            break;
        default:
            curandState *rand_state_ptr = results_ptr.get_rand_state_ptr();
            result = curand_uniform(rand_state_ptr) > prob0;
    }

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

template<SampleMode mode = SampleMode::Random>
static __host__
void cuda_compute_measure_result(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_compute_measure_result<mode>>
        (stream, shots_state_ptr, dimsof(shots_n));
}


static __device__
void op_apply_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Kid const amp_i = dims_idx.get<1>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const AmpsMapPtr amps_map_ptr = shot_state_ptr.get_amps_ptr();
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;

    // check amps_n & amp_i
    Kid &amps_n = *amps_map_ptr.get_amps_n_ptr();
    if (amps_n == 0) return; // 这个 shot 已经失败，不进行计算

    // load result
    const Rid results_n = *results_ptr.get_results_n_ptr();
    const Rid result_i = (results_n - 1) % results_m;
    const Bit result_bit = *results_ptr.get_bit_ptr(result_i);
    const Flt result_prob = *results_ptr.get_prob_ptr(result_i);

    // apply result
    const Kid amps_n_new = *(!result_bit ? amps_map_ptr.get_half0_amps_n_ptr() : amps_map_ptr.get_half1_amps_n_ptr());
    const Aid *aids_src = !result_bit ? amps_map_ptr.get_half0_aids_ptr() : amps_map_ptr.get_half1_aids_ptr();
    const Amp *amps_src = !result_bit ? amps_map_ptr.get_half0_amps_ptr() : amps_map_ptr.get_half1_amps_ptr();
    Aid *aids_dst = amps_map_ptr.get_aids_ptr();
    Amp *amps_dst = amps_map_ptr.get_amps_ptr();

    if (amp_i == 0) amps_n = amps_n_new; // 更新 amps_n（仅线程 0 更新，避免写入冲突）
    if (amp_i >= amps_n_new) return; // 线程超出了 amp_n 的范围，不进行计算

    aids_dst[amp_i] = aids_src[amp_i];
    amps_dst[amp_i] = amps_src[amp_i] / sqrt(result_prob);
}

static __host__
void cuda_apply_measure_result(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Kid amps_m = shots_state_ptr.amps_m;
    cuda_dims_op<ShotsStatePtr, 2, op_apply_measure_result>
        (stream, shots_state_ptr, dimsof(shots_n, amps_m / 2));
}


static __device__
void op_change_measure_basis_rowsum(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Sid const row_i = dims_idx.get<1>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

    // check pivot
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot == NullPivot)
        return; // situation0 不需要改 basis

    // update table
    const TablePtr table_ptr = shot_state_ptr.get_table_ptr();
    const Qid qubits_n = table_ptr.qubits_n;
    if (row_i == qubits_n + pivot)
        return; // skip the pivot row

    const TableRowPtr pivot_row_ptr = table_ptr.get_row_ptr(qubits_n + pivot);
    const TableRowPtr this_row_ptr = table_ptr.get_row_ptr(row_i);

    Phs phase = 0;
    compute_multiply_pauli_rows(
        pivot_row_ptr.get_pauli_ptr(),
        this_row_ptr.get_pauli_ptr(),
        this_row_ptr.get_pauli_ptr(),
        phase);

    const Bit sign = phase / 2 % 2;
    const Bit pivot_sign = *pivot_row_ptr.get_sign_ptr();
    Bit &this_sign = *this_row_ptr.get_sign_ptr();
    this_sign ^= pivot_sign ^ sign;
}

static __host__
void cuda_change_measure_basis_rowsum(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ShotsStatePtr, 2, op_change_measure_basis_rowsum>
        (stream, shots_state_ptr, dimsof(shots_n, rows_n));
}


struct ArgsApplyMeasureBasisPivot {
    const ShotsStatePtr shots_state_ptr;
    const Qid target;
};

static __device__
void op_change_measure_basis_pivot(const ArgsApplyMeasureBasisPivot args, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

    // check pivot
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot == NullPivot)
        return; // situation0 不需要改 basis

    // load result
    const ResultsPtr results_ptr = shot_state_ptr.get_results_ptr();
    const Rid results_m = results_ptr.results_m;
    const Rid results_n = *results_ptr.get_results_n_ptr();
    const Rid result_i = (results_n - 1) % results_m;
    const Bit result_bit = *results_ptr.get_bit_ptr(result_i);

    // update table
    const TablePtr table_ptr = shot_state_ptr.get_table_ptr();
    const Qid qubits_n = table_ptr.qubits_n;
    const Qid cols_n = 2 * qubits_n;

    const TableRowPtr pivot_row_ptr = table_ptr.get_row_ptr(qubits_n + pivot);
    for (int col_i = 0; col_i < cols_n; ++col_i)
        *pivot_row_ptr.get_pauli_ptr().get_bit_ptr(col_i) = false;
    *pivot_row_ptr.get_pauli_ptr().get_bit_ptr(qubits_n + args.target) = true;
    *pivot_row_ptr.get_sign_ptr() = result_bit;
}

static __host__
void cuda_change_measure_basis_pivot(cudaStream_t const stream, ShotsStatePtr const shots_state_ptr, const Qid target) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsApplyMeasureBasisPivot, 1, op_change_measure_basis_pivot>
        (stream, {shots_state_ptr, target}, dimsof(shots_n));
}


void Simulator::measure(const Qid target) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_amps(stream, shots_state_ptr);
    cuda_compute_measure_probs(stream, shots_state_ptr);
    cuda_compute_measure_result(stream, shots_state_ptr);
    cuda_apply_measure_result(stream, shots_state_ptr);

    cuda_change_measure_basis_rowsum(stream, shots_state_ptr);
    cuda_change_measure_basis_pivot(stream, shots_state_ptr, target);
}

void Simulator::desire(const Qid target, const Bit result) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_amps(stream, shots_state_ptr);
    cuda_compute_measure_probs(stream, shots_state_ptr);
    !result
        ? cuda_compute_measure_result<SampleMode::Desire0>(stream, shots_state_ptr)
        : cuda_compute_measure_result<SampleMode::Desire1>(stream, shots_state_ptr);
    cuda_apply_measure_result(stream, shots_state_ptr);

    cuda_change_measure_basis_rowsum(stream, shots_state_ptr);
    cuda_change_measure_basis_pivot(stream, shots_state_ptr, target);
}
