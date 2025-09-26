#include "./simulator.hpp"
#include "./decompose.hpp"
#include "./decompose.cuh"
#include "./datatype.cuh"
#include "./dimsop.cuh"
#include "./gates.cuh"

using namespace StnCuda;

static __device__
void compute_measure_entries_situation0(const ShotStatePtr shot_state_ptr, const Eid entry_i) {
    const Qid qubits_n = shot_state_ptr.qubits_n;
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *stab = decomp_ptr.get_stab_bits_ptr();
    const Phs decomp_phase = *decomp_ptr.get_phase_ptr();
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const Bst bst = *entries_ptr.get_bst_ptr(entry_i);
    const Amp amp = *entries_ptr.get_amp_ptr(entry_i);
    Bst &bst0 = *entries_ptr.get_half0_bst_ptr(entry_i);
    Amp &amp0 = *entries_ptr.get_half0_amp_ptr(entry_i);
    Bst &bst1 = *entries_ptr.get_half1_bst_ptr(entry_i);
    Amp &amp1 = *entries_ptr.get_half1_amp_ptr(entry_i);

    const Bit sign_phase = decomp_phase / 2 % 2; // definitely +1 or -1
    const Bit sign_stab = compute_sign(bst, stab, qubits_n);
    const Bit sign = sign_phase ^ sign_stab;

    bst0 = bst;
    bst1 = bst;
    if (!sign) {
        amp0 = amp;
        amp1 = 0;
    } else {
        amp0 = 0;
        amp1 = amp;
    }
}

static __device__
void compute_measure_entries_situation1(const ShotStatePtr shot_state_ptr, const Eid entry_i, const Qid pivot) {
    const Qid qubits_n = shot_state_ptr.qubits_n;
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *stab = decomp_ptr.get_stab_bits_ptr();
    const Bit *destab = decomp_ptr.get_destab_bits_ptr();
    const Phs decomp_phase = *decomp_ptr.get_phase_ptr();
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const Bst bst = *entries_ptr.get_bst_ptr(entry_i);
    const Amp amp = *entries_ptr.get_amp_ptr(entry_i);
    Bst &bst0 = *entries_ptr.get_half0_bst_ptr(entry_i);
    Amp &amp0 = *entries_ptr.get_half0_amp_ptr(entry_i);
    Bst &bst1 = *entries_ptr.get_half1_bst_ptr(entry_i);
    Amp &amp1 = *entries_ptr.get_half1_amp_ptr(entry_i);
    constexpr Flt coef = M_SQRT1_2; // sqrt(1/2);

    // 取 bst 中的第 pivot 那个 bit
    constexpr Bst bst_one = 1;
    const Bst bst_mask = bst_one << pivot;
    if (bst & bst_mask) {
        const Phs decomp_phase_inv = -decomp_phase;
        const Amp decomp_phase_amp = phase_to_amp(decomp_phase_inv);
        const Bit stab_sign = compute_sign(bst, stab, qubits_n);
        const Flt stab_sign_amp = sign_to_flt(stab_sign);
        const Flt result0_amp = +1;
        const Flt result1_amp = -1;

        amp0 = amp * coef * decomp_phase_amp * stab_sign_amp * result0_amp;
        amp1 = amp * coef * decomp_phase_amp * stab_sign_amp * result1_amp;
        bst0 = bst ^ bits_to_bst(destab, qubits_n);
        bst1 = bst ^ bits_to_bst(destab, qubits_n);
    } else {
        amp0 = amp * coef;
        amp1 = amp * coef;
        bst0 = bst;
        bst1 = bst;
    }
}

static __device__
void op_compute_measure_entries(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Bst const entry_i = dims_idx.get<1>();

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

    // check pivot
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot == NullPivot) {
        compute_measure_entries_situation0(shot_state_ptr, entry_i);
    } else {
        compute_measure_entries_situation1(shot_state_ptr, entry_i, pivot);
    }
}

static __host__
void cuda_compute_measure_entries(cudaStream_t const &stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Eid entries_m = shots_state_ptr.entries_m;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_measure_entries>
        (stream, shots_state_ptr, dimsof(shots_n, entries_m / 2));
}


static __device__
void op_compute_measure_probs_situation0(const EntriesPtr entries_ptr, const Bit result) {
    const Eid entries_n = *entries_ptr.get_entries_n_ptr();
    Eid &entries_n_new = *(!result ? entries_ptr.get_half0_entries_n_ptr() : entries_ptr.get_half1_entries_n_ptr());
    Flt &norm = *(!result ? entries_ptr.get_half0_norm_ptr() : entries_ptr.get_half1_norm_ptr());
    Bst *bsts = !result ? entries_ptr.get_half0_bsts_ptr() : entries_ptr.get_half1_bsts_ptr();
    Amp *amps = !result ? entries_ptr.get_half0_amps_ptr() : entries_ptr.get_half1_amps_ptr();

    norm = 0;
    entries_n_new = 0;
    for (Eid entry_i = 0; entry_i < entries_n; ++entry_i) {
        const Bst bst = bsts[entry_i];
        const Amp amp = amps[entry_i];

        constexpr Amp amp_zero = 0;
        if (amp == amp_zero) continue;

        const Eid amp_j = entries_n_new;
        bsts[amp_j] = bst;
        amps[amp_j] = amp;
        norm += cuda::std::norm(amp);
        entries_n_new++;
    }
}

static __device__
void op_compute_measure_probs_situation1(const EntriesPtr entries_ptr, const Bit result) {
    const Eid entries_n = *entries_ptr.get_entries_n_ptr();
    Eid &entries_n_new = *(!result ? entries_ptr.get_half0_entries_n_ptr() : entries_ptr.get_half1_entries_n_ptr());
    Flt &norm = *(!result ? entries_ptr.get_half0_norm_ptr() : entries_ptr.get_half1_norm_ptr());
    Bst *bsts = !result ? entries_ptr.get_half0_bsts_ptr() : entries_ptr.get_half1_bsts_ptr();
    Amp *amps = !result ? entries_ptr.get_half0_amps_ptr() : entries_ptr.get_half1_amps_ptr();

    entries_n_new = 0;
    for (Eid entry_i = 0; entry_i < entries_n; ++entry_i) {
        const Bst bst = bsts[entry_i];
        const Amp amp = amps[entry_i];

        Eid amp_j = 0;
        for (; amp_j < entries_n_new; ++amp_j) {
            if (bsts[amp_j] == bst) {
                amps[amp_j] += amp;
                break;
            }
        }

        if (amp_j == entries_n_new) {
            bsts[amp_j] = bst;
            amps[amp_j] = amp;
            entries_n_new++;
        }
    }

    norm = 0;
    for (int entry_i = 0; entry_i < entries_n_new; ++entry_i) {
        const Amp amp = amps[entry_i];
        norm += cuda::std::norm(amp);
    }
}

static __device__
void op_compute_measure_probs(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Bit const result = dims_idx.get<1>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    // check error
    const Err err = *work_ptr.get_err_ptr();
    if (err) return; // 这个 shot 已经失败，不进行计算

    // check pivot
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    pivot == NullPivot
        ? op_compute_measure_probs_situation0(entries_ptr, result)
        : op_compute_measure_probs_situation1(entries_ptr, result);
}

static __host__
void cuda_compute_measure_probs(cudaStream_t const &stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 2, op_compute_measure_probs>
        (stream, shots_state_ptr, dimsof(shots_n, 2));
}


enum class SampleMode {
    Random = -1,
    Maximum = -2,
    Desire0 = 0,
    Desire1 = 1,
};

template<SampleMode mode = SampleMode::Random>
static __device__
void op_compute_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<1> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();

    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    // check error
    const Err err = *work_ptr.get_err_ptr();
    if (err) return; // 这个 shot 已经失败，不进行计算

    // normalize probs
    const Flt norm0 = *entries_ptr.get_half0_norm_ptr();
    const Flt norm1 = *entries_ptr.get_half1_norm_ptr();
    const Flt total = norm0 + norm1;
    const Flt prob0 = norm0 / total;
    const Flt prob1 = norm1 / total;

    // random choice
    Bit result;
    switch (mode) {
        case SampleMode::Desire0:
            result = false;
            break;
        case SampleMode::Desire1:
            result = true;
            break;
        case SampleMode::Maximum:
            result = prob0 <= prob1;
        default:
            curandState *rand_state_ptr = work_ptr.get_rand_state_ptr();
            result = prob0 < curand_uniform_double(rand_state_ptr);
    }

    // save result
    Flt &result_prob = *work_ptr.get_flt_ptr();
    Int &result_value = *work_ptr.get_int_ptr();
    if (!result) {
        result_prob = prob0;
        result_value = 0;
    } else {
        result_prob = prob1;
        result_value = 1;
    }
}

template<SampleMode mode = SampleMode::Random>
static __host__
void cuda_compute_measure_result(cudaStream_t const &stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ShotsStatePtr, 1, op_compute_measure_result<mode>>
        (stream, shots_state_ptr, dimsof(shots_n));
}


static __device__
void op_apply_measure_result(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Eid const entry_i = dims_idx.get<1>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const EntriesPtr entries_ptr = shot_state_ptr.get_entries_ptr();
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    // check error
    const Err err = *work_ptr.get_err_ptr();
    if (err) return; // 这个 shot 已经失败，不进行计算

    // load result
    const Bit result_value = *work_ptr.get_int_ptr();
    const Flt result_norm = !result_value
        ? *entries_ptr.get_half0_norm_ptr()
        : *entries_ptr.get_half1_norm_ptr();

    // apply result
    Eid &entries_n = *entries_ptr.get_entries_n_ptr();
    const Eid entries_n_new = *(!result_value
        ? entries_ptr.get_half0_entries_n_ptr()
        : entries_ptr.get_half1_entries_n_ptr());
    const Bst *bsts_src = !result_value
        ? entries_ptr.get_half0_bsts_ptr()
        : entries_ptr.get_half1_bsts_ptr();
    const Amp *amps_src = !result_value
        ? entries_ptr.get_half0_amps_ptr()
        : entries_ptr.get_half1_amps_ptr();
    Bst *bsts_dst = entries_ptr.get_bsts_ptr();
    Amp *amps_dst = entries_ptr.get_amps_ptr();

    if (entry_i == 0) entries_n = entries_n_new; // 更新 entries_n（仅线程 0 更新，避免写入冲突）
    if (entry_i >= entries_n_new) return; // 线程超出了 amp_n 的范围，不进行计算

    bsts_dst[entry_i] = bsts_src[entry_i];
    amps_dst[entry_i] = amps_src[entry_i] / cuda::std::sqrt(result_norm);
}

static __host__
void cuda_apply_measure_result(cudaStream_t const &stream, ShotsStatePtr const shots_state_ptr) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Eid entries_m = shots_state_ptr.entries_m;
    cuda_dims_op<ShotsStatePtr, 2, op_apply_measure_result>
        (stream, shots_state_ptr, dimsof(shots_n, entries_m / 2));
}


static __device__
void op_change_measure_basis_rowsum(const ShotsStatePtr shots_state_ptr, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    Sid const row_i = dims_idx.get<1>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

    // check pivot
    const DecompPtr decomp_ptr = shot_state_ptr.get_decomp_ptr();
    const Bit *decomp_bits = decomp_ptr.get_bits_ptr();
    const Qid pivot = *decomp_ptr.get_pivot_ptr();
    if (pivot == NullPivot)
        return; // situation0 不需要改 basis

    // update table
    const TablePtr table_ptr = shot_state_ptr.get_table_ptr();
    const Qid qubits_n = table_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    if (row_i == pivot || row_i == qubits_n + pivot)
        return; // skip the pivot row
    if (!decomp_bits[(row_i + qubits_n) % rows_n])
        return; // skip the non-xy row

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
void cuda_change_measure_basis_rowsum(cudaStream_t const &stream, ShotsStatePtr const shots_state_ptr) {
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
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Int result_value = *work_ptr.get_int_ptr();

    // update table
    const TablePtr table_ptr = shot_state_ptr.get_table_ptr();
    const Qid qubits_n = table_ptr.qubits_n;
    const Qid cols_n = 2 * qubits_n;
    const TableRowPtr destab_pivot_row_ptr = table_ptr.get_row_ptr(pivot);
    const TableRowPtr stab_pivot_row_ptr = table_ptr.get_row_ptr(qubits_n + pivot);

    // set destab pivot = stab pivot
    for (Qid col_i = 0; col_i < cols_n; ++col_i)
        *destab_pivot_row_ptr.get_pauli_ptr().get_bit_ptr(col_i) =
            *stab_pivot_row_ptr.get_pauli_ptr().get_bit_ptr(col_i);
    *destab_pivot_row_ptr.get_sign_ptr() =
        *stab_pivot_row_ptr.get_sign_ptr();

    // set stab pivot = Z (sign=result)
    for (Qid col_i = 0; col_i < cols_n; ++col_i)
        *stab_pivot_row_ptr.get_pauli_ptr().get_bit_ptr(col_i) = false;
    *stab_pivot_row_ptr.get_pauli_ptr().get_bit_ptr(qubits_n + args.target) = true;
    *stab_pivot_row_ptr.get_sign_ptr() = result_value;
}

static __host__
void cuda_change_measure_basis_pivot(cudaStream_t const &stream, ShotsStatePtr const shots_state_ptr, const Qid target) {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsApplyMeasureBasisPivot, 1, op_change_measure_basis_pivot>
        (stream, {shots_state_ptr, target}, dimsof(shots_n));
}


struct ArgsAssignOperation {
    const ShotsStatePtr shots_state_ptr;
    const Qid target;
    const Bit value;
};

static __device__
void op_apply_reset(const ArgsAssignOperation args, const DimsIdx<2> dims_idx) {
    Sid const shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    const Bit result_value = *work_ptr.get_int_ptr();
    if (result_value != args.value)
        op_apply_gate1<op_apply_x>({shots_state_ptr, args.target}, dims_idx);
}

static __host__
void cuda_apply_reset(
    cudaStream_t const &stream,
    ShotsStatePtr const shots_state_ptr,
    const Qid target,
    const Bit value
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsAssignOperation, 2, op_apply_reset>
        (stream, {shots_state_ptr, target, value}, dimsof(shots_n, rows_n));
}


void Simulator::apply_measure(const Qid target) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_entries(stream, shots_state_ptr);
    cuda_compute_measure_probs(stream, shots_state_ptr);
    cuda_compute_measure_result(stream, shots_state_ptr);
    cuda_apply_measure_result(stream, shots_state_ptr);

    cuda_change_measure_basis_rowsum(stream, shots_state_ptr);
    cuda_change_measure_basis_pivot(stream, shots_state_ptr, target);
}

void Simulator::apply_desire(const Qid target, const Bit value) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_entries(stream, shots_state_ptr);
    cuda_compute_measure_probs(stream, shots_state_ptr);
    !value
        ? cuda_compute_measure_result<SampleMode::Desire0>(stream, shots_state_ptr)
        : cuda_compute_measure_result<SampleMode::Desire1>(stream, shots_state_ptr);
    cuda_apply_measure_result(stream, shots_state_ptr);

    cuda_change_measure_basis_rowsum(stream, shots_state_ptr);
    cuda_change_measure_basis_pivot(stream, shots_state_ptr, target);
}

void Simulator::apply_reset(const Qid target, const Bit value) const noexcept {
    cuda_compute_decomposed_bits(stream, shots_state_ptr, target);
    cuda_compute_decomposed_phase(stream, shots_state_ptr);
    cuda_compute_decomp_pivot(stream, shots_state_ptr);

    cuda_compute_measure_entries(stream, shots_state_ptr);
    cuda_compute_measure_probs(stream, shots_state_ptr);
    cuda_compute_measure_result(stream, shots_state_ptr);
    cuda_apply_measure_result(stream, shots_state_ptr);

    cuda_change_measure_basis_rowsum(stream, shots_state_ptr);
    cuda_change_measure_basis_pivot(stream, shots_state_ptr, target);

    cuda_apply_reset(stream, shots_state_ptr, target, value);
}
