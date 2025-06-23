#ifndef STN_CUDA_DATASTRUCT_CUH
#define STN_CUDA_DATASTRUCT_CUH

#include "./datatype.cuh"

namespace StnCuda {

template<typename Item, typename... Args>
static __device__ __host__
Item max(Item item0, Args... args) {
    if constexpr (sizeof...(Args) == 0) {
        return item0;
    } else {
        Item item1 = max(args...);
        return item0 > item1 ? item0 : item1;
    }
}

static __device__ __host__
size_t compute_pad_bytes_n(const size_t offset_bytes_n, const size_t align_bytes_n) {
    return align_bytes_n - offset_bytes_n % align_bytes_n;
}


struct PauliRowArgs {
    Qid qubits_n;
    
    __device__ __host__
    size_t get_bit_size_bytes_n() const {
        return sizeof(Bit);
    }
    
    __device__ __host__
    size_t get_bit_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_bit_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_bit_size_bytes_n(),
            get_bit_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_bits_size_bytes_n() const {
        return 
            2 * qubits_n * get_bit_size_bytes_n() +
            2 * qubits_n * get_bit_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bits_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_bits_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_bits_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_bit_offset_bytes_n(Qid bit_i) const {
        return get_bits_offset_bytes_n() +
            bit_i * get_bit_size_bytes_n() +
            bit_i * get_bit_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_bits_pad_bytes_n() +
            get_bits_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_bits_align_bytes_n());
    }
};

struct PauliRowPtr : PauliRowArgs {
    char *ptr;
    
    __device__ __host__
    Bit *get_bits_ptr() const {
        const size_t offset = get_bits_offset_bytes_n();
        return reinterpret_cast<Bit *>(ptr + offset);
    }
    
    __device__ __host__
    Bit *get_bit_ptr(const Qid bit_i) const {
        return get_bits_ptr() + bit_i;
    }
    
    __device__ __host__
    Bit *get_x_ptr(const Qid qubit_i) const {
        return get_bit_ptr(qubit_i);
    }
    
    __device__ __host__
    Bit *get_z_ptr(const Qid qubit_i) const {
        return get_bit_ptr(qubits_n + qubit_i);
    }
};

struct TableRowArgs {
    Qid qubits_n;
    
    __device__ __host__
    size_t get_pauli_size_bytes_n() const {
        return PauliRowArgs{qubits_n}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_pauli_align_bytes_n() const {
        return PauliRowArgs{qubits_n}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_pauli_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_pauli_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_sign_size_bytes_n() const {
        return sizeof(Bit);
    }
    
    __device__ __host__
    size_t get_sign_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_sign_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_pauli_offset_bytes_n() +
            get_pauli_size_bytes_n(),
            get_sign_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_sign_offset_bytes_n() const {
        return 
            get_pauli_offset_bytes_n() +
            get_pauli_size_bytes_n() +
            get_sign_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_pauli_pad_bytes_n() +
            get_pauli_size_bytes_n() +
            get_sign_pad_bytes_n() +
            get_sign_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_pauli_align_bytes_n(),
            get_sign_align_bytes_n());
    }
};

struct TableRowPtr : TableRowArgs {
    char *ptr;
    
    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        const size_t offset = get_pauli_offset_bytes_n();
        return {qubits_n, ptr + offset};
    }
    
    __device__ __host__
    Bit *get_sign_ptr() const {
        const size_t offset = get_sign_offset_bytes_n();
        return reinterpret_cast<Bit *>(ptr + offset);
    }
};

struct TableArgs {
    Qid qubits_n;
    
    __device__ __host__
    size_t get_row_size_bytes_n() const {
        return TableRowArgs{qubits_n}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_row_align_bytes_n() const {
        return TableRowArgs{qubits_n}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_row_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_row_size_bytes_n(),
            get_row_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_rows_size_bytes_n() const {
        return 
            2 * qubits_n * get_row_size_bytes_n() +
            2 * qubits_n * get_row_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_rows_align_bytes_n() const {
        return TableRowArgs{qubits_n}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_rows_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_rows_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_row_offset_bytes_n(Qid row_i) const {
        return get_rows_offset_bytes_n() +
            row_i * get_row_size_bytes_n() +
            row_i * get_row_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_rows_pad_bytes_n() +
            get_rows_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_rows_align_bytes_n());
    }
};

struct TablePtr : TableArgs {
    char *ptr;
    
    __device__ __host__
    TableRowPtr get_row_ptr(const Qid row_i) const {
        const size_t offset = get_row_offset_bytes_n(row_i);
        return {qubits_n, ptr + offset};
    }
};

struct DecompArgs {
    Qid qubits_n;
    
    __device__ __host__
    size_t get_bit_size_bytes_n() const {
        return sizeof(Bit);
    }
    
    __device__ __host__
    size_t get_bit_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_bit_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_bit_size_bytes_n(),
            get_bit_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_bits_size_bytes_n() const {
        return 
            2 * qubits_n * get_bit_size_bytes_n() +
            2 * qubits_n * get_bit_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bits_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_bits_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_bits_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_bit_offset_bytes_n(Qid bit_i) const {
        return get_bits_offset_bytes_n() +
            bit_i * get_bit_size_bytes_n() +
            bit_i * get_bit_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_pauli_size_bytes_n() const {
        return PauliRowArgs{qubits_n}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_pauli_align_bytes_n() const {
        return PauliRowArgs{qubits_n}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_pauli_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_bits_offset_bytes_n() +
            get_bits_size_bytes_n(),
            get_pauli_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_pauli_offset_bytes_n() const {
        return 
            get_bits_offset_bytes_n() +
            get_bits_size_bytes_n() +
            get_pauli_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_phase_size_bytes_n() const {
        return sizeof(Phs);
    }
    
    __device__ __host__
    size_t get_phase_align_bytes_n() const {
        return alignof(Phs);
    }
    
    __device__ __host__
    size_t get_phase_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_pauli_offset_bytes_n() +
            get_pauli_size_bytes_n(),
            get_phase_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_phase_offset_bytes_n() const {
        return 
            get_pauli_offset_bytes_n() +
            get_pauli_size_bytes_n() +
            get_phase_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_pivot_size_bytes_n() const {
        return sizeof(Qid);
    }
    
    __device__ __host__
    size_t get_pivot_align_bytes_n() const {
        return alignof(Qid);
    }
    
    __device__ __host__
    size_t get_pivot_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_phase_offset_bytes_n() +
            get_phase_size_bytes_n(),
            get_pivot_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_pivot_offset_bytes_n() const {
        return 
            get_phase_offset_bytes_n() +
            get_phase_size_bytes_n() +
            get_pivot_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_bits_pad_bytes_n() +
            get_bits_size_bytes_n() +
            get_pauli_pad_bytes_n() +
            get_pauli_size_bytes_n() +
            get_phase_pad_bytes_n() +
            get_phase_size_bytes_n() +
            get_pivot_pad_bytes_n() +
            get_pivot_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_bits_align_bytes_n(),
            get_pauli_align_bytes_n(),
            get_phase_align_bytes_n(),
            get_pivot_align_bytes_n());
    }
};

struct DecompPtr : DecompArgs {
    char *ptr;
    
    __device__ __host__
    Bit *get_bits_ptr() const {
        const size_t offset = get_bits_offset_bytes_n();
        return reinterpret_cast<Bit *>(ptr + offset);
    }
    
    __device__ __host__
    Bit *get_bit_ptr(const Qid bit_i) const {
        return get_bits_ptr() + bit_i;
    }
    
    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        const size_t offset = get_pauli_offset_bytes_n();
        return {qubits_n, ptr + offset};
    }
    
    __device__ __host__
    Phs *get_phase_ptr() const {
        const size_t offset = get_phase_offset_bytes_n();
        return reinterpret_cast<Phs *>(ptr + offset);
    }
    
    __device__ __host__
    Qid *get_pivot_ptr() const {
        const size_t offset = get_pivot_offset_bytes_n();
        return reinterpret_cast<Qid *>(ptr + offset);
    }
    
    __device__ __host__
    Bit *get_destab_bits_ptr() const {
        return get_bits_ptr();
    }
    
    __device__ __host__
    Bit *get_destab_bit_ptr(const Qid qubit_i) const {
        return get_destab_bits_ptr() + qubit_i;
    }
    
    __device__ __host__
    Bit *get_stab_bits_ptr() const {
        return get_bits_ptr() + qubits_n;
    }
    
    __device__ __host__
    Bit *get_stab_bit_ptr(const Qid qubit_i) const {
        return get_stab_bits_ptr() + qubit_i;
    }
};

struct AmpsMapArgs {
    Qid qubits_n;
    Kid amps_m;
    
    __device__ __host__
    size_t get_amps_n_size_bytes_n() const {
        return sizeof(Kid);
    }
    
    __device__ __host__
    size_t get_amps_n_align_bytes_n() const {
        return alignof(Kid);
    }
    
    __device__ __host__
    size_t get_amps_n_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_amps_n_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_aid_size_bytes_n() const {
        return sizeof(Aid);
    }
    
    __device__ __host__
    size_t get_aid_align_bytes_n() const {
        return alignof(Aid);
    }
    
    __device__ __host__
    size_t get_aid_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_aid_size_bytes_n(),
            get_aid_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_aids_size_bytes_n() const {
        return 
            amps_m * get_aid_size_bytes_n() +
            amps_m * get_aid_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_aids_align_bytes_n() const {
        return alignof(Aid);
    }
    
    __device__ __host__
    size_t get_aids_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_amps_n_offset_bytes_n() +
            get_amps_n_size_bytes_n(),
            get_aids_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_aids_offset_bytes_n() const {
        return 
            get_amps_n_offset_bytes_n() +
            get_amps_n_size_bytes_n() +
            get_aids_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_aid_offset_bytes_n(Kid amp_i) const {
        return get_aids_offset_bytes_n() +
            amp_i * get_aid_size_bytes_n() +
            amp_i * get_aid_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_amp_size_bytes_n() const {
        return sizeof(Amp);
    }
    
    __device__ __host__
    size_t get_amp_align_bytes_n() const {
        return alignof(Amp);
    }
    
    __device__ __host__
    size_t get_amp_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_amp_size_bytes_n(),
            get_amp_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_amps_size_bytes_n() const {
        return 
            amps_m * get_amp_size_bytes_n() +
            amps_m * get_amp_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_amps_align_bytes_n() const {
        return alignof(Amp);
    }
    
    __device__ __host__
    size_t get_amps_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_aids_offset_bytes_n() +
            get_aids_size_bytes_n(),
            get_amps_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_amps_offset_bytes_n() const {
        return 
            get_aids_offset_bytes_n() +
            get_aids_size_bytes_n() +
            get_amps_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_amp_offset_bytes_n(Kid amp_i) const {
        return get_amps_offset_bytes_n() +
            amp_i * get_amp_size_bytes_n() +
            amp_i * get_amp_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_amps_n_pad_bytes_n() +
            get_amps_n_size_bytes_n() +
            get_aids_pad_bytes_n() +
            get_aids_size_bytes_n() +
            get_amps_pad_bytes_n() +
            get_amps_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_amps_n_align_bytes_n(),
            get_aids_align_bytes_n(),
            get_amps_align_bytes_n());
    }
};

struct AmpsMapPtr : AmpsMapArgs {
    char *ptr;
    
    __device__ __host__
    Kid *get_amps_n_ptr() const {
        const size_t offset = get_amps_n_offset_bytes_n();
        return reinterpret_cast<Kid *>(ptr + offset);
    }
    
    __device__ __host__
    Aid *get_aids_ptr() const {
        const size_t offset = get_aids_offset_bytes_n();
        return reinterpret_cast<Aid *>(ptr + offset);
    }
    
    __device__ __host__
    Aid *get_aid_ptr(const Kid amp_i) const {
        return get_aids_ptr() + amp_i;
    }
    
    __device__ __host__
    Amp *get_amps_ptr() const {
        const size_t offset = get_amps_offset_bytes_n();
        return reinterpret_cast<Amp *>(ptr + offset);
    }
    
    __device__ __host__
    Amp *get_amp_ptr(const Kid amp_i) const {
        return get_amps_ptr() + amp_i;
    }
    
    __device__ __host__
    Aid *get_half0_aids_ptr() const {
        return get_aids_ptr();
    }
    
    __device__ __host__
    Aid *get_half0_aid_ptr(const Kid amp_i) const {
        return get_half0_aids_ptr() + amp_i;
    }
    
    __device__ __host__
    Amp *get_half0_amps_ptr() const {
        return get_amps_ptr();
    }
    
    __device__ __host__
    Amp *get_half0_amp_ptr(const Kid amp_i) const {
        return get_half0_amps_ptr() + amp_i;
    }
    
    __device__ __host__
    Aid *get_half1_aids_ptr() const {
        return get_aids_ptr() + amps_m / 2;
    }
    
    __device__ __host__
    Aid *get_half1_aid_ptr(const Kid amp_i) const {
        return get_half1_aids_ptr() + amp_i;
    }
    
    __device__ __host__
    Amp *get_half1_amps_ptr() const {
        return get_amps_ptr() + amps_m / 2;
    }
    
    __device__ __host__
    Amp *get_half1_amp_ptr(const Kid amp_i) const {
        return get_half1_amps_ptr() + amp_i;
    }
};

struct ResultsArgs {
    Rid results_m;
    
    __device__ __host__
    size_t get_rand_state_size_bytes_n() const {
        return sizeof(curandState);
    }
    
    __device__ __host__
    size_t get_rand_state_align_bytes_n() const {
        return alignof(curandState);
    }
    
    __device__ __host__
    size_t get_rand_state_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_rand_state_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_results_n_size_bytes_n() const {
        return sizeof(Rid);
    }
    
    __device__ __host__
    size_t get_results_n_align_bytes_n() const {
        return alignof(Rid);
    }
    
    __device__ __host__
    size_t get_results_n_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_rand_state_offset_bytes_n() +
            get_rand_state_size_bytes_n(),
            get_results_n_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_results_n_offset_bytes_n() const {
        return 
            get_rand_state_offset_bytes_n() +
            get_rand_state_size_bytes_n() +
            get_results_n_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_prob_size_bytes_n() const {
        return sizeof(Flt);
    }
    
    __device__ __host__
    size_t get_prob_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_prob_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_prob_size_bytes_n(),
            get_prob_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_probs_size_bytes_n() const {
        return 
            results_m * get_prob_size_bytes_n() +
            results_m * get_prob_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_probs_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_probs_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_results_n_offset_bytes_n() +
            get_results_n_size_bytes_n(),
            get_probs_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_probs_offset_bytes_n() const {
        return 
            get_results_n_offset_bytes_n() +
            get_results_n_size_bytes_n() +
            get_probs_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_prob_offset_bytes_n(Rid result_i) const {
        return get_probs_offset_bytes_n() +
            result_i * get_prob_size_bytes_n() +
            result_i * get_prob_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bit_size_bytes_n() const {
        return sizeof(Bit);
    }
    
    __device__ __host__
    size_t get_bit_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_bit_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_bit_size_bytes_n(),
            get_bit_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_bits_size_bytes_n() const {
        return 
            results_m * get_bit_size_bytes_n() +
            results_m * get_bit_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bits_align_bytes_n() const {
        return alignof(Bit);
    }
    
    __device__ __host__
    size_t get_bits_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_probs_offset_bytes_n() +
            get_probs_size_bytes_n(),
            get_bits_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_bits_offset_bytes_n() const {
        return 
            get_probs_offset_bytes_n() +
            get_probs_size_bytes_n() +
            get_bits_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bit_offset_bytes_n(Rid result_i) const {
        return get_bits_offset_bytes_n() +
            result_i * get_bit_size_bytes_n() +
            result_i * get_bit_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_rand_state_pad_bytes_n() +
            get_rand_state_size_bytes_n() +
            get_results_n_pad_bytes_n() +
            get_results_n_size_bytes_n() +
            get_probs_pad_bytes_n() +
            get_probs_size_bytes_n() +
            get_bits_pad_bytes_n() +
            get_bits_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_rand_state_align_bytes_n(),
            get_results_n_align_bytes_n(),
            get_probs_align_bytes_n(),
            get_bits_align_bytes_n());
    }
};

struct ResultsPtr : ResultsArgs {
    char *ptr;
    
    __device__ __host__
    curandState *get_rand_state_ptr() const {
        const size_t offset = get_rand_state_offset_bytes_n();
        return reinterpret_cast<curandState *>(ptr + offset);
    }
    
    __device__ __host__
    Rid *get_results_n_ptr() const {
        const size_t offset = get_results_n_offset_bytes_n();
        return reinterpret_cast<Rid *>(ptr + offset);
    }
    
    __device__ __host__
    Flt *get_probs_ptr() const {
        const size_t offset = get_probs_offset_bytes_n();
        return reinterpret_cast<Flt *>(ptr + offset);
    }
    
    __device__ __host__
    Flt *get_prob_ptr(const Rid result_i) const {
        return get_probs_ptr() + result_i;
    }
    
    __device__ __host__
    Bit *get_bits_ptr() const {
        const size_t offset = get_bits_offset_bytes_n();
        return reinterpret_cast<Bit *>(ptr + offset);
    }
    
    __device__ __host__
    Bit *get_bit_ptr(const Rid result_i) const {
        return get_bits_ptr() + result_i;
    }
};

struct ShotStateArgs {
    Qid qubits_n;
    Kid amps_m;
    Kid results_m;
    
    __device__ __host__
    size_t get_table_size_bytes_n() const {
        return TableArgs{qubits_n}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_table_align_bytes_n() const {
        return TableArgs{qubits_n}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_table_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_table_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_decomp_size_bytes_n() const {
        return DecompArgs{qubits_n}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_decomp_align_bytes_n() const {
        return DecompArgs{qubits_n}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_decomp_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_table_offset_bytes_n() +
            get_table_size_bytes_n(),
            get_decomp_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_decomp_offset_bytes_n() const {
        return 
            get_table_offset_bytes_n() +
            get_table_size_bytes_n() +
            get_decomp_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_amps_size_bytes_n() const {
        return AmpsMapArgs{qubits_n, amps_m}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_amps_align_bytes_n() const {
        return AmpsMapArgs{qubits_n, amps_m}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_amps_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_decomp_offset_bytes_n() +
            get_decomp_size_bytes_n(),
            get_amps_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_amps_offset_bytes_n() const {
        return 
            get_decomp_offset_bytes_n() +
            get_decomp_size_bytes_n() +
            get_amps_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_results_size_bytes_n() const {
        return ResultsArgs{results_m}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_results_align_bytes_n() const {
        return ResultsArgs{results_m}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_results_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_amps_offset_bytes_n() +
            get_amps_size_bytes_n(),
            get_results_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_results_offset_bytes_n() const {
        return 
            get_amps_offset_bytes_n() +
            get_amps_size_bytes_n() +
            get_results_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_table_pad_bytes_n() +
            get_table_size_bytes_n() +
            get_decomp_pad_bytes_n() +
            get_decomp_size_bytes_n() +
            get_amps_pad_bytes_n() +
            get_amps_size_bytes_n() +
            get_results_pad_bytes_n() +
            get_results_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_table_align_bytes_n(),
            get_decomp_align_bytes_n(),
            get_amps_align_bytes_n(),
            get_results_align_bytes_n());
    }
};

struct ShotStatePtr : ShotStateArgs {
    char *ptr;
    
    __device__ __host__
    TablePtr get_table_ptr() const {
        const size_t offset = get_table_offset_bytes_n();
        return {qubits_n, ptr + offset};
    }
    
    __device__ __host__
    DecompPtr get_decomp_ptr() const {
        const size_t offset = get_decomp_offset_bytes_n();
        return {qubits_n, ptr + offset};
    }
    
    __device__ __host__
    AmpsMapPtr get_amps_ptr() const {
        const size_t offset = get_amps_offset_bytes_n();
        return {qubits_n, amps_m, ptr + offset};
    }
    
    __device__ __host__
    ResultsPtr get_results_ptr() const {
        const size_t offset = get_results_offset_bytes_n();
        return {results_m, ptr + offset};
    }
};

struct ShotsStateArgs {
    Sid shots_n;
    Qid qubits_n;
    Kid amps_m;
    Kid results_m;
    
    __device__ __host__
    size_t get_shot_size_bytes_n() const {
        return ShotStateArgs{qubits_n, amps_m, results_m}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_shot_align_bytes_n() const {
        return ShotStateArgs{qubits_n, amps_m, results_m}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_shot_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_shot_size_bytes_n(),
            get_shot_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_shots_size_bytes_n() const {
        return 
            shots_n * get_shot_size_bytes_n() +
            shots_n * get_shot_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_shots_align_bytes_n() const {
        return ShotStateArgs{qubits_n, amps_m, results_m}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_shots_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_shots_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_shot_offset_bytes_n(Sid shot_i) const {
        return get_shots_offset_bytes_n() +
            shot_i * get_shot_size_bytes_n() +
            shot_i * get_shot_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_shots_pad_bytes_n() +
            get_shots_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_shots_align_bytes_n());
    }
};

struct ShotsStatePtr : ShotsStateArgs {
    char *ptr;
    
    __device__ __host__
    ShotStatePtr get_shot_ptr(const Sid shot_i) const {
        const size_t offset = get_shot_offset_bytes_n(shot_i);
        return {qubits_n, amps_m, results_m, ptr + offset};
    }
};

}

#endif
