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

struct EntriesArgs {
    Qid qubits_n;
    Eid entries_m;
    
    __device__ __host__
    size_t get_entries_n_size_bytes_n() const {
        return sizeof(Eid);
    }
    
    __device__ __host__
    size_t get_entries_n_align_bytes_n() const {
        return alignof(Eid);
    }
    
    __device__ __host__
    size_t get_entries_n_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_entries_n_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_bst_size_bytes_n() const {
        return sizeof(Bst);
    }
    
    __device__ __host__
    size_t get_bst_align_bytes_n() const {
        return alignof(Bst);
    }
    
    __device__ __host__
    size_t get_bst_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_bst_size_bytes_n(),
            get_bst_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_bsts_size_bytes_n() const {
        return 
            entries_m * get_bst_size_bytes_n() +
            entries_m * get_bst_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bsts_align_bytes_n() const {
        return alignof(Bst);
    }
    
    __device__ __host__
    size_t get_bsts_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_entries_n_offset_bytes_n() +
            get_entries_n_size_bytes_n(),
            get_bsts_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_bsts_offset_bytes_n() const {
        return 
            get_entries_n_offset_bytes_n() +
            get_entries_n_size_bytes_n() +
            get_bsts_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_bst_offset_bytes_n(Eid entry_i) const {
        return get_bsts_offset_bytes_n() +
            entry_i * get_bst_size_bytes_n() +
            entry_i * get_bst_pad_bytes_n();
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
            entries_m * get_amp_size_bytes_n() +
            entries_m * get_amp_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_amps_align_bytes_n() const {
        return alignof(Amp);
    }
    
    __device__ __host__
    size_t get_amps_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_bsts_offset_bytes_n() +
            get_bsts_size_bytes_n(),
            get_amps_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_amps_offset_bytes_n() const {
        return 
            get_bsts_offset_bytes_n() +
            get_bsts_size_bytes_n() +
            get_amps_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_amp_offset_bytes_n(Eid entry_i) const {
        return get_amps_offset_bytes_n() +
            entry_i * get_amp_size_bytes_n() +
            entry_i * get_amp_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_half0_entries_n_size_bytes_n() const {
        return sizeof(Eid);
    }
    
    __device__ __host__
    size_t get_half0_entries_n_align_bytes_n() const {
        return alignof(Eid);
    }
    
    __device__ __host__
    size_t get_half0_entries_n_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_amps_offset_bytes_n() +
            get_amps_size_bytes_n(),
            get_half0_entries_n_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_half0_entries_n_offset_bytes_n() const {
        return 
            get_amps_offset_bytes_n() +
            get_amps_size_bytes_n() +
            get_half0_entries_n_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_half1_entries_n_size_bytes_n() const {
        return sizeof(Eid);
    }
    
    __device__ __host__
    size_t get_half1_entries_n_align_bytes_n() const {
        return alignof(Eid);
    }
    
    __device__ __host__
    size_t get_half1_entries_n_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_half0_entries_n_offset_bytes_n() +
            get_half0_entries_n_size_bytes_n(),
            get_half1_entries_n_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_half1_entries_n_offset_bytes_n() const {
        return 
            get_half0_entries_n_offset_bytes_n() +
            get_half0_entries_n_size_bytes_n() +
            get_half1_entries_n_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_half0_norm_size_bytes_n() const {
        return sizeof(Flt);
    }
    
    __device__ __host__
    size_t get_half0_norm_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_half0_norm_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_half1_entries_n_offset_bytes_n() +
            get_half1_entries_n_size_bytes_n(),
            get_half0_norm_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_half0_norm_offset_bytes_n() const {
        return 
            get_half1_entries_n_offset_bytes_n() +
            get_half1_entries_n_size_bytes_n() +
            get_half0_norm_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_half1_norm_size_bytes_n() const {
        return sizeof(Flt);
    }
    
    __device__ __host__
    size_t get_half1_norm_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_half1_norm_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_half0_norm_offset_bytes_n() +
            get_half0_norm_size_bytes_n(),
            get_half1_norm_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_half1_norm_offset_bytes_n() const {
        return 
            get_half0_norm_offset_bytes_n() +
            get_half0_norm_size_bytes_n() +
            get_half1_norm_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_entries_n_pad_bytes_n() +
            get_entries_n_size_bytes_n() +
            get_bsts_pad_bytes_n() +
            get_bsts_size_bytes_n() +
            get_amps_pad_bytes_n() +
            get_amps_size_bytes_n() +
            get_half0_entries_n_pad_bytes_n() +
            get_half0_entries_n_size_bytes_n() +
            get_half1_entries_n_pad_bytes_n() +
            get_half1_entries_n_size_bytes_n() +
            get_half0_norm_pad_bytes_n() +
            get_half0_norm_size_bytes_n() +
            get_half1_norm_pad_bytes_n() +
            get_half1_norm_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_entries_n_align_bytes_n(),
            get_bsts_align_bytes_n(),
            get_amps_align_bytes_n(),
            get_half0_entries_n_align_bytes_n(),
            get_half1_entries_n_align_bytes_n(),
            get_half0_norm_align_bytes_n(),
            get_half1_norm_align_bytes_n());
    }
};

struct EntriesPtr : EntriesArgs {
    char *ptr;
    
    __device__ __host__
    Eid *get_entries_n_ptr() const {
        const size_t offset = get_entries_n_offset_bytes_n();
        return reinterpret_cast<Eid *>(ptr + offset);
    }
    
    __device__ __host__
    Bst *get_bsts_ptr() const {
        const size_t offset = get_bsts_offset_bytes_n();
        return reinterpret_cast<Bst *>(ptr + offset);
    }
    
    __device__ __host__
    Bst *get_bst_ptr(const Eid entry_i) const {
        return get_bsts_ptr() + entry_i;
    }
    
    __device__ __host__
    Amp *get_amps_ptr() const {
        const size_t offset = get_amps_offset_bytes_n();
        return reinterpret_cast<Amp *>(ptr + offset);
    }
    
    __device__ __host__
    Amp *get_amp_ptr(const Eid entry_i) const {
        return get_amps_ptr() + entry_i;
    }
    
    __device__ __host__
    Eid *get_half0_entries_n_ptr() const {
        const size_t offset = get_half0_entries_n_offset_bytes_n();
        return reinterpret_cast<Eid *>(ptr + offset);
    }
    
    __device__ __host__
    Eid *get_half1_entries_n_ptr() const {
        const size_t offset = get_half1_entries_n_offset_bytes_n();
        return reinterpret_cast<Eid *>(ptr + offset);
    }
    
    __device__ __host__
    Flt *get_half0_norm_ptr() const {
        const size_t offset = get_half0_norm_offset_bytes_n();
        return reinterpret_cast<Flt *>(ptr + offset);
    }
    
    __device__ __host__
    Flt *get_half1_norm_ptr() const {
        const size_t offset = get_half1_norm_offset_bytes_n();
        return reinterpret_cast<Flt *>(ptr + offset);
    }
    
    __device__ __host__
    Bst *get_half0_bsts_ptr() const {
        return get_bsts_ptr();
    }
    
    __device__ __host__
    Bst *get_half0_bst_ptr(const Eid entry_i) const {
        return get_half0_bsts_ptr() + entry_i;
    }
    
    __device__ __host__
    Amp *get_half0_amps_ptr() const {
        return get_amps_ptr();
    }
    
    __device__ __host__
    Amp *get_half0_amp_ptr(const Eid entry_i) const {
        return get_half0_amps_ptr() + entry_i;
    }
    
    __device__ __host__
    Bst *get_half1_bsts_ptr() const {
        return get_bsts_ptr() + entries_m / 2;
    }
    
    __device__ __host__
    Bst *get_half1_bst_ptr(const Eid entry_i) const {
        return get_half1_bsts_ptr() + entry_i;
    }
    
    __device__ __host__
    Amp *get_half1_amps_ptr() const {
        return get_amps_ptr() + entries_m / 2;
    }
    
    __device__ __host__
    Amp *get_half1_amp_ptr(const Eid entry_i) const {
        return get_half1_amps_ptr() + entry_i;
    }
};

struct WorkArgs {
    
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
    size_t get_err_size_bytes_n() const {
        return sizeof(Int);
    }
    
    __device__ __host__
    size_t get_err_align_bytes_n() const {
        return alignof(Int);
    }
    
    __device__ __host__
    size_t get_err_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_rand_state_offset_bytes_n() +
            get_rand_state_size_bytes_n(),
            get_err_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_err_offset_bytes_n() const {
        return 
            get_rand_state_offset_bytes_n() +
            get_rand_state_size_bytes_n() +
            get_err_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_int_size_bytes_n() const {
        return sizeof(Int);
    }
    
    __device__ __host__
    size_t get_int_align_bytes_n() const {
        return alignof(Int);
    }
    
    __device__ __host__
    size_t get_int_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_err_offset_bytes_n() +
            get_err_size_bytes_n(),
            get_int_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_int_offset_bytes_n() const {
        return 
            get_err_offset_bytes_n() +
            get_err_size_bytes_n() +
            get_int_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_flt_size_bytes_n() const {
        return sizeof(Flt);
    }
    
    __device__ __host__
    size_t get_flt_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_flt_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_int_offset_bytes_n() +
            get_int_size_bytes_n(),
            get_flt_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_flt_offset_bytes_n() const {
        return 
            get_int_offset_bytes_n() +
            get_int_size_bytes_n() +
            get_flt_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_rand_state_pad_bytes_n() +
            get_rand_state_size_bytes_n() +
            get_err_pad_bytes_n() +
            get_err_size_bytes_n() +
            get_int_pad_bytes_n() +
            get_int_size_bytes_n() +
            get_flt_pad_bytes_n() +
            get_flt_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_rand_state_align_bytes_n(),
            get_err_align_bytes_n(),
            get_int_align_bytes_n(),
            get_flt_align_bytes_n());
    }
};

struct WorkPtr : WorkArgs {
    char *ptr;
    
    __device__ __host__
    curandState *get_rand_state_ptr() const {
        const size_t offset = get_rand_state_offset_bytes_n();
        return reinterpret_cast<curandState *>(ptr + offset);
    }
    
    __device__ __host__
    Int *get_err_ptr() const {
        const size_t offset = get_err_offset_bytes_n();
        return reinterpret_cast<Int *>(ptr + offset);
    }
    
    __device__ __host__
    Int *get_int_ptr() const {
        const size_t offset = get_int_offset_bytes_n();
        return reinterpret_cast<Int *>(ptr + offset);
    }
    
    __device__ __host__
    Flt *get_flt_ptr() const {
        const size_t offset = get_flt_offset_bytes_n();
        return reinterpret_cast<Flt *>(ptr + offset);
    }
};

struct MemoryArgs {
    Mid mem_ints_m;
    Mid mem_flts_m;
    
    __device__ __host__
    size_t get_int_size_bytes_n() const {
        return sizeof(Int);
    }
    
    __device__ __host__
    size_t get_int_align_bytes_n() const {
        return alignof(Int);
    }
    
    __device__ __host__
    size_t get_int_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_int_size_bytes_n(),
            get_int_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_ints_size_bytes_n() const {
        return 
            mem_ints_m * get_int_size_bytes_n() +
            mem_ints_m * get_int_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_ints_align_bytes_n() const {
        return alignof(Int);
    }
    
    __device__ __host__
    size_t get_ints_pad_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_ints_offset_bytes_n() const {
        return 0;
    }
    
    __device__ __host__
    size_t get_int_offset_bytes_n(Mid int_i) const {
        return get_ints_offset_bytes_n() +
            int_i * get_int_size_bytes_n() +
            int_i * get_int_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_flt_size_bytes_n() const {
        return sizeof(Flt);
    }
    
    __device__ __host__
    size_t get_flt_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_flt_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_flt_size_bytes_n(),
            get_flt_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_flts_size_bytes_n() const {
        return 
            mem_flts_m * get_flt_size_bytes_n() +
            mem_flts_m * get_flt_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_flts_align_bytes_n() const {
        return alignof(Flt);
    }
    
    __device__ __host__
    size_t get_flts_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_ints_offset_bytes_n() +
            get_ints_size_bytes_n(),
            get_flts_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_flts_offset_bytes_n() const {
        return 
            get_ints_offset_bytes_n() +
            get_ints_size_bytes_n() +
            get_flts_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_flt_offset_bytes_n(Mid flt_i) const {
        return get_flts_offset_bytes_n() +
            flt_i * get_flt_size_bytes_n() +
            flt_i * get_flt_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_ints_pad_bytes_n() +
            get_ints_size_bytes_n() +
            get_flts_pad_bytes_n() +
            get_flts_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_ints_align_bytes_n(),
            get_flts_align_bytes_n());
    }
};

struct MemoryPtr : MemoryArgs {
    char *ptr;
    
    __device__ __host__
    Int *get_ints_ptr() const {
        const size_t offset = get_ints_offset_bytes_n();
        return reinterpret_cast<Int *>(ptr + offset);
    }
    
    __device__ __host__
    Int *get_int_ptr(const Mid int_i) const {
        return get_ints_ptr() + int_i;
    }
    
    __device__ __host__
    Flt *get_flts_ptr() const {
        const size_t offset = get_flts_offset_bytes_n();
        return reinterpret_cast<Flt *>(ptr + offset);
    }
    
    __device__ __host__
    Flt *get_flt_ptr(const Mid flt_i) const {
        return get_flts_ptr() + flt_i;
    }
};

struct ShotStateArgs {
    Qid qubits_n;
    Eid entries_m;
    Mid mem_ints_m;
    Mid mem_flts_m;
    
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
    size_t get_entries_size_bytes_n() const {
        return EntriesArgs{qubits_n, entries_m}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_entries_align_bytes_n() const {
        return EntriesArgs{qubits_n, entries_m}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_entries_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_decomp_offset_bytes_n() +
            get_decomp_size_bytes_n(),
            get_entries_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_entries_offset_bytes_n() const {
        return 
            get_decomp_offset_bytes_n() +
            get_decomp_size_bytes_n() +
            get_entries_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_work_size_bytes_n() const {
        return WorkArgs{}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_work_align_bytes_n() const {
        return WorkArgs{}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_work_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_entries_offset_bytes_n() +
            get_entries_size_bytes_n(),
            get_work_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_work_offset_bytes_n() const {
        return 
            get_entries_offset_bytes_n() +
            get_entries_size_bytes_n() +
            get_work_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_memory_size_bytes_n() const {
        return MemoryArgs{mem_ints_m, mem_flts_m}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_memory_align_bytes_n() const {
        return MemoryArgs{mem_ints_m, mem_flts_m}.get_align_bytes_n();
    }
    
    __device__ __host__
    size_t get_memory_pad_bytes_n() const {
        return compute_pad_bytes_n(
            get_work_offset_bytes_n() +
            get_work_size_bytes_n(),
            get_memory_align_bytes_n());
    }
    
    __device__ __host__
    size_t get_memory_offset_bytes_n() const {
        return 
            get_work_offset_bytes_n() +
            get_work_size_bytes_n() +
            get_memory_pad_bytes_n();
    }
    
    __device__ __host__
    size_t get_size_bytes_n() const {
        return 
            get_table_pad_bytes_n() +
            get_table_size_bytes_n() +
            get_decomp_pad_bytes_n() +
            get_decomp_size_bytes_n() +
            get_entries_pad_bytes_n() +
            get_entries_size_bytes_n() +
            get_work_pad_bytes_n() +
            get_work_size_bytes_n() +
            get_memory_pad_bytes_n() +
            get_memory_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_align_bytes_n() const {
        return max(
            get_table_align_bytes_n(),
            get_decomp_align_bytes_n(),
            get_entries_align_bytes_n(),
            get_work_align_bytes_n(),
            get_memory_align_bytes_n());
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
    EntriesPtr get_entries_ptr() const {
        const size_t offset = get_entries_offset_bytes_n();
        return {qubits_n, entries_m, ptr + offset};
    }
    
    __device__ __host__
    WorkPtr get_work_ptr() const {
        const size_t offset = get_work_offset_bytes_n();
        return {.ptr=ptr + offset};
    }
    
    __device__ __host__
    MemoryPtr get_memory_ptr() const {
        const size_t offset = get_memory_offset_bytes_n();
        return {mem_ints_m, mem_flts_m, ptr + offset};
    }
};

struct ShotsStateArgs {
    Sid shots_n;
    Qid qubits_n;
    Eid entries_m;
    Mid mem_ints_m;
    Mid mem_flts_m;
    
    __device__ __host__
    size_t get_shot_size_bytes_n() const {
        return ShotStateArgs{qubits_n, entries_m, mem_ints_m, mem_flts_m}.get_size_bytes_n();
    }
    
    __device__ __host__
    size_t get_shot_align_bytes_n() const {
        return ShotStateArgs{qubits_n, entries_m, mem_ints_m, mem_flts_m}.get_align_bytes_n();
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
        return ShotStateArgs{qubits_n, entries_m, mem_ints_m, mem_flts_m}.get_align_bytes_n();
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
        return {qubits_n, entries_m, mem_ints_m, mem_flts_m, ptr + offset};
    }
};

}

#endif
