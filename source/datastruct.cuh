#ifndef STN_CUDA_DATASTRUCT_CUH
#define STN_CUDA_DATASTRUCT_CUH

#include "./datatype.cuh"

namespace StnCuda {

template<typename Item>
static __device__ __host__
Item max() {
    return 0;
}

template<typename Item, typename... Args>
static __device__ __host__
Item max(Item item, Args... args) {
    return item + max<Item>(args...);
}

static __device__ __host__
size_t compute_pad_bytes_n(const size_t offset_bytes_n, const size_t align_bytes_n) {
    return align_bytes_n - offset_bytes_n % align_bytes_n;
}


struct PauliRowPtr {
    struct Args {
        Qid qubits_n;
    };
    
    static __device__ __host__
    size_t get_bit_size_bytes_n(const Args args) {
        return sizeof(Bit);
    }
    
    static __device__ __host__
    size_t get_bits_size_bytes_n(const Args args) {
        return 2 * args.qubits_n * get_bit_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_bits_align_bytes_n(const Args args) {
        return alignof(Bit);
    }
    
    static __device__ __host__
    size_t get_bits_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_bits_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_bits_pad_bytes_n(args) +
            get_bits_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_bits_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    Bit *get_bit_ptr(const Qid bit_i) const {
        const size_t bit_size = get_bit_size_bytes_n(args);
        const size_t offset0 = get_bits_offset_bytes_n(args);
        const size_t offset = offset0 + bit_i * bit_size;
        return reinterpret_cast<Bit *>(ptr + offset);
    }
    
    __device__ __host__
    Bit *get_x_ptr(const Qid qubit_i) const {
        return get_bit_ptr(qubit_i);
    }
    
    __device__ __host__
    Bit *get_z_ptr(const Qid qubit_i) const {
        return get_bit_ptr(args.qubits_n + qubit_i);
    }
};

struct TableRowPtr {
    struct Args {
        Qid qubits_n;
    };
    
    static __device__ __host__
    size_t get_pauli_size_bytes_n(const Args args) {
        return PauliRowPtr::get_size_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_pauli_align_bytes_n(const Args args) {
        return PauliRowPtr::get_align_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_pauli_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_pauli_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_sign_size_bytes_n(const Args args) {
        return sizeof(Bit);
    }
    
    static __device__ __host__
    size_t get_sign_align_bytes_n(const Args args) {
        return alignof(Bit);
    }
    
    static __device__ __host__
    size_t get_sign_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_pauli_offset_bytes_n(args);
        const size_t align_bytes_n = get_sign_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_sign_offset_bytes_n(const Args args) {
        return 
            get_pauli_offset_bytes_n(args) +
            get_pauli_size_bytes_n(args) +
            get_sign_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_pauli_pad_bytes_n(args) +
            get_pauli_size_bytes_n(args) +
            get_sign_pad_bytes_n(args) +
            get_sign_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_pauli_align_bytes_n(args),
            get_sign_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        const size_t offset = get_pauli_offset_bytes_n(args);
        return {{args.qubits_n}, ptr + offset};
    }
    
    __device__ __host__
    Bit *get_sign_ptr() const {
        const size_t offset = get_sign_offset_bytes_n(args);
        return reinterpret_cast<Bit *>(ptr + offset);
    }
};

struct TablePtr {
    struct Args {
        Qid qubits_n;
    };
    
    static __device__ __host__
    size_t get_row_size_bytes_n(const Args args) {
        return TableRowPtr::get_size_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_rows_size_bytes_n(const Args args) {
        return 2 * args.qubits_n * get_row_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_rows_align_bytes_n(const Args args) {
        return TableRowPtr::get_align_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_rows_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_rows_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_rows_pad_bytes_n(args) +
            get_rows_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_rows_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    TableRowPtr get_row_ptr(const Qid row_i) const {
        const size_t row_size = get_row_size_bytes_n(args);
        const size_t offset0 = get_rows_offset_bytes_n(args);
        const size_t offset = offset0 + row_i * row_size;
        return {{args.qubits_n}, ptr + offset};
    }
};

struct DecompPtr {
    struct Args {
        Qid qubits_n;
    };
    
    static __device__ __host__
    size_t get_bit_size_bytes_n(const Args args) {
        return sizeof(Bit);
    }
    
    static __device__ __host__
    size_t get_bits_size_bytes_n(const Args args) {
        return 2 * args.qubits_n * get_bit_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_bits_align_bytes_n(const Args args) {
        return alignof(Bit);
    }
    
    static __device__ __host__
    size_t get_bits_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_bits_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_pauli_size_bytes_n(const Args args) {
        return PauliRowPtr::get_size_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_pauli_align_bytes_n(const Args args) {
        return PauliRowPtr::get_align_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_pauli_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_bits_offset_bytes_n(args);
        const size_t align_bytes_n = get_pauli_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_pauli_offset_bytes_n(const Args args) {
        return 
            get_bits_offset_bytes_n(args) +
            get_bits_size_bytes_n(args) +
            get_pauli_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_phase_size_bytes_n(const Args args) {
        return sizeof(Phs);
    }
    
    static __device__ __host__
    size_t get_phase_align_bytes_n(const Args args) {
        return alignof(Phs);
    }
    
    static __device__ __host__
    size_t get_phase_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_pauli_offset_bytes_n(args);
        const size_t align_bytes_n = get_phase_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_phase_offset_bytes_n(const Args args) {
        return 
            get_pauli_offset_bytes_n(args) +
            get_pauli_size_bytes_n(args) +
            get_phase_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_pivot_size_bytes_n(const Args args) {
        return sizeof(Qid);
    }
    
    static __device__ __host__
    size_t get_pivot_align_bytes_n(const Args args) {
        return alignof(Qid);
    }
    
    static __device__ __host__
    size_t get_pivot_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_phase_offset_bytes_n(args);
        const size_t align_bytes_n = get_pivot_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_pivot_offset_bytes_n(const Args args) {
        return 
            get_phase_offset_bytes_n(args) +
            get_phase_size_bytes_n(args) +
            get_pivot_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_bits_pad_bytes_n(args) +
            get_bits_size_bytes_n(args) +
            get_pauli_pad_bytes_n(args) +
            get_pauli_size_bytes_n(args) +
            get_phase_pad_bytes_n(args) +
            get_phase_size_bytes_n(args) +
            get_pivot_pad_bytes_n(args) +
            get_pivot_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_bits_align_bytes_n(args),
            get_pauli_align_bytes_n(args),
            get_phase_align_bytes_n(args),
            get_pivot_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    Bit *get_bit_ptr(const Qid bit_i) const {
        const size_t bit_size = get_bit_size_bytes_n(args);
        const size_t offset0 = get_bits_offset_bytes_n(args);
        const size_t offset = offset0 + bit_i * bit_size;
        return reinterpret_cast<Bit *>(ptr + offset);
    }
    
    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        const size_t offset = get_pauli_offset_bytes_n(args);
        return {{args.qubits_n}, ptr + offset};
    }
    
    __device__ __host__
    Phs *get_phase_ptr() const {
        const size_t offset = get_phase_offset_bytes_n(args);
        return reinterpret_cast<Phs *>(ptr + offset);
    }
    
    __device__ __host__
    Qid *get_pivot_ptr() const {
        const size_t offset = get_pivot_offset_bytes_n(args);
        return reinterpret_cast<Qid *>(ptr + offset);
    }
};

struct AmpsMapPtr {
    struct Args {
        Qid qubits_n;
        Kid amps_m;
    };
    
    static __device__ __host__
    size_t get_amps_n_size_bytes_n(const Args args) {
        return sizeof(Kid);
    }
    
    static __device__ __host__
    size_t get_amps_n_align_bytes_n(const Args args) {
        return alignof(Kid);
    }
    
    static __device__ __host__
    size_t get_amps_n_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_amps_n_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_amp_size_bytes_n(const Args args) {
        return sizeof(Amp);
    }
    
    static __device__ __host__
    size_t get_amps_size_bytes_n(const Args args) {
        return args.amps_m * get_amp_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_amps_align_bytes_n(const Args args) {
        return alignof(Amp);
    }
    
    static __device__ __host__
    size_t get_amps_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_amps_n_offset_bytes_n(args);
        const size_t align_bytes_n = get_amps_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_amps_offset_bytes_n(const Args args) {
        return 
            get_amps_n_offset_bytes_n(args) +
            get_amps_n_size_bytes_n(args) +
            get_amps_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_aid_size_bytes_n(const Args args) {
        return sizeof(Aid);
    }
    
    static __device__ __host__
    size_t get_aids_size_bytes_n(const Args args) {
        return args.amps_m * get_aid_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_aids_align_bytes_n(const Args args) {
        return alignof(Aid);
    }
    
    static __device__ __host__
    size_t get_aids_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_amps_offset_bytes_n(args);
        const size_t align_bytes_n = get_aids_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_aids_offset_bytes_n(const Args args) {
        return 
            get_amps_offset_bytes_n(args) +
            get_amps_size_bytes_n(args) +
            get_aids_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_amps_n_pad_bytes_n(args) +
            get_amps_n_size_bytes_n(args) +
            get_amps_pad_bytes_n(args) +
            get_amps_size_bytes_n(args) +
            get_aids_pad_bytes_n(args) +
            get_aids_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_amps_n_align_bytes_n(args),
            get_amps_align_bytes_n(args),
            get_aids_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    Kid *get_amps_n_ptr() const {
        const size_t offset = get_amps_n_offset_bytes_n(args);
        return reinterpret_cast<Kid *>(ptr + offset);
    }
    
    __device__ __host__
    Amp *get_amp_ptr(const Kid amp_i) const {
        const size_t amp_size = get_amp_size_bytes_n(args);
        const size_t offset0 = get_amps_offset_bytes_n(args);
        const size_t offset = offset0 + amp_i * amp_size;
        return reinterpret_cast<Amp *>(ptr + offset);
    }
    
    __device__ __host__
    Aid *get_aid_ptr(const Kid amp_i) const {
        const size_t aid_size = get_aid_size_bytes_n(args);
        const size_t offset0 = get_aids_offset_bytes_n(args);
        const size_t offset = offset0 + amp_i * aid_size;
        return reinterpret_cast<Aid *>(ptr + offset);
    }
};

struct ShotStatePtr {
    struct Args {
        Qid qubits_n;
        Kid amps_m;
    };
    
    static __device__ __host__
    size_t get_table_size_bytes_n(const Args args) {
        return TablePtr::get_size_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_table_align_bytes_n(const Args args) {
        return TablePtr::get_align_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_table_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_table_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_decomp_size_bytes_n(const Args args) {
        return DecompPtr::get_size_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_decomp_align_bytes_n(const Args args) {
        return DecompPtr::get_align_bytes_n({args.qubits_n});
    }
    
    static __device__ __host__
    size_t get_decomp_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_table_offset_bytes_n(args);
        const size_t align_bytes_n = get_decomp_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_decomp_offset_bytes_n(const Args args) {
        return 
            get_table_offset_bytes_n(args) +
            get_table_size_bytes_n(args) +
            get_decomp_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_amps_size_bytes_n(const Args args) {
        return AmpsMapPtr::get_size_bytes_n({args.qubits_n, args.amps_m});
    }
    
    static __device__ __host__
    size_t get_amps_align_bytes_n(const Args args) {
        return AmpsMapPtr::get_align_bytes_n({args.qubits_n, args.amps_m});
    }
    
    static __device__ __host__
    size_t get_amps_pad_bytes_n(const Args args) {
        const size_t offset_bytes_n = get_decomp_offset_bytes_n(args);
        const size_t align_bytes_n = get_amps_align_bytes_n(args);
        return compute_pad_bytes_n(offset_bytes_n, align_bytes_n);
    }
    
    static __device__ __host__
    size_t get_amps_offset_bytes_n(const Args args) {
        return 
            get_decomp_offset_bytes_n(args) +
            get_decomp_size_bytes_n(args) +
            get_amps_pad_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_table_pad_bytes_n(args) +
            get_table_size_bytes_n(args) +
            get_decomp_pad_bytes_n(args) +
            get_decomp_size_bytes_n(args) +
            get_amps_pad_bytes_n(args) +
            get_amps_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_table_align_bytes_n(args),
            get_decomp_align_bytes_n(args),
            get_amps_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    TablePtr get_table_ptr() const {
        const size_t offset = get_table_offset_bytes_n(args);
        return {{args.qubits_n}, ptr + offset};
    }
    
    __device__ __host__
    DecompPtr get_decomp_ptr() const {
        const size_t offset = get_decomp_offset_bytes_n(args);
        return {{args.qubits_n}, ptr + offset};
    }
    
    __device__ __host__
    AmpsMapPtr get_amps_ptr() const {
        const size_t offset = get_amps_offset_bytes_n(args);
        return {{args.qubits_n, args.amps_m}, ptr + offset};
    }
};

struct ShotsStatePtr {
    struct Args {
        Sid shots_n;
        Qid qubits_n;
        Kid amps_m;
    };
    
    static __device__ __host__
    size_t get_shot_size_bytes_n(const Args args) {
        return ShotStatePtr::get_size_bytes_n({args.qubits_n, args.amps_m});
    }
    
    static __device__ __host__
    size_t get_shots_size_bytes_n(const Args args) {
        return args.shots_n * get_shot_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_shots_align_bytes_n(const Args args) {
        return ShotStatePtr::get_align_bytes_n({args.qubits_n, args.amps_m});
    }
    
    static __device__ __host__
    size_t get_shots_pad_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_shots_offset_bytes_n(const Args args) {
        return 0;
    }
    
    static __device__ __host__
    size_t get_size_bytes_n(const Args args) {
        return 
            get_shots_pad_bytes_n(args) +
            get_shots_size_bytes_n(args);
    }
    
    static __device__ __host__
    size_t get_align_bytes_n(const Args args) {
        return max(
            get_shots_align_bytes_n(args));
    }
    
    Args args;
    char *ptr;
    
    __device__ __host__
    ShotStatePtr get_shot_ptr(const Sid shot_i) const {
        const size_t shot_size = get_shot_size_bytes_n(args);
        const size_t offset0 = get_shots_offset_bytes_n(args);
        const size_t offset = offset0 + shot_i * shot_size;
        return {{args.qubits_n, args.amps_m}, ptr + offset};
    }
};

}

#endif
