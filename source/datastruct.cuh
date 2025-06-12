#ifndef STN_CUDA_DATASTRUCT_CUH
#define STN_CUDA_DATASTRUCT_CUH

#include "./datatype.cuh"

namespace StnCuda {

struct PauliRowPtr {
    Qid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n) {
        return 2 * qubits_n * sizeof(Bit);
    }

    __device__ __host__
    Bit *get_ptr(const Qid col_i) const {
        const size_t offset = col_i * sizeof(Bit);
        return reinterpret_cast<Bit *>(ptr + offset);
    }

    __device__ __host__
    Bit *get_x_ptr(const Qid qubit_i) const {
        const size_t offset = qubit_i * sizeof(Bit);
        return reinterpret_cast<Bit *>(ptr + offset);
    }

    __device__ __host__
    Bit *get_z_ptr(const Qid qubit_i) const {
        const size_t offset = (qubits_n + qubit_i) * sizeof(Bit);
        return reinterpret_cast<Bit *>(ptr + offset);
    }
};

struct TableRowPtr {
    Qid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t _compute_pauli_bytes_n(const Qid qubits_n) {
        return PauliRowPtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t _compute_sign_bytes_n() {
        return sizeof(Bit);
    }

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n) {
        return
            _compute_pauli_bytes_n(qubits_n) +
            _compute_sign_bytes_n();
    }

    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        return PauliRowPtr{qubits_n, ptr};
    }

    __device__ __host__
    Bit *get_sign_ptr() const {
        const size_t offset = _compute_pauli_bytes_n(qubits_n);
        return reinterpret_cast<Bit *>(ptr + offset);
    }
};

struct TablePtr {
    Qid qubits_n;
    char *ptr;

    static __device__ __host__
    Qid get_rows_n(const Qid qubits_n) {
        return 2 * qubits_n;
    }

    static __device__ __host__
    size_t _compute_row_bytes_n(const Qid qubits_n) {
        return TableRowPtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n) {
        const Qid rows_n = get_rows_n(qubits_n);
        return rows_n * _compute_row_bytes_n(qubits_n);
    }

    __device__ __host__
    Qid get_rows_n() const {
        return get_rows_n(qubits_n);
    }

    __device__ __host__
    TableRowPtr get_row_ptr(const Qid row_i) const {
        const size_t offset = row_i * _compute_row_bytes_n(qubits_n);
        return TableRowPtr{qubits_n, ptr + offset};
    }
};

struct DecompPtr {
    Qid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t _compute_bits_bytes_n(const Qid qubits_n) {
        return 2 * qubits_n * sizeof(Bit);
    }

    static __device__ __host__
    size_t _compute_pauli_bytes_n(const Qid qubits_n) {
        return PauliRowPtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t _compute_phase_bytes_n() {
        return sizeof(Phs);
    }

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n) {
        return
            _compute_bits_bytes_n(qubits_n) +
            _compute_pauli_bytes_n(qubits_n) +
            _compute_phase_bytes_n();
    }

    __device__ __host__
    Bit *get_bits_ptr() const {
        return reinterpret_cast<Bit *>(ptr);
    }

    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        const size_t offset = _compute_bits_bytes_n(qubits_n);
        return PauliRowPtr{qubits_n, ptr + offset};
    }

    __device__ __host__
    Phs *get_phase_ptr() const {
        const size_t offset =
            _compute_bits_bytes_n(qubits_n) +
            _compute_pauli_bytes_n(qubits_n);
        return reinterpret_cast<Phs *>(ptr + offset);
    }
};

struct AmpEntry {
    Aid key;
    Amp value;
};

struct AmpsMapPtr {
    Kid amps_m;
    char *ptr;

    static __device__ __host__
    size_t get_entries_n_bytes_n() {
        return sizeof(Kid);
    }

    static __device__ __host__
    size_t get_entries_bytes_n(const Kid amps_m) {
        return amps_m * sizeof(AmpEntry);
    }

    static __device__ __host__
    size_t get_bytes_n(const Kid amps_m) {
        return
            get_entries_n_bytes_n() +
            get_entries_bytes_n(amps_m);
    }

    __device__ __host__
    Kid *get_entries_n_ptr() {
        return reinterpret_cast<Kid *>(ptr);
    }

    __device__ __host__
    AmpEntry *get_entries_ptr() {
        const size_t offset = get_entries_n_bytes_n();
        return reinterpret_cast<AmpEntry *>(ptr + offset);
    }

    __device__ __host__
    AmpEntry *get_entry_ptr(const Kid amp_i) {
        return get_entries_ptr() + amp_i;
    }

};

struct ShotStatePtr {
    Qid qubits_n;
    Kid amps_m;
    char *ptr;

    static __device__ __host__
    size_t _compute_table_bytes_n(const Qid qubits_n) {
        return TablePtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t _compute_decomp_bytes_n(const Qid qubits_n) {
        return DecompPtr::compute_bytes_n(qubits_n);
    }
    static __device__ __host__
    size_t _compute_amps_bytes_n(const Qid qubits_n) {
        return AmpsMapPtr::get_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n, const Kid amps_m) {
        return
            _compute_table_bytes_n(qubits_n) +
            _compute_decomp_bytes_n(qubits_n) +
            _compute_amps_bytes_n(amps_m);
    }

    __device__ __host__
    TablePtr get_table_ptr() const {
        return TablePtr{qubits_n, ptr};
    }

    __device__ __host__
    DecompPtr get_decomp_ptr() const {
        const size_t offset = _compute_table_bytes_n(qubits_n);
        return DecompPtr{qubits_n, ptr + offset};
    }

    __device__ __host__
    AmpsMapPtr get_amps_map_ptr() const {
        const size_t offset =
            _compute_table_bytes_n(qubits_n) +
            _compute_decomp_bytes_n(qubits_n);
        return AmpsMapPtr{amps_m, ptr + offset};
    }
};

struct ShotsStatePtr {
    Sid shots_n;
    Qid qubits_n;
    Kid amps_m;
    char *ptr;

    static __device__ __host__
    size_t _compute_shot_bytes_n(const Qid qubits_n, const Kid amps_m) {
        return ShotStatePtr::compute_bytes_n(qubits_n, amps_m);
    }

    static __device__ __host__
    size_t compute_bytes_n(const Sid shots_n, const Qid qubits_n, const Kid amps_m) {
        return shots_n * _compute_shot_bytes_n(qubits_n, amps_m);
    }

    __device__ __host__
    ShotStatePtr get_shot_state_ptr(const Sid shots_i) const {
        const size_t offset = shots_i * _compute_shot_bytes_n(qubits_n, amps_m);
        return ShotStatePtr{qubits_n, amps_m, ptr + offset};
    }
};

}

#endif
