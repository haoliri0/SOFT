#ifndef STN_CUDA_DATASTRUCT_CUH
#define STN_CUDA_DATASTRUCT_CUH

#include "./datatype.cuh"

namespace StnCuda {

struct PauliRowPtr {
    Qid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n) {
        return 2 * qubits_n * sizeof(CudaBit);
    }

    __device__ __host__
    CudaBit *get_ptr(const Qid qubit_i) const {
        const size_t offset = qubit_i * sizeof(CudaBit);
        return reinterpret_cast<CudaBit*>(ptr + offset);
    }

    __device__ __host__
    CudaBit *get_x_ptr(const Qid qubit_i) const {
        const size_t offset = qubit_i * sizeof(CudaBit);
        return reinterpret_cast<CudaBit*>(ptr + offset);
    }

    __device__ __host__
    CudaBit *get_z_ptr(const Qid qubit_i) const {
        const size_t offset = (qubits_n + qubit_i) * sizeof(CudaBit);
        return reinterpret_cast<CudaBit*>(ptr + offset);
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
        return sizeof(CudaBit);
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
    CudaBit *get_sign_ptr() const {
        const size_t offset = _compute_pauli_bytes_n(qubits_n);
        return reinterpret_cast<CudaBit*>(ptr + offset);
    }
};

struct TablePtr {
    Qid qubits_n;
    char *ptr;

    static __device__ __host__
    Qid _rows_n(const Qid qubits_n) {
        return 2 * qubits_n;
    }

    static __device__ __host__
    size_t _compute_row_bytes_n(const Qid qubits_n) {
        return TableRowPtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t compute_bytes_n(const Qid qubits_n) {
        const CudaQid rows_n = _rows_n(qubits_n);
        return rows_n * _compute_row_bytes_n(qubits_n);
    }

    __device__ __host__
    Qid rows_n() const {
        return _rows_n(qubits_n);
    }

    __device__ __host__
    TableRowPtr get_row_ptr(const Qid row_i) const {
        const size_t offset = row_i * _compute_row_bytes_n(qubits_n);
        return TableRowPtr{qubits_n, ptr + offset};
    }
};

struct DecompPtr {
    CudaQid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t _compute_bits_bytes_n(const CudaQid qubits_n) {
        return 2 * qubits_n * sizeof(CudaBit);
    }

    static __device__ __host__
    size_t _compute_pauli_bytes_n(const CudaQid qubits_n) {
        return PauliRowPtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t _compute_phase_bytes_n() {
        return sizeof(CudaPhs);
    }

    static __device__ __host__
    size_t compute_bytes_n(const CudaQid qubits_n) {
        return
            _compute_bits_bytes_n(qubits_n) +
            _compute_pauli_bytes_n(qubits_n) +
            _compute_phase_bytes_n();
    }

    __device__ __host__
    CudaBit *get_bits_ptr() const {
        return reinterpret_cast<CudaBit*>(ptr);
    }

    __device__ __host__
    PauliRowPtr get_pauli_ptr() const {
        const size_t offset = _compute_bits_bytes_n(qubits_n);
        return PauliRowPtr{qubits_n, ptr + offset};
    }

    __device__ __host__
    CudaPhs *get_phase_ptr() const {
        const size_t offset =
            _compute_bits_bytes_n(qubits_n) +
            _compute_pauli_bytes_n(qubits_n);
        return reinterpret_cast<CudaPhs*>(ptr + offset);
    }
};

struct ShotStatePtr {
    CudaQid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t _compute_table_bytes_n(const CudaQid qubits_n) {
        return TablePtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t _compute_decomp_bytes_n(const CudaQid qubits_n) {
        return DecompPtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t compute_bytes_n(const CudaQid qubits_n) {
        return
            _compute_table_bytes_n(qubits_n) +
            _compute_decomp_bytes_n(qubits_n);
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
};

struct ShotsStatePtr {
    CudaSid shots_n;
    CudaQid qubits_n;
    char *ptr;

    static __device__ __host__
    size_t _compute_shot_bytes_n(const CudaQid qubits_n) {
        return ShotStatePtr::compute_bytes_n(qubits_n);
    }

    static __device__ __host__
    size_t compute_bytes_n(const CudaSid shots_n, const CudaQid qubits_n) {
        return shots_n * _compute_shot_bytes_n(qubits_n);
    }

    __device__ __host__
    ShotStatePtr get_shot_state_ptr(const Sid shots_i) const {
        const size_t offset = shots_i * _compute_shot_bytes_n(qubits_n);
        return ShotStatePtr{qubits_n, ptr + offset};
    }
};

}

#endif
