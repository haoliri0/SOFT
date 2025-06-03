#include <cuda_runtime.h>
#include "simulator.hpp"
#include "utils/dimsop.cuh"

template<typename Value>
static __device__ __host__
void op_for_xz(Value &xz, Value &p) {
    p ^= xz;
}

template<typename Value>
static __device__ __host__
void op_for_y(Value &x, Value &z, Value &p) {
    p ^= x ^ z;
}

template<typename Value>
static __device__ __host__
void op_for_h(Value &x, Value &z, Value &p) {
    p ^= x & z;
    cuda::std::swap(x, z);
}

template<typename Value>
static __device__ __host__
void op_for_s(Value &x, Value &z, Value &p) {
    p ^= x & z;
    z ^= x;
}

template<typename Value>
static __device__ __host__
void op_for_sdg(Value &x, Value &z, Value &p) {
    p ^= x & ~z;
    z ^= x;
}


using namespace StnCuda;

cudaError_t Simulator::apply_x(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid z_col = this->qubits_n + qubit;
    const Qid p_col = this->qubits_n * 2;

    cuda_dims_op2<CudaSti, op_for_xz<CudaSti>>(
        this->stream, this->table,
        z_col, p_col,
        Dim{this->shots_n},
        Dim{rows_n},
        Dim{cols_n, 0, 1});
    return cudaSuccess;
}

cudaError_t Simulator::apply_y(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid x_col = qubit;
    const Qid z_col = this->qubits_n + qubit;
    const Qid p_col = this->qubits_n * 2;

    cuda_dims_op3<CudaSti, op_for_y<CudaSti>>(
        this->stream, this->table,
        x_col, z_col, p_col,
        Dim{this->shots_n},
        Dim{rows_n},
        Dim{cols_n, 0, 1});
    return cudaSuccess;
}

cudaError_t Simulator::apply_z(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid x_col = qubit;
    const Qid p_col = this->qubits_n * 2;

    cuda_dims_op2<CudaSti, op_for_xz<CudaSti>>(
        this->stream, this->table,
        x_col, p_col,
        Dim{this->shots_n},
        Dim{rows_n},
        Dim{cols_n, 0, 1});
    return cudaSuccess;
}

cudaError_t Simulator::apply_h(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid x_col = qubit;
    const Qid z_col = this->qubits_n + qubit;
    const Qid p_col = this->qubits_n * 2;

    cuda_dims_op3<CudaSti, op_for_h<CudaSti>>(
        this->stream, this->table,
        x_col, z_col, p_col,
        Dim{this->shots_n},
        Dim{rows_n},
        Dim{cols_n, 0, 1});
    return cudaSuccess;
}

cudaError_t Simulator::apply_s(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid x_col = qubit;
    const Qid z_col = this->qubits_n + qubit;
    const Qid p_col = this->qubits_n * 2;

    cuda_dims_op3<CudaSti, op_for_s<CudaSti>>(
        this->stream, this->table,
        x_col, z_col, p_col,
        Dim{this->shots_n},
        Dim{rows_n},
        Dim{cols_n, 0, 1});
    return cudaSuccess;
}

cudaError_t Simulator::apply_sdg(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid x_col = qubit;
    const Qid z_col = this->qubits_n + qubit;
    const Qid p_col = this->qubits_n * 2;

    cuda_dims_op3<CudaSti, op_for_sdg<CudaSti>>(
        this->stream, this->table,
        x_col, z_col, p_col,
        Dim{this->shots_n},
        Dim{rows_n},
        Dim{cols_n, 0, 1});
    return cudaSuccess;
}
