#include <cuda_runtime.h>
#include "simulator.hpp"
#include "utils/dimsop.cuh"

using namespace StnCuda;


cudaError_t Simulator::apply_x(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid src_row = this->qubits_n + qubit;
    const Qid dst_row = this->qubits_n * 2;

    cuda_dims_xor(this->stream, this->table, src_row, dst_row, Dim{this->shots_n}, Dim{rows_n}, Dim{cols_n, 0, 1});
    return cudaSuccess;
}

cudaError_t Simulator::apply_y(const int qubit) const noexcept {
    cudaError_t err = cudaSuccess;
    err = this->apply_x(qubit);
    if (err != cudaSuccess) return err;
    err = this->apply_z(qubit);
    if (err != cudaSuccess) return err;
    // phase is ignored
    return err;
}

cudaError_t Simulator::apply_z(const int qubit) const noexcept {
    const Qid qubits_n = this->qubits_n;
    const Qid rows_n = 2 * qubits_n;
    const Qid cols_n = 2 * qubits_n + 1;

    const Qid src_row = qubit;
    const Qid dst_row = this->qubits_n * 2;

    cuda_dims_xor(this->stream, this->table, src_row, dst_row, Dim{this->shots_n}, Dim{rows_n}, Dim{cols_n, 0, 1});
    return cudaSuccess;
}
