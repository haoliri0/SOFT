#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./utils/dimsop.cuh"

using namespace StnCuda;

cudaError_t Simulator::create(Sid const shots_n, Qid const qubits_n, Aid const map_limit) noexcept {
    this->shots_n = shots_n;
    this->qubits_n = qubits_n;
    this->map_limit = map_limit;

    cudaError_t err = cudaSuccess;
    do {
        // create stream
        err = cudaStreamCreate(&this->stream);
        if (err != cudaSuccess) break;

        // allocate table
        const size_t table_bytes_n = 2 * qubits_n * (2 * qubits_n + 1) * sizeof(CudaSti);
        err = cudaMallocAsync(&this->table, table_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate mpa_n
        const size_t map_n_bytes_n = shots_n * sizeof(Kid);
        err = cudaMallocAsync(&this->map_n, map_n_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate map_keys
        const size_t map_keys_bytes_n = shots_n * map_limit * sizeof(Aid);
        err = cudaMallocAsync(&this->map_keys, map_keys_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // allocate map_values
        const size_t map_values_bytes_n = shots_n * map_limit * sizeof(Amp);
        err = cudaMallocAsync(&this->map_values, map_values_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        // initialize table
        err = cudaMemsetAsync(this->table, 0, table_bytes_n, this->stream);
        if (err != cudaSuccess) break;

        constexpr CudaSti sti_one = true;
        const CudaQid rows_n = 2 * qubits_n;
        const CudaQid cols_n = 2 * qubits_n + 1;
        cuda_dims_fill(this->stream, this->table, sti_one, Dim{shots_n}, Dim{rows_n}, Dim{cols_n + 1, 0, 1});

        // initialize map_n
        constexpr CudaKid kid_one = 1;
        cuda_dims_fill(this->stream, this->map_n, kid_one, Dim{shots_n});

        // initialize map_keys
        constexpr CudaAid aid_zero = 0;
        cuda_dims_fill(this->stream, this->map_keys, aid_zero, Dim{shots_n}, Dim{map_limit, 0, 1});

        // initialize map_values
        constexpr CudaAmp amp_one = 1;
        cuda_dims_fill(this->stream, this->map_values, amp_one, Dim{shots_n}, Dim{map_limit, 0, 1});

        // wait for async operations to complete
        err = cudaStreamSynchronize(this->stream);
        if (err != cudaSuccess) break;
    } while (false);

    if (err != cudaSuccess)
        this->destroy();

    return err;
}

cudaError_t Simulator::destroy() noexcept {
    if (this->table != nullptr) cudaFree(this->table);
    if (this->map_n != nullptr) cudaFree(this->map_n);
    if (this->map_keys != nullptr) cudaFree(this->map_keys);
    if (this->map_values != nullptr) cudaFree(this->map_values);
    if (this->stream != nullptr) cudaStreamDestroy(this->stream);
    this->table = nullptr;
    this->map_n = nullptr;
    this->map_keys = nullptr;
    this->map_values = nullptr;
    this->stream = nullptr;
    return cudaSuccess;
}
