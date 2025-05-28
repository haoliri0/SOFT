#include <cuda_runtime.h>
#include "./simulator.hpp"
#include "./utils_stripes.cuh"
#include "./utils_exception.hpp"

using namespace StnCuda;


static
cudaStream_t create_stream() {
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));
    return stream;
}

static
CudaSti *create_table(Sid const shots_n, Qid const qubits_n) {
    const Qid table_bytes_n = 2 * qubits_n * (2 * qubits_n + 1) * sizeof(CudaSti);

    CudaSti *table;
    cudaCheck(cudaMalloc(&table, table_bytes_n));
    cudaCheck(cudaMemset(table, 0, table_bytes_n));

    constexpr CudaSti one = true;
    const CudaQid rows_n = 2 * qubits_n;
    const CudaQid cols_n = 2 * qubits_n + 1;
    cuda_stripes_set(one, table, Stripe{shots_n}, Stripe{rows_n, cols_n + 1});

    return table;
}

static
CudaAid *create_map_n(Sid const shots_n) {
    const Aid map_n_bytes_n = shots_n * sizeof(Aid);

    CudaAid *map_n;
    cudaCheck(cudaMalloc(&map_n, map_n_bytes_n));

    constexpr CudaAid one = 1;
    cuda_stripes_set(one, map_n, Stripe{shots_n});

    return map_n;
}

static
CudaAid *create_map_keys(Sid const shots_n, Aid const map_limit) {
    const Aid map_keys_bytes_n = shots_n * map_limit * sizeof(Aid);

    CudaAid *map_keys;
    cudaCheck(cudaMalloc(&map_keys, map_keys_bytes_n));

    constexpr CudaAid zero = 0;
    cuda_stripes_set(zero, map_keys, Stripe{shots_n, map_limit});

    return map_keys;
}

static
CudaAmp *create_map_values(Sid const shots_n, Aid const map_limit) {
    const Aid map_values_bytes_n = shots_n * map_limit * sizeof(Amp);

    CudaAmp *map_values;
    cudaCheck(cudaMalloc(&map_values, map_values_bytes_n));

    constexpr CudaAmp one = 1.0;
    cuda_stripes_set(one, map_values, Stripe{shots_n, map_limit});

    return map_values;
}


Simulator::Simulator(Sid const shots_n, Qid const qubits_n, Aid const map_limit) :
    shots_n(shots_n),
    qubits_n(qubits_n),
    map_limit(map_limit),
    stream(create_stream()),
    table(create_table(shots_n, qubits_n)),
    map_n(create_map_n(shots_n)),
    map_keys(create_map_keys(shots_n, map_limit)),
    map_values(create_map_values(shots_n, map_limit)) {
}

Simulator::~Simulator() {
    cudaFree(this->map_n);
    cudaFree(this->map_keys);
    cudaFree(this->map_values);
    cudaFree(this->table);
    cudaStreamDestroy(this->stream);
}
