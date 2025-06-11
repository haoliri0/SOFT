#ifndef STN_CUDA_DATAOPS_CUH
#define STN_CUDA_DATAOPS_CUH

#include "./utils/thread.cuh"
#include "./utils/dimsop.cuh"
#include "./datastruct.cuh"

using namespace StnCuda;


template<typename Args>
struct TableRowsOpArgs {
    ShotsStatePtr ptr;
    Args args;
};

template<typename Args, void (*op)(CudaQid row_i, TableRowPtr ptr, Args args)>
static __device__
void shots_table_rows_op(
    TableRowsOpArgs<Args> const args,
    DimsIdx<2> const dims_idx
) {
    const Sid shot_i = dims_idx.get<0>();
    const Qid row_i = dims_idx.get<1>();
    const TableRowPtr ptr = args.ptr
        .get_shot_state_ptr(shot_i)
        .get_table_ptr()
        .get_row_ptr(row_i);
    op(row_i, ptr, args.args);
}

template<typename Args, void (*op)(CudaQid row_i, TableRowPtr ptr, Args args)>
static __host__
void cuda_shots_table_rows_op(
    cudaStream_t stream,
    ShotsStatePtr shots_state_ptr,
    Args args
) {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid rows_n = TablePtr::get_rows_n(shots_state_ptr.qubits_n);
    cuda_dims_op<TableRowsOpArgs<Args>, 2, shots_table_rows_op<Args, op>>
        (stream, {shots_state_ptr, args}, dimsof(shots_n, rows_n));
}

#endif
