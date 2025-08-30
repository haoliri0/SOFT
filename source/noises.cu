#include "dimsop.cuh"
#include "gates.cuh"
#include "simulator.hpp"

using namespace StnCuda;

struct RandomChooseResult {
    const Int value;
    const Flt prob;
};

template<unsigned int probs_n>
static __device__ __host__
RandomChooseResult compute_random_choose_result(
    const Array<Flt, probs_n> probs,
    const Flt sample,
    const Flt head_prob,
    const Int head_value
) noexcept {
    if constexpr (probs_n == 0) {
        return {0, 1 - head_prob};
    } else {
        const Flt prob = probs.item;
        const Flt tail_prob = head_prob + prob;
        const Int tail_value = head_value + 1;
        if (sample <= tail_prob) return {tail_value, prob};
        return compute_random_choose_result<probs_n - 1>(probs.tail, sample, tail_prob, tail_value);
    }
}

template<unsigned int probs_n>
static __device__ __host__
RandomChooseResult compute_random_choose_result(
    const Array<Flt, probs_n> probs,
    const Flt sample
) noexcept {
    return compute_random_choose_result(probs, sample, 0., 0);
}

template<unsigned int probs_n>
struct ArgsRandomSample {
    const ShotsStatePtr shots_state_ptr;
    const Array<Flt, probs_n> probs;
};

template<unsigned int probs_n>
static __device__
void op_random_choose(const ArgsRandomSample<probs_n> args, const DimsIdx<1> dims_idx) noexcept {
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const Array<Flt, probs_n> probs = args.probs;

    Sid const shot_i = dims_idx.get<0>();
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();

    curandState *rand_state_ptr = work_ptr.get_rand_state_ptr();
    const Flt sample = curand_uniform(rand_state_ptr);
    const auto result = compute_random_choose_result<probs_n>(probs, sample);

    Flt &result_prob = *work_ptr.get_flt_ptr();
    Int &result_value = *work_ptr.get_int_ptr();
    result_value = result.value;
    result_prob = result.prob;
}

template<unsigned int probs_n>
void cuda_random_choose(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Array<Flt, probs_n> const probs
) noexcept {
    const Sid shots_n = shots_state_ptr.shots_n;
    cuda_dims_op<ArgsRandomSample<probs_n>, 1, op_random_choose<probs_n>>
        (stream, {shots_state_ptr, probs}, dimsof(shots_n));
}


template<void (*op)(Bit &s, Bit &x, Bit &z)>
static __device__
void op_noise_gate(const ArgsApplyGate1 args, const DimsIdx<2> dims_idx) noexcept {
    const Sid shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Int result_value = *work_ptr.get_int_ptr();
    if (result_value) op_apply_gate1<op>(args, dims_idx);
}

template<void (*op)(Bit &s, Bit &x, Bit &z)>
void cuda_noise_gate(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Qid const target
) noexcept {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsApplyGate1, 2, op_noise_gate<op>>
        (stream, {shots_state_ptr, target}, dimsof(shots_n, rows_n));
}

void Simulator::apply_noise_x(const Flt prob, const Qid target) const noexcept {
    cuda_random_choose(stream, shots_state_ptr, arrayof<Flt>(prob));
    cuda_noise_gate<op_apply_x>(stream, shots_state_ptr, target);
}

void Simulator::apply_noise_z(const Flt prob, const Qid target) const noexcept {
    cuda_random_choose(stream, shots_state_ptr, arrayof<Flt>(prob));
    cuda_noise_gate<op_apply_z>(stream, shots_state_ptr, target);
}


static __device__
void op_noise_depo1(const ArgsApplyGate1 args, const DimsIdx<2> dims_idx) noexcept {
    const Sid shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Int result_value = *work_ptr.get_int_ptr();
    if (result_value == 1) op_apply_gate1<op_apply_x>(args, dims_idx);
    if (result_value == 2) op_apply_gate1<op_apply_y>(args, dims_idx);
    if (result_value == 3) op_apply_gate1<op_apply_z>(args, dims_idx);
}

void cuda_noise_depo1(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Qid const target
) noexcept {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsApplyGate1, 2, op_noise_depo1>
        (stream, {shots_state_ptr, target}, dimsof(shots_n, rows_n));
}

void Simulator::apply_noise_depo1(const Flt prob, const Qid target) const noexcept {
    cuda_random_choose(stream, shots_state_ptr, arrayof<Flt>(prob / 3, prob / 3, prob / 3));
    cuda_noise_depo1(stream, shots_state_ptr, target);
}


static __device__
void op_noise_depo2(const ArgsApplyGate2 args, const DimsIdx<2> dims_idx) noexcept {
    const Sid shot_i = dims_idx.get<0>();
    const ShotsStatePtr shots_state_ptr = args.shots_state_ptr;
    const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);
    const WorkPtr work_ptr = shot_state_ptr.get_work_ptr();
    const Int result_value = *work_ptr.get_int_ptr();

    const Int result_value0 = result_value % 4;
    const Int result_value1 = result_value / 4 % 4;
    const ArgsApplyGate1 args0 = {args.shots_state_ptr, args.target0};
    const ArgsApplyGate1 args1 = {args.shots_state_ptr, args.target1};

    if (result_value0 == 1) op_apply_gate1<op_apply_x>(args0, dims_idx);
    if (result_value0 == 2) op_apply_gate1<op_apply_y>(args0, dims_idx);
    if (result_value0 == 3) op_apply_gate1<op_apply_z>(args0, dims_idx);

    if (result_value1 == 1) op_apply_gate1<op_apply_x>(args1, dims_idx);
    if (result_value1 == 2) op_apply_gate1<op_apply_y>(args1, dims_idx);
    if (result_value1 == 3) op_apply_gate1<op_apply_z>(args1, dims_idx);
}

void cuda_noise_depo2(
    cudaStream_t const stream,
    ShotsStatePtr const shots_state_ptr,
    Qid const target0,
    Qid const target1
) noexcept {
    const Sid shots_n = shots_state_ptr.shots_n;
    const Qid qubits_n = shots_state_ptr.qubits_n;
    const Qid rows_n = 2 * qubits_n;
    cuda_dims_op<ArgsApplyGate2, 2, op_noise_depo2>
        (stream, {shots_state_ptr, target0, target1}, dimsof(shots_n, rows_n));
}

void Simulator::apply_noise_depo2(const Flt prob, const Qid target0, const Qid target1) const noexcept {
    cuda_random_choose(stream, shots_state_ptr, arrayof<Flt>(
        prob / 15, prob / 15, prob / 15, prob / 15, prob / 15,
        prob / 15, prob / 15, prob / 15, prob / 15, prob / 15,
        prob / 15, prob / 15, prob / 15, prob / 15, prob / 15));
    cuda_noise_depo2(stream, shots_state_ptr, target0, target1);
}
