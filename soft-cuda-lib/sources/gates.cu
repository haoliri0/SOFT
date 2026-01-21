#include "./gates.cuh"

using namespace SoftCuda;

void Simulator::apply_x(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_x>(stream, shots_state_ptr, target);
}

void Simulator::apply_y(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_y>(stream, shots_state_ptr, target);
}

void Simulator::apply_z(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_z>(stream, shots_state_ptr, target);
}

void Simulator::apply_h(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_h>(stream, shots_state_ptr, target);
}

void Simulator::apply_s(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_s>(stream, shots_state_ptr, target);
}

void Simulator::apply_sdg(const Qid target) const noexcept {
    cuda_apply_gate1_op<op_apply_sdg>(stream, shots_state_ptr, target);
}

void Simulator::apply_cx(const Qid control, const Qid target) const noexcept {
    cuda_apply_gate2_op<op_apply_cx>(stream, shots_state_ptr, control, target);
}
