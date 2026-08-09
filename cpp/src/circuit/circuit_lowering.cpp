#include "core/internal.hpp"

#include "circuit/circuit.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace symft {
using namespace detail;

namespace {

struct CircuitLoweringAccumulator {
    FrameFactoredState state;
    std::vector<SymbolicBool> records;
};

std::pair<int, int> reserve_measurement_record(std::vector<SymbolicBool>& records, SymbolicContext& context) {
    const int condition = context.fresh_condition();
    records.push_back(symbolic_bool(condition));
    return {static_cast<int>(records.size()), condition};
}

SymbolicBool measurement_sign_with_readout_error(SymbolicContext& context, bool inverted, double probability) {
    SymbolicBool sign(inverted);
    if (probability != 0.0) {
        sign = xor_bool(sign, context.fresh_bernoulli_bool(check_probability(probability)));
    }
    return sign;
}

PauliString pauli_on_axis(int n, char axis, int q) {
    if (axis == 'X') {
        return pauli_x(n, q);
    }
    if (axis == 'Y') {
        return pauli_y(n, q);
    }
    if (axis == 'Z') {
        return pauli_z(n, q);
    }
    fail("unsupported Pauli axis");
}

SymbolicBool record_symbol(const std::vector<SymbolicBool>& records, int record) {
    if (record <= 0 || record > static_cast<int>(records.size())) {
        fail("measurement record target out of range");
    }
    return records[static_cast<std::size_t>(record - 1)];
}

void require_even_targets(const CircuitInstruction& instruction, const char* message) {
    if ((instruction.qubits.size() & 1u) != 0u) {
        fail(message);
    }
}

void require_target_multiple(const CircuitInstruction& instruction, std::size_t multiple, const char* message) {
    if (multiple == 0 || instruction.qubits.size() % multiple != 0) {
        fail(message);
    }
}

void require_distinct_targets(const std::vector<int>& qubits) {
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        for (std::size_t j = i + 1; j < qubits.size(); ++j) {
            if (qubits[i] == qubits[j]) {
                fail("multi-qubit operation requires distinct qubits");
            }
        }
    }
}

std::pair<bool, bool> pauli_axis_bits(int axis) {
    if (axis == 0) {
        return {false, false};
    }
    if (axis == 1) {
        return {true, false};
    }
    if (axis == 2) {
        return {true, true};
    }
    if (axis == 3) {
        return {false, true};
    }
    fail("invalid Pauli channel axis");
}

std::pair<bool, bool> pauli_axis_bits(char axis) {
    if (axis == 'X') {
        return {true, false};
    }
    if (axis == 'Y') {
        return {true, true};
    }
    if (axis == 'Z') {
        return {false, true};
    }
    fail("unsupported Pauli axis");
}

void apply_pauli_on_axis(
    FrameFactoredState& state,
    char axis,
    int q,
    const SymbolicBool& condition) {
    const auto [x, z] = pauli_axis_bits(axis);
    apply_single_qubit_pauli(state, q, x, z, condition);
}

PauliString maybe_inverted_pauli_product(CircuitPauliProduct product);

void apply_single_qubit_clifford(FrameFactoredState& state, CircuitInstructionKind kind, int q) {
    switch (kind) {
    case CircuitInstructionKind::H:
        left_H(state, q);
        return;
    case CircuitInstructionKind::H_NXY:
        left_H_NXY(state, q);
        return;
    case CircuitInstructionKind::H_NXZ:
        left_H_NXZ(state, q);
        return;
    case CircuitInstructionKind::H_NYZ:
        left_H_NYZ(state, q);
        return;
    case CircuitInstructionKind::H_XY:
        left_H_XY(state, q);
        return;
    case CircuitInstructionKind::H_YZ:
        left_H_YZ(state, q);
        return;
    case CircuitInstructionKind::C_NXYZ:
        left_C_NXYZ(state, q);
        return;
    case CircuitInstructionKind::C_NZYX:
        left_C_NZYX(state, q);
        return;
    case CircuitInstructionKind::C_XNYZ:
        left_C_XNYZ(state, q);
        return;
    case CircuitInstructionKind::C_XYNZ:
        left_C_XYNZ(state, q);
        return;
    case CircuitInstructionKind::C_XYZ:
        left_C_XYZ(state, q);
        return;
    case CircuitInstructionKind::C_ZNYX:
        left_C_ZNYX(state, q);
        return;
    case CircuitInstructionKind::C_ZYNX:
        left_C_ZYNX(state, q);
        return;
    case CircuitInstructionKind::C_ZYX:
        left_C_ZYX(state, q);
        return;
    case CircuitInstructionKind::S:
        left_S(state, q);
        return;
    case CircuitInstructionKind::SDG:
        left_SDG(state, q);
        return;
    case CircuitInstructionKind::SqrtX:
        left_SQRT_X(state, q);
        return;
    case CircuitInstructionKind::SqrtXDag:
        left_SQRT_X_DAG(state, q);
        return;
    case CircuitInstructionKind::SqrtY:
        left_SQRT_Y(state, q);
        return;
    case CircuitInstructionKind::SqrtYDag:
        left_SQRT_Y_DAG(state, q);
        return;
    case CircuitInstructionKind::X:
        left_X(state, q);
        return;
    case CircuitInstructionKind::Y:
        left_Y(state, q);
        return;
    case CircuitInstructionKind::Z:
        left_Z(state, q);
        return;
    default:
        fail("unsupported single-qubit Clifford kind");
    }
}

void apply_two_qubit_clifford(FrameFactoredState& state, CircuitInstructionKind kind, int a, int b) {
    switch (kind) {
    case CircuitInstructionKind::CX:
        left_CX(state, a, b);
        return;
    case CircuitInstructionKind::CY:
        left_CY(state, a, b);
        return;
    case CircuitInstructionKind::CZ:
        left_CZ(state, a, b);
        return;
    case CircuitInstructionKind::SWAP:
        left_SWAP(state, a, b);
        return;
    case CircuitInstructionKind::CXSWAP:
        left_CXSWAP(state, a, b);
        return;
    case CircuitInstructionKind::CZSWAP:
        left_CZSWAP(state, a, b);
        return;
    case CircuitInstructionKind::ISWAP:
        left_ISWAP(state, a, b);
        return;
    case CircuitInstructionKind::ISWAP_DAG:
        left_ISWAP_DAG(state, a, b);
        return;
    case CircuitInstructionKind::SQRT_XX:
        left_SQRT_XX(state, a, b);
        return;
    case CircuitInstructionKind::SQRT_XX_DAG:
        left_SQRT_XX_DAG(state, a, b);
        return;
    case CircuitInstructionKind::SQRT_YY:
        left_SQRT_YY(state, a, b);
        return;
    case CircuitInstructionKind::SQRT_YY_DAG:
        left_SQRT_YY_DAG(state, a, b);
        return;
    case CircuitInstructionKind::SQRT_ZZ:
        left_SQRT_ZZ(state, a, b);
        return;
    case CircuitInstructionKind::SQRT_ZZ_DAG:
        left_SQRT_ZZ_DAG(state, a, b);
        return;
    case CircuitInstructionKind::SWAPCX:
        left_SWAPCX(state, a, b);
        return;
    case CircuitInstructionKind::XCX:
        left_XCX(state, a, b);
        return;
    case CircuitInstructionKind::XCY:
        left_XCY(state, a, b);
        return;
    case CircuitInstructionKind::XCZ:
        left_XCZ(state, a, b);
        return;
    case CircuitInstructionKind::YCX:
        left_YCX(state, a, b);
        return;
    case CircuitInstructionKind::YCY:
        left_YCY(state, a, b);
        return;
    case CircuitInstructionKind::YCZ:
        left_YCZ(state, a, b);
        return;
    default:
        fail("unsupported two-qubit Clifford kind");
    }
}

void apply_depolarize1(FrameFactoredState& state, int q, double probability) {
    probability = check_probability(probability);
    const std::vector<std::vector<std::uint64_t>> assignments{
        packed_bits({false, false}),
        packed_bits({false, true}),
        packed_bits({true, false}),
        packed_bits({true, true}),
    };
    const auto bits = state.context->fresh_categorical_bools(
        2,
        assignments,
        {1.0 - probability, probability / 3.0, probability / 3.0, probability / 3.0});
    apply_single_qubit_pauli(state, q, true, false, bits[0]);
    apply_single_qubit_pauli(state, q, false, true, bits[1]);
}

void apply_depolarize2(FrameFactoredState& state, int q1, int q2, double probability) {
    probability = check_probability(probability);
    std::vector<std::vector<std::uint64_t>> assignments;
    assignments.push_back(packed_bits({false, false, false, false}));
    for (bool x1 : {false, true}) {
        for (bool z1 : {false, true}) {
            for (bool x2 : {false, true}) {
                for (bool z2 : {false, true}) {
                    if (x1 || z1 || x2 || z2) {
                        assignments.push_back(packed_bits({x1, z1, x2, z2}));
                    }
                }
            }
        }
    }
    std::vector<double> probabilities(16, probability / 15.0);
    probabilities[0] = 1.0 - probability;
    const auto bits = state.context->fresh_categorical_bools(4, assignments, probabilities);
    apply_single_qubit_pauli(state, q1, true, false, bits[0]);
    apply_single_qubit_pauli(state, q1, false, true, bits[1]);
    apply_single_qubit_pauli(state, q2, true, false, bits[2]);
    apply_single_qubit_pauli(state, q2, false, true, bits[3]);
}

void apply_pauli_channel(FrameFactoredState& state, const std::vector<int>& qubits, const std::vector<double>& probabilities) {
    require_distinct_targets(qubits);
    const std::size_t n = qubits.size();
    std::size_t cases = 1;
    for (std::size_t i = 0; i < n; ++i) {
        cases *= 4;
    }
    if (probabilities.size() != cases - 1) {
        fail("Pauli channel probability count does not match target arity");
    }
    std::vector<std::vector<std::uint64_t>> assignments;
    assignments.push_back(packed_bits(std::vector<bool>(2 * n, false)));
    std::vector<double> distribution;
    distribution.reserve(cases);
    double total_error_probability = 0.0;
    for (double probability : probabilities) {
        total_error_probability += check_probability(probability);
    }
    if (total_error_probability > 1.0 + 1e-12) {
        fail("Pauli channel probabilities sum to more than 1");
    }
    distribution.push_back(std::max(0.0, 1.0 - total_error_probability));
    for (std::size_t code = 1; code < cases; ++code) {
        std::vector<bool> assignment(2 * n, false);
        std::size_t remainder = code;
        for (std::size_t pos = n; pos-- > 0;) {
            const int axis = static_cast<int>(remainder & 3u);
            remainder >>= 2;
            auto [x, z] = pauli_axis_bits(axis);
            assignment[2 * pos] = x;
            assignment[2 * pos + 1] = z;
        }
        assignments.push_back(packed_bits(assignment));
        distribution.push_back(check_probability(probabilities[code - 1]));
    }
    const auto bits = state.context->fresh_categorical_bools(static_cast<int>(2 * n), assignments, distribution);
    for (std::size_t i = 0; i < n; ++i) {
        apply_single_qubit_pauli(state, qubits[i], true, false, bits[2 * i]);
        apply_single_qubit_pauli(state, qubits[i], false, true, bits[2 * i + 1]);
    }
}

void apply_depolarize_n(FrameFactoredState& state, const std::vector<int>& qubits, double probability) {
    probability = check_probability(probability);
    std::size_t cases = 1;
    for (std::size_t i = 0; i < qubits.size(); ++i) {
        cases *= 4;
    }
    apply_pauli_channel(state, qubits, std::vector<double>(cases - 1, probability / static_cast<double>(cases - 1)));
}

void apply_pauli_product_channel(
    FrameFactoredState& state,
    const std::vector<CircuitPauliProduct>& products,
    const std::vector<double>& probabilities) {
    if (products.size() != probabilities.size()) {
        fail("Pauli product channel probability count mismatch");
    }
    std::vector<std::vector<std::uint64_t>> assignments;
    assignments.push_back(packed_bits(std::vector<bool>(products.size(), false)));
    std::vector<double> distribution;
    distribution.reserve(products.size() + 1);
    double total = 0.0;
    for (double probability : probabilities) {
        total += check_probability(probability);
    }
    if (total > 1.0 + 1e-12) {
        fail("Pauli product channel probabilities sum to more than 1");
    }
    distribution.push_back(std::max(0.0, 1.0 - total));
    for (std::size_t i = 0; i < products.size(); ++i) {
        std::vector<bool> assignment(products.size(), false);
        assignment[i] = true;
        assignments.push_back(packed_bits(assignment));
        distribution.push_back(check_probability(probabilities[i]));
    }
    const auto bits =
        state.context->fresh_categorical_bools(static_cast<int>(products.size()), assignments, distribution);
    for (std::size_t i = 0; i < products.size(); ++i) {
        apply_pauli(state, maybe_inverted_pauli_product(products[i]), bits[i]);
    }
}

void apply_heralded_channel(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    int q,
    const std::vector<double>& probabilities) {
    if (probabilities.size() != 4) {
        fail("heralded Pauli channel expects four probabilities");
    }
    double total = 0.0;
    for (double probability : probabilities) {
        total += check_probability(probability);
    }
    if (total > 1.0 + 1e-12) {
        fail("heralded channel probabilities sum to more than 1");
    }
    const std::vector<std::vector<std::uint64_t>> assignments{
        packed_bits({false, false, false}),
        packed_bits({true, false, false}),
        packed_bits({true, true, false}),
        packed_bits({true, true, true}),
        packed_bits({true, false, true}),
    };
    const auto bits = state.context->fresh_categorical_bools(
        3,
        assignments,
        {1.0 - total, probabilities[0], probabilities[1], probabilities[2], probabilities[3]});
    records.push_back(bits[0]);
    apply_classical_record(state, bits[0], static_cast<int>(records.size()), std::nullopt);
    apply_single_qubit_pauli(state, q, true, false, bits[1]);
    apply_single_qubit_pauli(state, q, false, true, bits[2]);
}

void apply_mpad(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    const CircuitMeasurementTarget& target,
    double probability) {
    if (target.qubit != 0 && target.qubit != 1) {
        fail("MPAD targets must be 0 or 1");
    }
    SymbolicBool outcome((target.qubit != 0) != target.inverted);
    if (probability != 0.0) {
        outcome = xor_bool(outcome, state.context->fresh_bernoulli_bool(check_probability(probability)));
    }
    records.push_back(outcome);
    apply_classical_record(state, outcome, static_cast<int>(records.size()), std::nullopt);
}

PauliString maybe_inverted_pauli_product(CircuitPauliProduct product) {
    if (product.inverted) {
        product.pauli.phase_shift(2);
    }
    return std::move(product.pauli);
}

void apply_record_feedback(
    FrameFactoredState& state,
    const std::vector<SymbolicBool>& records,
    const CircuitInstruction& instruction) {
    const char axis = instruction.kind == CircuitInstructionKind::FeedbackX
                          ? 'X'
                          : (instruction.kind == CircuitInstructionKind::FeedbackY ? 'Y' : 'Z');
    for (const auto& target : instruction.feedback_targets) {
        apply_pauli_on_axis(state, axis, target.qubit, record_symbol(records, target.record));
    }
}

void apply_reset(
    FrameFactoredState& state,
    const PauliString& measurement_pauli,
    char correction_axis,
    int q) {
    const int condition = state.context->fresh_condition();
    apply_pauli_measurement(state, measurement_pauli, SymbolicBool(false), std::nullopt, condition);
    apply_pauli_on_axis(state, correction_axis, q, symbolic_bool(condition));
}

void apply_measurement_reset(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    const PauliString& measurement_pauli,
    char correction_axis,
    const CircuitMeasurementTarget& target,
    double probability) {
    const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, target.inverted, probability);
    auto [record, record_condition] = reserve_measurement_record(records, *state.context);
    apply_pauli_measurement(state, measurement_pauli, sign, record, record_condition);
    apply_pauli_on_axis(
        state,
        correction_axis,
        target.qubit,
        xor_bool(symbolic_bool(record_condition), sign));
}

void apply_measurement(
    FrameFactoredState& state,
    std::vector<SymbolicBool>& records,
    char axis,
    const CircuitMeasurementTarget& target,
    double probability) {
    const SymbolicBool sign = measurement_sign_with_readout_error(*state.context, target.inverted, probability);
    auto [record, record_condition] = reserve_measurement_record(records, *state.context);
    apply_pauli_measurement(state, pauli_on_axis(state.n, axis, target.qubit), sign, record, record_condition);
}

void apply_instruction(CircuitLoweringAccumulator& acc, const CircuitInstruction& instruction) {
    FrameFactoredState& state = acc.state;
    switch (instruction.kind) {
    case CircuitInstructionKind::Tick:
        return;
    case CircuitInstructionKind::H:
    case CircuitInstructionKind::H_NXY:
    case CircuitInstructionKind::H_NXZ:
    case CircuitInstructionKind::H_NYZ:
    case CircuitInstructionKind::H_XY:
    case CircuitInstructionKind::H_YZ:
    case CircuitInstructionKind::C_NXYZ:
    case CircuitInstructionKind::C_NZYX:
    case CircuitInstructionKind::C_XNYZ:
    case CircuitInstructionKind::C_XYNZ:
    case CircuitInstructionKind::C_XYZ:
    case CircuitInstructionKind::C_ZNYX:
    case CircuitInstructionKind::C_ZYNX:
    case CircuitInstructionKind::C_ZYX:
    case CircuitInstructionKind::S:
    case CircuitInstructionKind::SDG:
    case CircuitInstructionKind::SqrtX:
    case CircuitInstructionKind::SqrtXDag:
    case CircuitInstructionKind::SqrtY:
    case CircuitInstructionKind::SqrtYDag:
    case CircuitInstructionKind::X:
    case CircuitInstructionKind::Y:
    case CircuitInstructionKind::Z:
        for (int q : instruction.qubits) {
            apply_single_qubit_clifford(state, instruction.kind, q);
        }
        return;
    case CircuitInstructionKind::CX:
    case CircuitInstructionKind::CY:
    case CircuitInstructionKind::CZ:
    case CircuitInstructionKind::SWAP:
    case CircuitInstructionKind::CXSWAP:
    case CircuitInstructionKind::CZSWAP:
    case CircuitInstructionKind::ISWAP:
    case CircuitInstructionKind::ISWAP_DAG:
    case CircuitInstructionKind::SQRT_XX:
    case CircuitInstructionKind::SQRT_XX_DAG:
    case CircuitInstructionKind::SQRT_YY:
    case CircuitInstructionKind::SQRT_YY_DAG:
    case CircuitInstructionKind::SQRT_ZZ:
    case CircuitInstructionKind::SQRT_ZZ_DAG:
    case CircuitInstructionKind::SWAPCX:
    case CircuitInstructionKind::XCX:
    case CircuitInstructionKind::XCY:
    case CircuitInstructionKind::XCZ:
    case CircuitInstructionKind::YCX:
    case CircuitInstructionKind::YCY:
    case CircuitInstructionKind::YCZ:
        require_even_targets(instruction, "two-qubit Clifford requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            apply_two_qubit_clifford(state, instruction.kind, instruction.qubits[i], instruction.qubits[i + 1]);
        }
        return;
    case CircuitInstructionKind::T:
    case CircuitInstructionKind::TDag: {
        const double kernel_angle = instruction.kind == CircuitInstructionKind::T ? kPi / 8.0 : -kPi / 8.0;
        for (int q : instruction.qubits) {
            apply_pauli_rotation(state, pauli_z(state.n, q), kernel_angle);
        }
        return;
    }
    case CircuitInstructionKind::PauliRotation:
        for (const auto& product : instruction.pauli_products) {
            apply_pauli_rotation(state, maybe_inverted_pauli_product(product), instruction.kernel_angle);
        }
        return;
    case CircuitInstructionKind::MZ:
    case CircuitInstructionKind::MX:
    case CircuitInstructionKind::MY: {
        const char axis = instruction.kind == CircuitInstructionKind::MX
                              ? 'X'
                              : (instruction.kind == CircuitInstructionKind::MY ? 'Y' : 'Z');
        for (const auto& target : instruction.measurement_targets) {
            apply_measurement(state, acc.records, axis, target, instruction.probability);
        }
        return;
    }
    case CircuitInstructionKind::MRZ:
    case CircuitInstructionKind::MRX:
    case CircuitInstructionKind::MRY:
        for (const auto& target : instruction.measurement_targets) {
            if (instruction.kind == CircuitInstructionKind::MRX) {
                apply_measurement_reset(
                    state,
                    acc.records,
                    pauli_x(state.n, target.qubit),
                    'Z',
                    target,
                    instruction.probability);
            } else if (instruction.kind == CircuitInstructionKind::MRY) {
                apply_measurement_reset(
                    state,
                    acc.records,
                    pauli_y(state.n, target.qubit),
                    'X',
                    target,
                    instruction.probability);
            } else {
                apply_measurement_reset(
                    state,
                    acc.records,
                    pauli_z(state.n, target.qubit),
                    'X',
                    target,
                    instruction.probability);
            }
        }
        return;
    case CircuitInstructionKind::RZ:
    case CircuitInstructionKind::RX:
    case CircuitInstructionKind::RY:
        for (int q : instruction.qubits) {
            if (instruction.kind == CircuitInstructionKind::RX) {
                apply_reset(state, pauli_x(state.n, q), 'Z', q);
            } else if (instruction.kind == CircuitInstructionKind::RY) {
                apply_reset(state, pauli_y(state.n, q), 'X', q);
            } else {
                apply_reset(state, pauli_z(state.n, q), 'X', q);
            }
        }
        return;
    case CircuitInstructionKind::MPP:
        for (const auto& product : instruction.pauli_products) {
            const SymbolicBool sign =
                measurement_sign_with_readout_error(*state.context, product.inverted, instruction.probability);
            auto [record, record_condition] = reserve_measurement_record(acc.records, *state.context);
            apply_pauli_measurement(state, product.pauli, sign, record, record_condition);
        }
        return;
    case CircuitInstructionKind::EXP_VAL:
        if (instruction.exp_val < 0) {
            fail("EXP_VAL index range is invalid");
        }
        for (std::size_t i = 0; i < instruction.pauli_products.size(); ++i) {
            const auto& product = instruction.pauli_products[i];
            PauliString pauli = maybe_inverted_pauli_product(product);
            apply_pauli_expectation(state, pauli, instruction.exp_val + static_cast<int>(i));
        }
        return;
    case CircuitInstructionKind::XError:
    case CircuitInstructionKind::YError:
    case CircuitInstructionKind::ZError: {
        const double probability = check_probability(instruction.probability);
        const char axis = instruction.kind == CircuitInstructionKind::XError
                              ? 'X'
                              : (instruction.kind == CircuitInstructionKind::YError ? 'Y' : 'Z');
        for (int q : instruction.qubits) {
            apply_pauli_on_axis(state, axis, q, state.context->fresh_bernoulli_bool(probability));
        }
        return;
    }
    case CircuitInstructionKind::Depolarize1:
        for (int q : instruction.qubits) {
            apply_depolarize1(state, q, instruction.probability);
        }
        return;
    case CircuitInstructionKind::Depolarize2:
        require_even_targets(instruction, "DEPOLARIZE2 requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            apply_depolarize2(state, instruction.qubits[i], instruction.qubits[i + 1], instruction.probability);
        }
        return;
    case CircuitInstructionKind::Depolarize3:
        require_target_multiple(instruction, 3, "DEPOLARIZE3 requires triples of targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 3) {
            apply_depolarize_n(
                state,
                {instruction.qubits[i], instruction.qubits[i + 1], instruction.qubits[i + 2]},
                instruction.probability);
        }
        return;
    case CircuitInstructionKind::PauliChannel1:
        for (int q : instruction.qubits) {
            apply_pauli_channel(state, {q}, instruction.probabilities);
        }
        return;
    case CircuitInstructionKind::PauliChannel2:
        require_even_targets(instruction, "PAULI_CHANNEL_2 requires paired targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 2) {
            apply_pauli_channel(state, {instruction.qubits[i], instruction.qubits[i + 1]}, instruction.probabilities);
        }
        return;
    case CircuitInstructionKind::PauliChannel3:
        require_target_multiple(instruction, 3, "PAULI_CHANNEL_3 requires triples of targets");
        for (std::size_t i = 0; i < instruction.qubits.size(); i += 3) {
            apply_pauli_channel(
                state,
                {instruction.qubits[i], instruction.qubits[i + 1], instruction.qubits[i + 2]},
                instruction.probabilities);
        }
        return;
    case CircuitInstructionKind::PauliProductError:
        for (const auto& product : instruction.pauli_products) {
            apply_pauli(
                state,
                maybe_inverted_pauli_product(product),
                state.context->fresh_bernoulli_bool(instruction.probability));
        }
        return;
    case CircuitInstructionKind::PauliProductChannel:
        apply_pauli_product_channel(state, instruction.pauli_products, instruction.probabilities);
        return;
    case CircuitInstructionKind::HeraldedErase: {
        const double p = check_probability(instruction.probability) / 4.0;
        for (int q : instruction.qubits) {
            apply_heralded_channel(state, acc.records, q, {p, p, p, p});
        }
        return;
    }
    case CircuitInstructionKind::HeraldedPauliChannel1:
        for (int q : instruction.qubits) {
            apply_heralded_channel(state, acc.records, q, instruction.probabilities);
        }
        return;
    case CircuitInstructionKind::MPad:
        for (const auto& target : instruction.measurement_targets) {
            apply_mpad(state, acc.records, target, instruction.probability);
        }
        return;
    case CircuitInstructionKind::FeedbackX:
    case CircuitInstructionKind::FeedbackY:
    case CircuitInstructionKind::FeedbackZ:
        apply_record_feedback(state, acc.records, instruction);
        return;
    }
    fail("unsupported circuit operation");
}

template <typename T>
std::uint64_t count_targets(const std::vector<T>& targets) {
    return static_cast<std::uint64_t>(targets.size());
}

std::uint64_t count_pauli_product_touches(
    const std::vector<CircuitPauliProduct>& products) {
    std::uint64_t count = 0;
    for (const auto& product : products) {
        for (std::size_t word = 0; word < product.pauli.x.size(); ++word) {
            count += static_cast<std::uint64_t>(
                popcount64(product.pauli.x[word] | product.pauli.z[word]));
        }
    }
    return count;
}

std::uint64_t estimated_record_terms(
    const std::vector<std::uint64_t>& records,
    int record) {
    if (record <= 0 || record > static_cast<int>(records.size())) {
        fail("measurement record target out of range in factorization estimator");
    }
    return records[static_cast<std::size_t>(record - 1)];
}

bool is_phase_only_clifford(CircuitInstructionKind kind) {
    return kind == CircuitInstructionKind::X ||
           kind == CircuitInstructionKind::Y ||
           kind == CircuitInstructionKind::Z;
}

long double intra_instruction_correction_history_work(
    const CircuitInstruction& instruction) {
    long double work = 0.0;
    std::uint64_t preceding_corrections = 0;
    if (instruction.kind == CircuitInstructionKind::MRZ ||
        instruction.kind == CircuitInstructionKind::MRX ||
        instruction.kind == CircuitInstructionKind::MRY) {
        for (const auto& target : instruction.measurement_targets) {
            work += static_cast<long double>(preceding_corrections);
            preceding_corrections +=
                1 + static_cast<std::uint64_t>(instruction.probability != 0.0) +
                static_cast<std::uint64_t>(target.inverted);
        }
    } else if (instruction.kind == CircuitInstructionKind::RZ ||
               instruction.kind == CircuitInstructionKind::RX ||
               instruction.kind == CircuitInstructionKind::RY) {
        const long double targets =
            static_cast<long double>(instruction.qubits.size());
        work = targets * (targets - 1.0L) / 2.0L;
    }
    return work;
}

void accumulate_factorization_counts(
    CircuitFactorizationEstimate& estimate,
    std::vector<std::uint64_t>& records,
    const CircuitInstruction& instruction) {
    switch (instruction.kind) {
    case CircuitInstructionKind::Tick:
        return;
    case CircuitInstructionKind::MPad:
        for (const auto& target : instruction.measurement_targets) {
            records.push_back(
                static_cast<std::uint64_t>((target.qubit != 0) != target.inverted) +
                static_cast<std::uint64_t>(instruction.probability != 0.0));
        }
        return;
    case CircuitInstructionKind::H:
    case CircuitInstructionKind::H_NXY:
    case CircuitInstructionKind::H_NXZ:
    case CircuitInstructionKind::H_NYZ:
    case CircuitInstructionKind::H_XY:
    case CircuitInstructionKind::H_YZ:
    case CircuitInstructionKind::C_NXYZ:
    case CircuitInstructionKind::C_NZYX:
    case CircuitInstructionKind::C_XNYZ:
    case CircuitInstructionKind::C_XYNZ:
    case CircuitInstructionKind::C_XYZ:
    case CircuitInstructionKind::C_ZNYX:
    case CircuitInstructionKind::C_ZYNX:
    case CircuitInstructionKind::C_ZYX:
    case CircuitInstructionKind::S:
    case CircuitInstructionKind::SDG:
    case CircuitInstructionKind::SqrtX:
    case CircuitInstructionKind::SqrtXDag:
    case CircuitInstructionKind::SqrtY:
    case CircuitInstructionKind::SqrtYDag:
    case CircuitInstructionKind::X:
    case CircuitInstructionKind::Y:
    case CircuitInstructionKind::Z:
        estimate.clifford_operations += count_targets(instruction.qubits);
        estimate.pullback_event_qubit_touches += count_targets(instruction.qubits);
        return;
    case CircuitInstructionKind::CX:
    case CircuitInstructionKind::CY:
    case CircuitInstructionKind::CZ:
    case CircuitInstructionKind::SWAP:
    case CircuitInstructionKind::CXSWAP:
    case CircuitInstructionKind::CZSWAP:
    case CircuitInstructionKind::ISWAP:
    case CircuitInstructionKind::ISWAP_DAG:
    case CircuitInstructionKind::SQRT_XX:
    case CircuitInstructionKind::SQRT_XX_DAG:
    case CircuitInstructionKind::SQRT_YY:
    case CircuitInstructionKind::SQRT_YY_DAG:
    case CircuitInstructionKind::SQRT_ZZ:
    case CircuitInstructionKind::SQRT_ZZ_DAG:
    case CircuitInstructionKind::SWAPCX:
    case CircuitInstructionKind::XCX:
    case CircuitInstructionKind::XCY:
    case CircuitInstructionKind::XCZ:
    case CircuitInstructionKind::YCX:
    case CircuitInstructionKind::YCY:
    case CircuitInstructionKind::YCZ:
        estimate.clifford_operations += count_targets(instruction.qubits) / 2;
        estimate.pullback_event_qubit_touches += count_targets(instruction.qubits);
        return;
    case CircuitInstructionKind::T:
    case CircuitInstructionKind::TDag:
        estimate.pending_pauli_operations += count_targets(instruction.qubits);
        return;
    case CircuitInstructionKind::PauliRotation:
    case CircuitInstructionKind::EXP_VAL:
        estimate.pending_pauli_operations += count_targets(instruction.pauli_products);
        return;
    case CircuitInstructionKind::MPP:
        estimate.pending_pauli_operations += count_targets(instruction.pauli_products);
        records.insert(
            records.end(),
            instruction.pauli_products.size(),
            1);
        return;
    case CircuitInstructionKind::MZ:
    case CircuitInstructionKind::MX:
    case CircuitInstructionKind::MY:
        estimate.pending_pauli_operations += count_targets(instruction.measurement_targets);
        records.insert(
            records.end(),
            instruction.measurement_targets.size(),
            1);
        return;
    case CircuitInstructionKind::MRZ:
    case CircuitInstructionKind::MRX:
    case CircuitInstructionKind::MRY:
        estimate.pending_pauli_operations += count_targets(instruction.measurement_targets);
        for (const auto& target : instruction.measurement_targets) {
            records.push_back(1);
            // The reset correction is controlled by the measurement result,
            // the optional readout-error bit, and an optional constant sign.
            estimate.conditional_pauli_operations +=
                1 + static_cast<std::uint64_t>(instruction.probability != 0.0) +
                static_cast<std::uint64_t>(target.inverted);
            estimate.pullback_event_qubit_touches +=
                1 + static_cast<std::uint64_t>(instruction.probability != 0.0) +
                static_cast<std::uint64_t>(target.inverted);
        }
        return;
    case CircuitInstructionKind::RZ:
    case CircuitInstructionKind::RX:
    case CircuitInstructionKind::RY:
        estimate.pending_pauli_operations += count_targets(instruction.qubits);
        estimate.conditional_pauli_operations += count_targets(instruction.qubits);
        estimate.pullback_event_qubit_touches += count_targets(instruction.qubits);
        return;
    case CircuitInstructionKind::XError:
    case CircuitInstructionKind::YError:
    case CircuitInstructionKind::ZError:
        estimate.conditional_pauli_operations += count_targets(instruction.qubits);
        estimate.pullback_event_qubit_touches += count_targets(instruction.qubits);
        return;
    case CircuitInstructionKind::Depolarize1:
    case CircuitInstructionKind::PauliChannel1:
    case CircuitInstructionKind::Depolarize2:
    case CircuitInstructionKind::PauliChannel2:
    case CircuitInstructionKind::Depolarize3:
    case CircuitInstructionKind::PauliChannel3:
        // Each target contributes one symbolic X and one symbolic Z term.
        estimate.conditional_pauli_operations += 2 * count_targets(instruction.qubits);
        estimate.pullback_event_qubit_touches += 2 * count_targets(instruction.qubits);
        return;
    case CircuitInstructionKind::PauliProductError:
    case CircuitInstructionKind::PauliProductChannel:
        estimate.conditional_pauli_operations += count_targets(instruction.pauli_products);
        estimate.pullback_event_qubit_touches +=
            count_pauli_product_touches(instruction.pauli_products);
        return;
    case CircuitInstructionKind::HeraldedErase:
    case CircuitInstructionKind::HeraldedPauliChannel1:
        estimate.conditional_pauli_operations += 2 * count_targets(instruction.qubits);
        estimate.pullback_event_qubit_touches += 2 * count_targets(instruction.qubits);
        records.insert(
            records.end(),
            instruction.qubits.size(),
            1);
        return;
    case CircuitInstructionKind::FeedbackX:
    case CircuitInstructionKind::FeedbackY:
    case CircuitInstructionKind::FeedbackZ:
        for (const auto& target : instruction.feedback_targets) {
            const std::uint64_t terms = estimated_record_terms(records, target.record);
            estimate.conditional_pauli_operations += terms;
            estimate.pullback_event_qubit_touches += terms;
        }
        return;
    }
    fail("unsupported circuit operation in factorization estimator");
}

struct FactorizationHistory {
    long double correction_work = 0.0;
    long double pullback_touch_work = 0.0;
    std::uint64_t support_rebuilds = 0;
    bool support_valid = false;
};

void select_factorization_strategy(
    CircuitFactorizationEstimate& estimate,
    int nqubits,
    const FactorizationHistory& history) {
    const long double n = static_cast<long double>(std::max(1, nqubits));
    const long double nc = static_cast<long double>(estimate.clifford_operations);
    const long double ne = static_cast<long double>(estimate.conditional_pauli_operations);
    const long double np = static_cast<long double>(estimate.pending_pauli_operations);

    estimate.frame_work = n * nc + n * ne + (n * n + n * ne) * np;
    estimate.direct_pullback_work = np * (nc + ne);
    if (estimate.pending_pauli_operations == 0) {
        return;
    }

    // Small collections use the indexed sparse pullback only while their
    // total history remains smaller than a modest multiple of one tableau.
    if (estimate.pending_pauli_operations < detail::kMinimumBatchedPullbackCount) {
        constexpr long double kLocalityBudgetPerTableauEntry = 8.0L;
        const long double indexed_work =
            np * static_cast<long double>(estimate.pullback_event_qubit_touches);
        if (estimate.direct_pullback_work < estimate.frame_work &&
            indexed_work <= kLocalityBudgetPerTableauEntry * n * n) {
            estimate.preferred_strategy = FactorizationStrategy::DirectPullback;
        }
        return;
    }

    const long double pauli_words = std::ceil(n / kWordBits);
    const long double pending_words = std::ceil(np / kWordBits);
    const long double frame_cost =
        pauli_words * (nc + ne + np) +
        history.correction_work / kWordBits +
        2.0L * n * pauli_words * history.support_rebuilds;
    const long double pullback_cost =
        history.pullback_touch_work / kWordBits +
        static_cast<long double>(estimate.pullback_event_qubit_touches) +
        2.0L * n * pending_words;

    // The startup floor and 5% margin keep tiny or near-tied cases on the
    // established frame implementation.
    constexpr long double kMinimumBatchedWork =
        static_cast<long double>(kWordBits) * kWordBits;
    if (pullback_cost >= kMinimumBatchedWork &&
        1.05L * pullback_cost < frame_cost) {
        estimate.preferred_strategy = FactorizationStrategy::DirectPullback;
    }
}

} // namespace

CircuitFactorizationEstimate estimate_circuit_factorization(const QuantumCircuit& circuit) {
    if (circuit.nrecords < 0) {
        fail("circuit measurement record count is negative");
    }
    CircuitFactorizationEstimate estimate;
    std::vector<std::uint64_t> records;
    records.reserve(static_cast<std::size_t>(circuit.nrecords));
    FactorizationHistory history;
    for (const auto& instruction : circuit.instructions) {
        const std::uint64_t pending_before = estimate.pending_pauli_operations;
        const std::uint64_t cliffords_before = estimate.clifford_operations;
        const std::uint64_t corrections_before =
            estimate.conditional_pauli_operations;
        const std::uint64_t history_touches_before =
            estimate.pullback_event_qubit_touches;
        accumulate_factorization_counts(estimate, records, instruction);
        const std::uint64_t pending_delta =
            estimate.pending_pauli_operations - pending_before;
        const long double intra_instruction_correction_work =
            intra_instruction_correction_history_work(instruction);
        history.pullback_touch_work +=
            static_cast<long double>(pending_delta) * history_touches_before +
            intra_instruction_correction_work;
        history.correction_work +=
            static_cast<long double>(pending_delta) * corrections_before +
            intra_instruction_correction_work;
        const std::uint64_t clifford_delta =
            estimate.clifford_operations - cliffords_before;
        const std::uint64_t correction_delta =
            estimate.conditional_pauli_operations - corrections_before;
        if (clifford_delta != 0 && !is_phase_only_clifford(instruction.kind)) {
            history.support_valid = false;
        }
        if ((pending_delta != 0 || correction_delta != 0) &&
            !history.support_valid) {
            ++history.support_rebuilds;
            history.support_valid = true;
        }
    }
    if (static_cast<int>(records.size()) != circuit.nrecords) {
        fail("circuit measurement record count mismatch in factorization estimator");
    }

    // This selector deliberately models factorization only. Both strategies
    // produce identical pending operations, so planning is a common downstream
    // cost and must not influence the choice.
    select_factorization_strategy(estimate, circuit.nqubits, history);
    return estimate;
}

CircuitLoweringResult lower_circuit_to_factored(
    const QuantumCircuit& circuit,
    FactorizationStrategy strategy) {
    CircuitFactorizationEstimate estimate = estimate_circuit_factorization(circuit);
    if (strategy == FactorizationStrategy::Automatic) {
        strategy = estimate.preferred_strategy;
    }
    CircuitLoweringAccumulator acc{
        FrameFactoredState(circuit.nqubits, 0, strategy),
        {}};
    if (strategy == FactorizationStrategy::DirectPullback) {
        begin_deferred_direct_pullback(acc.state);
    }
    std::vector<int> pending_counts;
    pending_counts.reserve(circuit.instructions.size() + 1);
    pending_counts.push_back(0);
    for (const auto& instruction : circuit.instructions) {
        apply_instruction(acc, instruction);
        pending_counts.push_back(static_cast<int>(acc.state.pending_operations.size()));
    }
    if (static_cast<int>(acc.records.size()) != circuit.nrecords) {
        fail("circuit measurement record count mismatch");
    }
    finish_deferred_direct_pullback(acc.state);
    return CircuitLoweringResult{
        std::move(acc.state),
        std::move(acc.records),
        std::move(pending_counts),
        std::move(estimate),
        strategy,
    };
}

} // namespace symft
