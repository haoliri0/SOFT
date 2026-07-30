#include "core/internal.hpp"

#include "frontend/stim.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace symft {
using namespace detail;

namespace {

std::string trim(const std::string& input) {
    std::size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) {
        ++first;
    }
    std::size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) {
        --last;
    }
    return input.substr(first, last - first);
}

std::vector<std::string> split_ws(const std::string& input) {
    std::istringstream stream(input);
    std::vector<std::string> out;
    std::string item;
    while (stream >> item) {
        out.push_back(item);
    }
    return out;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string upper(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool all_digits(const std::string& value) {
    return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char ch) { return std::isdigit(ch); });
}

class ArgumentExpressionParser {
public:
    explicit ArgumentExpressionParser(std::string input_) : input(std::move(input_)) {}

    double parse() {
        const double value = parse_expression();
        skip_ws();
        if (pos != input.size()) {
            fail("invalid numeric expression");
        }
        return value;
    }

private:
    std::string input;
    std::size_t pos = 0;

    void skip_ws() {
        while (pos < input.size() && std::isspace(static_cast<unsigned char>(input[pos]))) {
            ++pos;
        }
    }

    bool consume(char ch) {
        skip_ws();
        if (pos < input.size() && input[pos] == ch) {
            ++pos;
            return true;
        }
        return false;
    }

    double parse_expression() {
        double value = parse_term();
        while (true) {
            if (consume('+')) {
                value += parse_term();
            } else if (consume('-')) {
                value -= parse_term();
            } else {
                return value;
            }
        }
    }

    double parse_term() {
        double value = parse_factor();
        while (true) {
            if (consume('*')) {
                value *= parse_factor();
            } else if (consume('/')) {
                value /= parse_factor();
            } else {
                return value;
            }
        }
    }

    double parse_factor() {
        skip_ws();
        if (consume('+')) {
            return parse_factor();
        }
        if (consume('-')) {
            return -parse_factor();
        }
        if (consume('(')) {
            const double value = parse_expression();
            if (!consume(')')) {
                fail("unterminated numeric expression");
            }
            return value;
        }
        if (pos < input.size() && std::isalpha(static_cast<unsigned char>(input[pos]))) {
            const std::size_t start = pos;
            while (pos < input.size() && std::isalpha(static_cast<unsigned char>(input[pos]))) {
                ++pos;
            }
            const std::string name = upper(input.substr(start, pos - start));
            if (name == "PI") {
                return kPi;
            }
            fail("unknown numeric constant: " + name);
        }
        const char* begin = input.c_str() + pos;
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        if (end == begin) {
            fail("invalid numeric expression");
        }
        pos = static_cast<std::size_t>(end - input.c_str());
        return value;
    }
};

double parse_numeric_expression(const std::string& input) {
    char* end = nullptr;
    const double value = std::strtod(input.c_str(), &end);
    if (end != input.c_str() && *end == '\0') {
        return value;
    }
    return ArgumentExpressionParser(input).parse();
}

std::size_t find_matching_paren(const std::string& body, std::size_t open) {
    int depth = 0;
    for (std::size_t i = open; i < body.size(); ++i) {
        if (body[i] == '(') {
            ++depth;
        } else if (body[i] == ')') {
            --depth;
            if (depth == 0) {
                return i;
            }
        }
    }
    fail("unterminated Stim argument list");
}

std::vector<std::string> split_arguments(const std::string& args) {
    std::vector<std::string> out;
    std::size_t start = 0;
    int depth = 0;
    for (std::size_t i = 0; i <= args.size(); ++i) {
        const char ch = i < args.size() ? args[i] : ',';
        if (ch == '(') {
            ++depth;
        } else if (ch == ')') {
            --depth;
            if (depth < 0) {
                fail("invalid numeric argument nesting");
            }
        } else if (ch == ',' && depth == 0) {
            out.push_back(trim(args.substr(start, i - start)));
            start = i + 1;
        }
    }
    if (depth != 0) {
        fail("invalid numeric argument nesting");
    }
    return out;
}

std::string strip_stim_comment(const std::string& line) {
    bool in_brackets = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (ch == '[') {
            in_brackets = true;
        } else if (ch == ']') {
            in_brackets = false;
        } else if (ch == '#' && !in_brackets) {
            return trim(line.substr(0, i));
        }
    }
    return trim(line);
}

struct StimInstruction {
    std::string op;
    std::optional<std::string> tag;
    bool has_parens = false;
    std::vector<double> parens;
    std::vector<std::string> targets;
    int line = 0;
};

struct StimRepeatBlock;
using StimNode = std::variant<StimInstruction, std::shared_ptr<StimRepeatBlock>>;

struct StimRepeatBlock {
    int count = 0;
    std::vector<StimNode> body;
    int line = 0;
};

constexpr std::uint64_t kStimMaxRepeatCount = 1000000000000000000ull;

std::uint64_t parse_bounded_unsigned_integer(
    const std::string& token,
    std::uint64_t max_value,
    const char* message) {
    if (!all_digits(token)) {
        fail(message);
    }
    std::uint64_t value = 0;
    for (char ch : token) {
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - '0');
        if (value > (max_value - digit) / 10) {
            fail(message);
        }
        value = value * 10 + digit;
    }
    return value;
}

int parse_repeat_count(const std::string& token) {
    const std::uint64_t count =
        parse_bounded_unsigned_integer(token, kStimMaxRepeatCount, "REPEAT count must be in [1, 10^18]");
    if (count == 0) {
        fail("REPEAT count must be in [1, 10^18]");
    }
    if (count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        fail("REPEAT count is too large for this flattened circuit frontend");
    }
    return static_cast<int>(count);
}

StimInstruction parse_instruction(const std::string& body, int line) {
    std::size_t idx = 0;
    while (idx < body.size() && (std::isalpha(static_cast<unsigned char>(body[idx])) ||
                                 (idx > 0 && (std::isdigit(static_cast<unsigned char>(body[idx])) || body[idx] == '_')))) {
        ++idx;
    }
    if (idx == 0) {
        fail("invalid Stim instruction");
    }
    StimInstruction inst;
    inst.op = upper(body.substr(0, idx));
    inst.line = line;
    if (idx < body.size() && body[idx] == '[') {
        const std::size_t close = body.find(']', idx + 1);
        if (close == std::string::npos) {
            fail("unterminated Stim tag");
        }
        inst.tag = body.substr(idx + 1, close - idx - 1);
        idx = close + 1;
    }
    if (idx < body.size() && body[idx] == '(') {
        inst.has_parens = true;
        const std::size_t close = find_matching_paren(body, idx);
        const std::string args = trim(body.substr(idx + 1, close - idx - 1));
        if (!args.empty()) {
            for (const auto& piece : split_arguments(args)) {
                inst.parens.push_back(parse_numeric_expression(piece));
            }
        }
        idx = close + 1;
    }
    if (idx < body.size()) {
        const std::string rest = trim(body.substr(idx));
        inst.targets = split_ws(rest);
    }
    return inst;
}

std::vector<StimNode> parse_nodes(const std::vector<std::string>& lines, std::size_t& pos, bool in_block) {
    std::vector<StimNode> nodes;
    while (pos < lines.size()) {
        std::string body = strip_stim_comment(lines[pos]);
        const int line = static_cast<int>(pos + 1);
        ++pos;
        if (body.empty()) {
            continue;
        }
        if (body == "}") {
            if (!in_block) {
                fail("unmatched Stim block terminator");
            }
            return nodes;
        }
        bool block_start = false;
        if (ends_with(body, "{")) {
            block_start = true;
            body = trim(body.substr(0, body.size() - 1));
        }
        StimInstruction inst = parse_instruction(body, line);
        if (block_start) {
            if (inst.op != "REPEAT" || inst.has_parens || inst.targets.size() != 1) {
                fail("invalid Stim REPEAT block");
            }
            auto block = std::make_shared<StimRepeatBlock>();
            block->count = parse_repeat_count(inst.targets[0]);
            block->line = line;
            block->body = parse_nodes(lines, pos, true);
            nodes.push_back(block);
        } else {
            if (inst.op == "REPEAT") {
                fail("REPEAT must start a block");
            }
            nodes.push_back(inst);
        }
    }
    if (in_block) {
        fail("unterminated Stim block");
    }
    return nodes;
}

std::pair<int, bool> stim_qubit_target(const std::string& target, int line) {
    bool inverted = false;
    std::string body = target;
    if (starts_with(body, "!")) {
        inverted = true;
        body = body.substr(1);
    }
    if (!all_digits(body)) {
        (void)line;
        fail("invalid qubit target");
    }
    return {std::stoi(body), inverted};
}

std::vector<int> stim_qubit_targets(const StimInstruction& inst) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        auto [q, inverted] = stim_qubit_target(target, inst.line);
        if (inverted) {
            fail("operation does not accept inverted targets");
        }
        out.push_back(q);
    }
    return out;
}

std::vector<std::pair<int, bool>> stim_measurement_targets(const StimInstruction& inst) {
    std::vector<std::pair<int, bool>> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_qubit_target(target, inst.line));
    }
    return out;
}

bool is_record_target(const std::string& target) {
    if (!starts_with(target, "rec[") || !ends_with(target, "]")) {
        return false;
    }
    const std::string offset = target.substr(4, target.size() - 5);
    return offset.size() >= 2 && offset[0] == '-' && all_digits(offset.substr(1));
}

int stim_record_index(const std::string& target, int nrecords) {
    if (!is_record_target(target)) {
        fail("invalid measurement record target");
    }
    const int offset = std::stoi(target.substr(4, target.size() - 5));
    const int idx = nrecords + offset + 1;
    if (idx <= 0 || idx > nrecords) {
        fail("measurement record target out of range");
    }
    return idx;
}

std::vector<int> stim_record_indices(const StimInstruction& inst, int nrecords) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        out.push_back(stim_record_index(target, nrecords));
    }
    return out;
}

bool is_sweep_target(const std::string& target) {
    if (!starts_with(target, "sweep[") || !ends_with(target, "]")) {
        return false;
    }
    return all_digits(target.substr(6, target.size() - 7));
}

bool is_pauli_target_or_combiner(std::string target) {
    if (target == "*") {
        return true;
    }
    if (starts_with(target, "!")) {
        target = target.substr(1);
    }
    return target.size() >= 2 && (target[0] == 'X' || target[0] == 'Y' || target[0] == 'Z') &&
           all_digits(target.substr(1));
}

std::vector<int> stim_observable_record_indices(const StimInstruction& inst, int nrecords) {
    std::vector<int> out;
    for (const auto& target : inst.targets) {
        if (is_record_target(target)) {
            out.push_back(stim_record_index(target, nrecords));
        } else if (!is_pauli_target_or_combiner(target)) {
            fail("OBSERVABLE_INCLUDE targets must be measurement records or Pauli targets");
        }
    }
    return out;
}

double stim_paren_probability(const StimInstruction& inst, double default_probability = 0.0) {
    if (!inst.has_parens) {
        return default_probability;
    }
    if (inst.parens.size() != 1) {
        fail("operation expects at most one probability argument");
    }
    return check_probability(inst.parens[0]);
}

double stim_required_probability(const StimInstruction& inst) {
    if (inst.parens.size() != 1) {
        fail("operation expects one probability argument");
    }
    return check_probability(inst.parens[0]);
}

void require_paren_count(const StimInstruction& inst, std::size_t count, const char* message) {
    if (inst.parens.size() != count) {
        fail(message);
    }
}

void require_no_parens(const StimInstruction& inst) {
    if (inst.has_parens) {
        fail("operation does not accept parens arguments");
    }
}

void require_no_targets(const StimInstruction& inst) {
    if (!inst.targets.empty()) {
        fail("operation does not accept targets");
    }
}

void require_coordinate_count(
    const StimInstruction& inst,
    bool parens_required,
    const char* message) {
    if (inst.has_parens) {
        if (inst.parens.empty() || inst.parens.size() > 16) {
            fail(message);
        }
    } else if (parens_required) {
        fail(message);
    }
}

int nonnegative_integer_argument(double value, const char* message) {
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value ||
        value > static_cast<double>(std::numeric_limits<int>::max())) {
        fail(message);
    }
    return static_cast<int>(value);
}

void check_disjoint_probability_list(const StimInstruction& inst, const char* message) {
    if (inst.has_parens && inst.parens.empty()) {
        fail(message);
    }
    double total = 0.0;
    for (double p : inst.parens) {
        total += check_probability(p);
    }
    if (total > 1.0 + 1e-12) {
        fail(message);
    }
}

int target_max_qubit(const std::string& target) {
    if (target == "*" || is_record_target(target) || is_sweep_target(target)) {
        return -1;
    }
    std::string body = target;
    if (starts_with(body, "!")) {
        body = body.substr(1);
    }
    int max_q = -1;
    std::size_t start = 0;
    while (start <= body.size()) {
        const std::size_t star = body.find('*', start);
        std::string factor = body.substr(start, star == std::string::npos ? std::string::npos : star - start);
        if (starts_with(factor, "!")) {
            factor = factor.substr(1);
        }
        if (!factor.empty() && (factor[0] == 'X' || factor[0] == 'Y' || factor[0] == 'Z')) {
            factor = factor.substr(1);
        }
        if (!factor.empty() && all_digits(factor)) {
            max_q = std::max(max_q, std::stoi(factor));
        }
        if (star == std::string::npos) {
            break;
        }
        start = star + 1;
    }
    return max_q;
}

int all_qubits(const std::vector<StimNode>& nodes) {
    int max_q = -1;
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            max_q = std::max(max_q, all_qubits((*block)->body) - 1);
            continue;
        }
        const auto& inst = std::get<StimInstruction>(node);
        if (inst.op == "DETECTOR" || inst.op == "OBSERVABLE_INCLUDE" || inst.op == "MPAD" ||
            inst.op == "TICK" || inst.op == "SHIFT_COORDS") {
            continue;
        }
        for (const auto& target : inst.targets) {
            max_q = std::max(max_q, target_max_qubit(target));
        }
    }
    return max_q + 1;
}

PauliString pauli_on_target(int n, char axis, int q) {
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

std::pair<PauliString, bool> stim_mpp_pauli(int n, const std::string& target) {
    PauliString out = pauli_identity(n);
    bool inverted = false;
    std::size_t start = 0;
    while (start <= target.size()) {
        const std::size_t star = target.find('*', start);
        std::string factor = target.substr(start, star == std::string::npos ? std::string::npos : star - start);
        if (starts_with(factor, "!")) {
            inverted = !inverted;
            factor = factor.substr(1);
        }
        if (factor.size() < 2 || !(factor[0] == 'X' || factor[0] == 'Y' || factor[0] == 'Z')) {
            fail("invalid MPP factor");
        }
        const int q = std::stoi(factor.substr(1));
        out = out * pauli_on_target(n, factor[0], q);
        if (star == std::string::npos) {
            break;
        }
        start = star + 1;
    }
    return {out, inverted};
}

std::vector<std::string> stim_mpp_targets(const StimInstruction& inst) {
    std::vector<std::string> groups;
    std::string current;
    for (const auto& target : inst.targets) {
        if (target == "*") {
            if (current.empty() || ends_with(current, "*")) {
                fail("misplaced MPP combiner");
            }
            current += "*";
        } else if (current.empty()) {
            current = target;
        } else if (ends_with(current, "*")) {
            current += target;
        } else {
            groups.push_back(current);
            current = target;
        }
    }
    if (!current.empty()) {
        if (ends_with(current, "*")) {
            fail("dangling MPP combiner");
        }
        groups.push_back(current);
    }
    return groups;
}

std::vector<double> coords_with_shift(const std::vector<double>& coords, const std::vector<double>& shift) {
    std::vector<double> out(std::max(coords.size(), shift.size()), 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        if (i < coords.size()) {
            out[i] += coords[i];
        }
        if (i < shift.size()) {
            out[i] += shift[i];
        }
    }
    return out;
}

struct StimCircuitBuilder {
    QuantumCircuit circuit;
    std::vector<double> coord_shift;
    std::vector<CircuitPauliProduct> correlated_error_products;
    std::vector<double> correlated_error_probabilities;
    double correlated_error_remaining_probability = 1.0;
};

CircuitInstruction new_circuit_instruction(CircuitInstructionKind kind, int line) {
    CircuitInstruction instruction;
    instruction.kind = kind;
    instruction.line = line;
    return instruction;
}

void flush_correlated_error_group(StimCircuitBuilder& builder, int line) {
    if (builder.correlated_error_products.empty()) {
        builder.correlated_error_remaining_probability = 1.0;
        return;
    }
    CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::PauliProductChannel, line);
    instruction.pauli_products = std::move(builder.correlated_error_products);
    instruction.probabilities = std::move(builder.correlated_error_probabilities);
    builder.correlated_error_products.clear();
    builder.correlated_error_probabilities.clear();
    builder.correlated_error_remaining_probability = 1.0;
    builder.circuit.instructions.push_back(std::move(instruction));
}

std::vector<CircuitMeasurementTarget> circuit_measurement_targets(const StimInstruction& inst) {
    std::vector<CircuitMeasurementTarget> out;
    for (auto [q, inverted] : stim_measurement_targets(inst)) {
        out.push_back({q, inverted});
    }
    return out;
}

std::vector<CircuitPauliProduct> circuit_mpp_targets(int n, const StimInstruction& inst) {
    std::vector<CircuitPauliProduct> out;
    for (const auto& target : stim_mpp_targets(inst)) {
        auto [pauli, inverted] = stim_mpp_pauli(n, target);
        out.push_back({std::move(pauli), inverted});
    }
    return out;
}

CircuitPauliProduct circuit_implicit_pauli_product(int n, const StimInstruction& inst) {
    PauliString out = pauli_identity(n);
    bool inverted = false;
    for (const auto& target : inst.targets) {
        if (target == "*") {
            fail("implicit Pauli products do not use combiner targets");
        }
        auto [pauli, factor_inverted] = stim_mpp_pauli(n, target);
        out = out * pauli;
        inverted = inverted != factor_inverted;
    }
    return {std::move(out), inverted};
}

CircuitPauliProduct circuit_single_pauli_product(int n, char axis, int q, bool inverted = false) {
    return {pauli_on_target(n, axis, q), inverted};
}

CircuitPauliProduct circuit_pair_pauli_product(int n, char axis, int q1, int q2, bool inverted = false) {
    if (q1 == q2) {
        fail("two-qubit Pauli operation requires distinct qubits");
    }
    return {pauli_on_target(n, axis, q1) * pauli_on_target(n, axis, q2), inverted};
}

std::vector<CircuitPauliProduct> circuit_pair_measurement_products(int n, const StimInstruction& inst, char axis) {
    const auto targets = stim_measurement_targets(inst);
    if ((targets.size() & 1u) != 0u) {
        fail("pair measurement requires paired targets");
    }
    std::vector<CircuitPauliProduct> out;
    for (std::size_t i = 0; i < targets.size(); i += 2) {
        const bool inverted = targets[i].second != targets[i + 1].second;
        out.push_back(circuit_pair_pauli_product(n, axis, targets[i].first, targets[i + 1].first, inverted));
    }
    return out;
}

double pauli_rotation_kernel_angle_from_half_turns(double half_turns) {
    // Clifft parameters are physical half-turns, while the kernel applies exp(-i * angle * P).
    return half_turns * kPi / 2.0;
}

void append_pauli_rotation_instruction(
    StimCircuitBuilder& builder,
    int line,
    double kernel_angle,
    std::vector<CircuitPauliProduct> products) {
    CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::PauliRotation, line);
    instruction.kernel_angle = kernel_angle;
    instruction.pauli_products = std::move(products);
    builder.circuit.instructions.push_back(std::move(instruction));
}

void append_single_axis_rotation(StimCircuitBuilder& builder, const StimInstruction& inst, char axis) {
    require_paren_count(inst, 1, "single-qubit rotation expects one angle argument");
    std::vector<CircuitPauliProduct> products;
    for (int q : stim_qubit_targets(inst)) {
        products.push_back(circuit_single_pauli_product(builder.circuit.nqubits, axis, q));
    }
    append_pauli_rotation_instruction(
        builder,
        inst.line,
        pauli_rotation_kernel_angle_from_half_turns(inst.parens[0]),
        std::move(products));
}

void append_two_axis_rotation(StimCircuitBuilder& builder, const StimInstruction& inst, char axis) {
    require_paren_count(inst, 1, "two-qubit Pauli rotation expects one angle argument");
    const auto qubits = stim_qubit_targets(inst);
    if ((qubits.size() & 1u) != 0u) {
        fail("two-qubit Pauli rotation requires paired targets");
    }
    std::vector<CircuitPauliProduct> products;
    for (std::size_t i = 0; i < qubits.size(); i += 2) {
        products.push_back(circuit_pair_pauli_product(builder.circuit.nqubits, axis, qubits[i], qubits[i + 1]));
    }
    append_pauli_rotation_instruction(
        builder,
        inst.line,
        pauli_rotation_kernel_angle_from_half_turns(inst.parens[0]),
        std::move(products));
}

void append_u3_rotation(StimCircuitBuilder& builder, const StimInstruction& inst) {
    require_paren_count(inst, 3, "U3 expects three angle arguments");
    const auto qubits = stim_qubit_targets(inst);
    std::vector<CircuitPauliProduct> z_lambda;
    std::vector<CircuitPauliProduct> y_theta;
    std::vector<CircuitPauliProduct> z_phi;
    for (int q : qubits) {
        z_lambda.push_back(circuit_single_pauli_product(builder.circuit.nqubits, 'Z', q));
        y_theta.push_back(circuit_single_pauli_product(builder.circuit.nqubits, 'Y', q));
        z_phi.push_back(circuit_single_pauli_product(builder.circuit.nqubits, 'Z', q));
    }
    append_pauli_rotation_instruction(
        builder,
        inst.line,
        pauli_rotation_kernel_angle_from_half_turns(inst.parens[2]),
        std::move(z_lambda));
    append_pauli_rotation_instruction(
        builder,
        inst.line,
        pauli_rotation_kernel_angle_from_half_turns(inst.parens[0]),
        std::move(y_theta));
    append_pauli_rotation_instruction(
        builder,
        inst.line,
        pauli_rotation_kernel_angle_from_half_turns(inst.parens[1]),
        std::move(z_phi));
}

void append_feedback_pair(StimCircuitBuilder& builder, int line, CircuitInstructionKind kind, int record, int q) {
    CircuitInstruction instruction = new_circuit_instruction(kind, line);
    instruction.feedback_targets.push_back({record, q});
    builder.circuit.instructions.push_back(std::move(instruction));
}

int feedback_qubit_target(const std::string& target, int line) {
    auto [q, inverted] = stim_qubit_target(target, line);
    if (inverted) {
        fail("record-controlled feedback does not accept inverted qubit targets");
    }
    return q;
}

void append_qubit_instruction(StimCircuitBuilder& builder, const StimInstruction& inst, CircuitInstructionKind kind) {
    CircuitInstruction instruction = new_circuit_instruction(kind, inst.line);
    instruction.qubits = stim_qubit_targets(inst);
    builder.circuit.instructions.push_back(std::move(instruction));
}

void append_qubit_pair_instruction(
    StimCircuitBuilder& builder,
    int line,
    CircuitInstructionKind kind,
    int a,
    int b) {
    CircuitInstruction instruction = new_circuit_instruction(kind, line);
    instruction.qubits = {a, b};
    builder.circuit.instructions.push_back(std::move(instruction));
}

void append_measurement_instruction(StimCircuitBuilder& builder, const StimInstruction& inst, CircuitInstructionKind kind) {
    CircuitInstruction instruction = new_circuit_instruction(kind, inst.line);
    instruction.probability = stim_paren_probability(inst);
    instruction.measurement_targets = circuit_measurement_targets(inst);
    builder.circuit.nrecords += static_cast<int>(instruction.measurement_targets.size());
    builder.circuit.instructions.push_back(std::move(instruction));
}

std::optional<CircuitInstructionKind> single_qubit_clifford_kind(const std::string& op) {
    if (op == "H" || op == "H_XZ") {
        return CircuitInstructionKind::H;
    }
    if (op == "H_NXY") {
        return CircuitInstructionKind::H_NXY;
    }
    if (op == "H_NXZ") {
        return CircuitInstructionKind::H_NXZ;
    }
    if (op == "H_NYZ") {
        return CircuitInstructionKind::H_NYZ;
    }
    if (op == "H_XY") {
        return CircuitInstructionKind::H_XY;
    }
    if (op == "H_YZ") {
        return CircuitInstructionKind::H_YZ;
    }
    if (op == "C_NXYZ") {
        return CircuitInstructionKind::C_NXYZ;
    }
    if (op == "C_NZYX") {
        return CircuitInstructionKind::C_NZYX;
    }
    if (op == "C_XNYZ") {
        return CircuitInstructionKind::C_XNYZ;
    }
    if (op == "C_XYNZ") {
        return CircuitInstructionKind::C_XYNZ;
    }
    if (op == "C_XYZ") {
        return CircuitInstructionKind::C_XYZ;
    }
    if (op == "C_ZNYX") {
        return CircuitInstructionKind::C_ZNYX;
    }
    if (op == "C_ZYNX") {
        return CircuitInstructionKind::C_ZYNX;
    }
    if (op == "C_ZYX") {
        return CircuitInstructionKind::C_ZYX;
    }
    if (op == "S" || op == "SQRT_Z") {
        return CircuitInstructionKind::S;
    }
    if (op == "S_DAG" || op == "SQRT_Z_DAG") {
        return CircuitInstructionKind::SDG;
    }
    if (op == "SQRT_X") {
        return CircuitInstructionKind::SqrtX;
    }
    if (op == "SQRT_X_DAG") {
        return CircuitInstructionKind::SqrtXDag;
    }
    if (op == "SQRT_Y") {
        return CircuitInstructionKind::SqrtY;
    }
    if (op == "SQRT_Y_DAG") {
        return CircuitInstructionKind::SqrtYDag;
    }
    if (op == "X") {
        return CircuitInstructionKind::X;
    }
    if (op == "Y") {
        return CircuitInstructionKind::Y;
    }
    if (op == "Z") {
        return CircuitInstructionKind::Z;
    }
    return std::nullopt;
}

std::optional<CircuitInstructionKind> two_qubit_clifford_kind(const std::string& op) {
    if (op == "CX" || op == "CNOT" || op == "ZCX") {
        return CircuitInstructionKind::CX;
    }
    if (op == "CY" || op == "ZCY") {
        return CircuitInstructionKind::CY;
    }
    if (op == "CZ" || op == "ZCZ") {
        return CircuitInstructionKind::CZ;
    }
    if (op == "SWAP") {
        return CircuitInstructionKind::SWAP;
    }
    if (op == "CXSWAP") {
        return CircuitInstructionKind::CXSWAP;
    }
    if (op == "CZSWAP" || op == "SWAPCZ") {
        return CircuitInstructionKind::CZSWAP;
    }
    if (op == "ISWAP") {
        return CircuitInstructionKind::ISWAP;
    }
    if (op == "ISWAP_DAG") {
        return CircuitInstructionKind::ISWAP_DAG;
    }
    if (op == "SQRT_XX") {
        return CircuitInstructionKind::SQRT_XX;
    }
    if (op == "SQRT_XX_DAG") {
        return CircuitInstructionKind::SQRT_XX_DAG;
    }
    if (op == "SQRT_YY") {
        return CircuitInstructionKind::SQRT_YY;
    }
    if (op == "SQRT_YY_DAG") {
        return CircuitInstructionKind::SQRT_YY_DAG;
    }
    if (op == "SQRT_ZZ") {
        return CircuitInstructionKind::SQRT_ZZ;
    }
    if (op == "SQRT_ZZ_DAG") {
        return CircuitInstructionKind::SQRT_ZZ_DAG;
    }
    if (op == "SWAPCX") {
        return CircuitInstructionKind::SWAPCX;
    }
    if (op == "XCX") {
        return CircuitInstructionKind::XCX;
    }
    if (op == "XCY") {
        return CircuitInstructionKind::XCY;
    }
    if (op == "XCZ") {
        return CircuitInstructionKind::XCZ;
    }
    if (op == "YCX") {
        return CircuitInstructionKind::YCX;
    }
    if (op == "YCY") {
        return CircuitInstructionKind::YCY;
    }
    if (op == "YCZ") {
        return CircuitInstructionKind::YCZ;
    }
    return std::nullopt;
}

void append_correlated_error(StimCircuitBuilder& builder, const StimInstruction& inst, bool starts_group) {
    if (starts_group) {
        flush_correlated_error_group(builder, inst.line);
    } else if (builder.correlated_error_products.empty()) {
        fail("ELSE_CORRELATED_ERROR must follow CORRELATED_ERROR or ELSE_CORRELATED_ERROR");
    }
    const double p = stim_required_probability(inst);
    builder.correlated_error_products.push_back(circuit_implicit_pauli_product(builder.circuit.nqubits, inst));
    builder.correlated_error_probabilities.push_back(builder.correlated_error_remaining_probability * p);
    builder.correlated_error_remaining_probability *= (1.0 - p);
}

bool append_classical_controlled_pairs(StimCircuitBuilder& builder, const StimInstruction& inst) {
    const std::string& op = inst.op;
    bool has_classical_target = false;
    for (const auto& target : inst.targets) {
        if (is_sweep_target(target)) {
            fail("sweep-controlled operations are not supported");
        }
        has_classical_target = has_classical_target || is_record_target(target);
    }
    if (!has_classical_target) {
        return false;
    }
    if ((inst.targets.size() & 1u) != 0u) {
        fail("controlled two-qubit gate requires paired targets");
    }
    for (std::size_t i = 0; i < inst.targets.size(); i += 2) {
        const auto& a = inst.targets[i];
        const auto& b = inst.targets[i + 1];
        if (op == "CX" || op == "CNOT" || op == "ZCX" || op == "CY" || op == "ZCY") {
            const CircuitInstructionKind feedback =
                (op == "CY" || op == "ZCY") ? CircuitInstructionKind::FeedbackY : CircuitInstructionKind::FeedbackX;
            if (is_record_target(a)) {
                const int q = feedback_qubit_target(b, inst.line);
                append_feedback_pair(builder, inst.line, feedback, stim_record_index(a, builder.circuit.nrecords), q);
            } else {
                auto [qa, inv_a] = stim_qubit_target(a, inst.line);
                auto [qb, inv_b] = stim_qubit_target(b, inst.line);
                if (inv_a || inv_b) {
                    fail("two-qubit Clifford does not accept inverted qubit targets");
                }
                append_qubit_pair_instruction(
                    builder,
                    inst.line,
                    (op == "CY" || op == "ZCY") ? CircuitInstructionKind::CY : CircuitInstructionKind::CX,
                    qa,
                    qb);
            }
        } else if (op == "CZ" || op == "ZCZ") {
            if (is_record_target(a)) {
                const int q = feedback_qubit_target(b, inst.line);
                append_feedback_pair(builder, inst.line, CircuitInstructionKind::FeedbackZ, stim_record_index(a, builder.circuit.nrecords), q);
            } else if (is_record_target(b)) {
                const int q = feedback_qubit_target(a, inst.line);
                append_feedback_pair(builder, inst.line, CircuitInstructionKind::FeedbackZ, stim_record_index(b, builder.circuit.nrecords), q);
            } else {
                auto [qa, inv_a] = stim_qubit_target(a, inst.line);
                auto [qb, inv_b] = stim_qubit_target(b, inst.line);
                if (inv_a || inv_b) {
                    fail("two-qubit Clifford does not accept inverted qubit targets");
                }
                append_qubit_pair_instruction(builder, inst.line, CircuitInstructionKind::CZ, qa, qb);
            }
        } else if (op == "XCZ" || op == "YCZ") {
            const CircuitInstructionKind feedback =
                op == "YCZ" ? CircuitInstructionKind::FeedbackY : CircuitInstructionKind::FeedbackX;
            if (is_record_target(b)) {
                const int q = feedback_qubit_target(a, inst.line);
                append_feedback_pair(builder, inst.line, feedback, stim_record_index(b, builder.circuit.nrecords), q);
            } else {
                auto [qa, inv_a] = stim_qubit_target(a, inst.line);
                auto [qb, inv_b] = stim_qubit_target(b, inst.line);
                if (inv_a || inv_b) {
                    fail("two-qubit Clifford does not accept inverted qubit targets");
                }
                append_qubit_pair_instruction(
                    builder,
                    inst.line,
                    op == "YCZ" ? CircuitInstructionKind::YCZ : CircuitInstructionKind::XCZ,
                    qa,
                    qb);
            }
        } else {
            fail("unsupported classically controlled gate");
        }
    }
    return true;
}

void append_instruction(StimCircuitBuilder& builder, const StimInstruction& inst) {
    const std::string& op = inst.op;
    if (op == "E" || op == "CORRELATED_ERROR") {
        append_correlated_error(builder, inst, true);
        return;
    }
    if (op == "ELSE_CORRELATED_ERROR") {
        append_correlated_error(builder, inst, false);
        return;
    }
    flush_correlated_error_group(builder, inst.line);
    if (op == "QUBIT_COORDS") {
        require_coordinate_count(inst, false, "QUBIT_COORDS expects 1 to 16 coordinate arguments when parens are present");
        if (inst.targets.empty()) {
            fail("QUBIT_COORDS expects qubit targets");
        }
        (void)stim_qubit_targets(inst);
        return;
    }
    if (op == "TICK") {
        require_no_parens(inst);
        require_no_targets(inst);
        builder.circuit.instructions.push_back(new_circuit_instruction(CircuitInstructionKind::Tick, inst.line));
        return;
    }
    if (op == "SHIFT_COORDS") {
        require_coordinate_count(inst, true, "SHIFT_COORDS expects 1 to 16 coordinate arguments");
        require_no_targets(inst);
        if (builder.coord_shift.size() < inst.parens.size()) {
            builder.coord_shift.resize(inst.parens.size(), 0.0);
        }
        for (std::size_t i = 0; i < inst.parens.size(); ++i) {
            builder.coord_shift[i] += inst.parens[i];
        }
        return;
    }
    if (op == "DETECTOR") {
        require_coordinate_count(inst, false, "DETECTOR expects 1 to 16 coordinate arguments when parens are present");
        CircuitDetector detector;
        detector.records = stim_record_indices(inst, builder.circuit.nrecords);
        detector.coords = coords_with_shift(inst.parens, builder.coord_shift);
        detector.line = inst.line;
        detector.after_instruction = static_cast<int>(builder.circuit.instructions.size());
        builder.circuit.detectors.push_back(std::move(detector));
        return;
    }
    if (op == "OBSERVABLE_INCLUDE") {
        if (inst.parens.size() != 1) {
            fail("OBSERVABLE_INCLUDE expects one nonnegative observable index");
        }
        const int observable_index =
            nonnegative_integer_argument(inst.parens[0], "OBSERVABLE_INCLUDE expects one nonnegative integer index");
        builder.circuit.observables.push_back(
            {observable_index,
             stim_observable_record_indices(inst, builder.circuit.nrecords),
             inst.line});
        return;
    }
    if (append_classical_controlled_pairs(builder, inst)) {
        return;
    }
    if (op == "I") {
        require_no_parens(inst);
        (void)stim_qubit_targets(inst);
    } else if (op == "II") {
        require_no_parens(inst);
        const auto qubits = stim_qubit_targets(inst);
        if ((qubits.size() & 1u) != 0u) {
            fail("II requires paired targets");
        }
    } else if (const auto kind = single_qubit_clifford_kind(op)) {
        require_no_parens(inst);
        append_qubit_instruction(builder, inst, *kind);
    } else if (const auto kind = two_qubit_clifford_kind(op)) {
        require_no_parens(inst);
        append_qubit_instruction(builder, inst, *kind);
    } else if (op == "T") {
        require_no_parens(inst);
        append_qubit_instruction(builder, inst, CircuitInstructionKind::T);
    } else if (op == "T_DAG") {
        require_no_parens(inst);
        append_qubit_instruction(builder, inst, CircuitInstructionKind::TDag);
    } else if (op == "R_X") {
        append_single_axis_rotation(builder, inst, 'X');
    } else if (op == "R_Y") {
        append_single_axis_rotation(builder, inst, 'Y');
    } else if (op == "R_Z") {
        append_single_axis_rotation(builder, inst, 'Z');
    } else if (op == "U3" || op == "U") {
        append_u3_rotation(builder, inst);
    } else if (op == "R_XX") {
        append_two_axis_rotation(builder, inst, 'X');
    } else if (op == "R_YY") {
        append_two_axis_rotation(builder, inst, 'Y');
    } else if (op == "R_ZZ") {
        append_two_axis_rotation(builder, inst, 'Z');
    } else if (op == "R_PAULI") {
        require_paren_count(inst, 1, "R_PAULI expects one angle argument");
        append_pauli_rotation_instruction(
            builder,
            inst.line,
            pauli_rotation_kernel_angle_from_half_turns(inst.parens[0]),
            circuit_mpp_targets(builder.circuit.nqubits, inst));
    } else if (op == "M" || op == "MZ" || op == "MX" || op == "MY") {
        append_measurement_instruction(
            builder,
            inst,
            op == "MX" ? CircuitInstructionKind::MX : (op == "MY" ? CircuitInstructionKind::MY : CircuitInstructionKind::MZ));
    } else if (op == "MR" || op == "MRZ" || op == "MRX" || op == "MRY") {
        append_measurement_instruction(
            builder,
            inst,
            op == "MRX" ? CircuitInstructionKind::MRX : (op == "MRY" ? CircuitInstructionKind::MRY : CircuitInstructionKind::MRZ));
    } else if (op == "R" || op == "RZ" || op == "RX" || op == "RY") {
        require_no_parens(inst);
        append_qubit_instruction(
            builder,
            inst,
            op == "RX" ? CircuitInstructionKind::RX : (op == "RY" ? CircuitInstructionKind::RY : CircuitInstructionKind::RZ));
    } else if (op == "MPP") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::MPP, inst.line);
        instruction.probability = stim_paren_probability(inst);
        instruction.pauli_products = circuit_mpp_targets(builder.circuit.nqubits, inst);
        builder.circuit.nrecords += static_cast<int>(instruction.pauli_products.size());
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "MXX" || op == "MYY" || op == "MZZ") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::MPP, inst.line);
        instruction.probability = stim_paren_probability(inst);
        const char axis = op == "MXX" ? 'X' : (op == "MYY" ? 'Y' : 'Z');
        instruction.pauli_products = circuit_pair_measurement_products(builder.circuit.nqubits, inst, axis);
        builder.circuit.nrecords += static_cast<int>(instruction.pauli_products.size());
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "SPP" || op == "SPP_DAG") {
        require_no_parens(inst);
        append_pauli_rotation_instruction(
            builder,
            inst.line,
            op == "SPP" ? kPi / 4.0 : -kPi / 4.0,
            circuit_mpp_targets(builder.circuit.nqubits, inst));
    } else if (op == "X_ERROR" || op == "Y_ERROR" || op == "Z_ERROR") {
        CircuitInstruction instruction = new_circuit_instruction(
            op == "X_ERROR" ? CircuitInstructionKind::XError
                             : (op == "Y_ERROR" ? CircuitInstructionKind::YError : CircuitInstructionKind::ZError),
            inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "I_ERROR") {
        check_disjoint_probability_list(inst, "I_ERROR probabilities must be disjoint and sum to at most 1");
        (void)stim_qubit_targets(inst);
    } else if (op == "II_ERROR") {
        check_disjoint_probability_list(inst, "II_ERROR probabilities must be disjoint and sum to at most 1");
        const auto qubits = stim_qubit_targets(inst);
        if ((qubits.size() & 1u) != 0u) {
            fail("II_ERROR requires paired targets");
        }
    } else if (op == "DEPOLARIZE1") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::Depolarize1, inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "DEPOLARIZE2") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::Depolarize2, inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "DEPOLARIZE3") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::Depolarize3, inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "PAULI_CHANNEL_1" || op == "PAULI_CHANNEL_2" || op == "PAULI_CHANNEL_3") {
        const std::size_t expected =
            op == "PAULI_CHANNEL_1" ? 3u : (op == "PAULI_CHANNEL_2" ? 15u : 63u);
        if (inst.parens.size() != expected) {
            fail("PAULI_CHANNEL probability count does not match gate arity");
        }
        CircuitInstruction instruction = new_circuit_instruction(
            op == "PAULI_CHANNEL_1"
                ? CircuitInstructionKind::PauliChannel1
                : (op == "PAULI_CHANNEL_2" ? CircuitInstructionKind::PauliChannel2
                                            : CircuitInstructionKind::PauliChannel3),
            inst.line);
        instruction.probabilities = inst.parens;
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "HERALDED_ERASE") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::HeraldedErase, inst.line);
        instruction.probability = stim_required_probability(inst);
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.nrecords += static_cast<int>(instruction.qubits.size());
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "HERALDED_PAULI_CHANNEL_1") {
        require_paren_count(inst, 4, "HERALDED_PAULI_CHANNEL_1 expects four probabilities");
        CircuitInstruction instruction =
            new_circuit_instruction(CircuitInstructionKind::HeraldedPauliChannel1, inst.line);
        instruction.probabilities = inst.parens;
        instruction.qubits = stim_qubit_targets(inst);
        builder.circuit.nrecords += static_cast<int>(instruction.qubits.size());
        builder.circuit.instructions.push_back(std::move(instruction));
    } else if (op == "MPAD") {
        CircuitInstruction instruction = new_circuit_instruction(CircuitInstructionKind::MPad, inst.line);
        instruction.probability = stim_paren_probability(inst);
        instruction.measurement_targets = circuit_measurement_targets(inst);
        builder.circuit.nrecords += static_cast<int>(instruction.measurement_targets.size());
        builder.circuit.instructions.push_back(std::move(instruction));
    } else {
        fail("unsupported Stim operation: " + op);
    }
}

void append_nodes(StimCircuitBuilder& builder, const std::vector<StimNode>& nodes) {
    for (const auto& node : nodes) {
        if (const auto* block = std::get_if<std::shared_ptr<StimRepeatBlock>>(&node)) {
            flush_correlated_error_group(builder, (*block)->line);
            for (int i = 0; i < (*block)->count; ++i) {
                append_nodes(builder, (*block)->body);
            }
        } else {
            append_instruction(builder, std::get<StimInstruction>(node));
        }
    }
    const int line = nodes.empty() ? 0 : std::visit(
        [](const auto& node) {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, StimInstruction>) {
                return node.line;
            } else {
                return node->line;
            }
        },
        nodes.back());
    flush_correlated_error_group(builder, line);
}

std::vector<StimDetector> detectors_with_lowered_positions(
    const QuantumCircuit& circuit,
    const CircuitLoweringResult& lowered) {
    std::vector<StimDetector> detectors = circuit.detectors;
    for (auto& detector : detectors) {
        if (detector.after_instruction < 0 ||
            detector.after_instruction >= static_cast<int>(lowered.instruction_pending_operation_counts.size())) {
            fail("detector source position is out of range");
        }
        detector.after_pending_operation =
            lowered.instruction_pending_operation_counts[static_cast<std::size_t>(detector.after_instruction)];
    }
    return detectors;
}

} // namespace

QuantumCircuit parse_stim_circuit_lines(const std::vector<std::string>& lines) {
    std::size_t pos = 0;
    const auto nodes = parse_nodes(lines, pos, false);
    StimCircuitBuilder builder;
    builder.circuit.nqubits = all_qubits(nodes);
    append_nodes(builder, nodes);
    return builder.circuit;
}

QuantumCircuit parse_stim_circuit_text(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }
    return parse_stim_circuit_lines(lines);
}

QuantumCircuit parse_stim_circuit_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        fail("failed to open Stim file: " + path);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        lines.push_back(line);
    }
    return parse_stim_circuit_lines(lines);
}

StimParseResult parse_stim_lines(const std::vector<std::string>& lines) {
    const QuantumCircuit circuit = parse_stim_circuit_lines(lines);
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    auto detectors = detectors_with_lowered_positions(circuit, lowered);
    return StimParseResult{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        std::move(detectors),
        circuit.observables};
}

StimParseResult parse_stim_text(const std::string& text) {
    const QuantumCircuit circuit = parse_stim_circuit_text(text);
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    auto detectors = detectors_with_lowered_positions(circuit, lowered);
    return StimParseResult{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        std::move(detectors),
        circuit.observables};
}

StimParseResult parse_stim_file(const std::string& path) {
    const QuantumCircuit circuit = parse_stim_circuit_file(path);
    CircuitLoweringResult lowered = lower_circuit_to_factored(circuit);
    auto detectors = detectors_with_lowered_positions(circuit, lowered);
    return StimParseResult{
        std::move(lowered.state),
        std::move(lowered.measurement_records),
        std::move(detectors),
        circuit.observables};
}

} // namespace symft
