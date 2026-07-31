#include "stabilizers.h"
#include <cassert>
#include <random>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <stdexcept>

const double PI = 3.14159265358979323846;


namespace {

std::string trim(const std::string& input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        begin++;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        end--;
    }
    return input.substr(begin, end - begin);
}

std::string strip_comment(const std::string& input) {
    auto pos = input.find('#');
    if (pos == std::string::npos) {
        return trim(input);
    }
    return trim(input.substr(0, pos));
}

std::string to_upper(std::string input) {
    std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return input;
}
int parse_int_checked(const std::string& input, const std::string& context) {
    try {
        size_t used = 0;
        int value = std::stoi(input, &used);
        if (used != input.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer target '" + input + "' in " + context);
    }
}

double parse_double_checked(const std::string& input, const std::string& context) {
    try {
        size_t used = 0;
        double value = std::stod(input, &used);
        if (used != input.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid numeric argument '" + input + "' in " + context);
    }
}

double validate_probability(double probability, const std::string& context) {
    if (!std::isfinite(probability) || probability < 0.0 || probability > 1.0) {
        throw std::runtime_error("Probability must be between 0 and 1 in " + context);
    }
    return probability;
}

std::vector<double> parse_optional_probability_args(const std::vector<double>& args, const std::string& context) {
    if (args.empty()) {
        return std::vector<double>();
    }
    if (args.size() != 1) {
        throw std::runtime_error("Expected zero or one probability argument in " + context);
    }
    return std::vector<double>{validate_probability(args[0], context)};
}

std::vector<double> parse_required_probability_args_or_token(const std::vector<double>& args,
                                                            std::vector<std::string>& tokens,
                                                            const std::string& context) {
    if (args.size() == 1) {
        return std::vector<double>{validate_probability(args[0], context)};
    }
    if (!args.empty()) {
        throw std::runtime_error("Expected one probability argument in " + context);
    }
    if (tokens.empty()) {
        throw std::runtime_error("Expected one probability argument in " + context);
    }
    double probability = validate_probability(parse_double_checked(tokens.front(), context), context);
    tokens.erase(tokens.begin());
    return std::vector<double>{probability};
}

std::vector<std::string> split_ws(const std::string& input) {
    std::vector<std::string> tokens;
    std::istringstream iss(input);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> strip_target_inversions(const std::vector<std::string>& tokens) {
    std::vector<std::string> stripped;
    stripped.reserve(tokens.size());
    for (auto token : tokens) {
        while (!token.empty() && token[0] == '!') {
            token = token.substr(1);
        }
        stripped.push_back(token);
    }
    return stripped;
}

std::string read_instruction_head(const std::string& line, std::string& rest) {
    int depth = 0;
    size_t pos = 0;
    for (; pos < line.size(); pos++) {
        char ch = line[pos];
        if (ch == '(') {
            depth++;
        } else if (ch == ')') {
            depth--;
            if (depth < 0) {
                throw std::runtime_error("Unbalanced ')' in stim line: " + line);
            }
        } else if (std::isspace(static_cast<unsigned char>(ch)) && depth == 0) {
            break;
        }
    }
    if (depth != 0) {
        throw std::runtime_error("Unbalanced '(' in stim line: " + line);
    }
    rest = trim(line.substr(pos));
    return trim(line.substr(0, pos));
}

void parse_instruction_head(const std::string& head, std::string& name, std::vector<double>& args, const std::string& context) {
    args.clear();
    auto open = head.find('(');
    if (open == std::string::npos) {
        name = to_upper(head);
        return;
    }
    auto close = head.rfind(')');
    if (close == std::string::npos || close < open) {
        throw std::runtime_error("Invalid instruction arguments in " + context);
    }
    if (close + 1 != head.size()) {
        throw std::runtime_error("Unexpected characters after instruction arguments in " + context);
    }
    name = to_upper(trim(head.substr(0, open)));
    std::string body = trim(head.substr(open + 1, close - open - 1));
    if (body.empty()) {
        return;
    }
    size_t begin = 0;
    while (begin <= body.size()) {
        size_t comma = body.find(',', begin);
        std::string token = trim(body.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin));
        if (!token.empty()) {
            args.push_back(parse_double_checked(token, context));
        }
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
}

int parse_target_token(const std::string& token, bool& invert, const std::string& context) {
    std::string t = token;
    while (!t.empty() && t[0] == '!') {
        invert = !invert;
        t = t.substr(1);
    }
    std::string upper = to_upper(t);
    if (upper.size() >= 5 && upper.substr(0, 4) == "REC[" && upper.back() == ']') {
        return parse_int_checked(t.substr(4, t.size() - 5), context);
    }
    if (upper.size() >= 7 && upper.substr(0, 6) == "SWEEP[") {
        throw std::runtime_error("sweep targets are not supported by this CPU simulator in " + context);
    }
    return parse_int_checked(t, context);
}

void update_max_qubit(int target, int& max_qubit) {
    if (target >= 0 && target > max_qubit) {
        max_qubit = target;
    }
}

void append_gate(std::vector<QuantumGate>& circuit, int& max_qubit, const std::string& gate,
                 const std::vector<int>& targets, const std::vector<double>& args = std::vector<double>(), bool invert = false) {
    QuantumGate op;
    op.gate = gate;
    op.targets = targets;
    op.args = args;
    op.invert = invert;
    for (auto target : targets) {
        update_max_qubit(target, max_qubit);
    }
    circuit.push_back(op);
}

std::vector<int> parse_qubit_targets(const std::vector<std::string>& tokens, const std::string& context) {
    std::vector<int> targets;
    for (auto& token : tokens) {
        bool invert = false;
        int target = parse_target_token(token, invert, context);
        if (invert) {
            throw std::runtime_error("Inverted targets are only supported for measurement records in " + context);
        }
        if (target < 0) {
            throw std::runtime_error("Expected a qubit target, got measurement record target in " + context);
        }
        targets.push_back(target);
    }
    return targets;
}

void append_single_qubit_ops(std::vector<QuantumGate>& circuit, int& max_qubit, const std::string& gate,
                             const std::vector<std::string>& tokens, const std::string& context) {
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty()) {
        throw std::runtime_error("Instruction '" + gate + "' has no targets in " + context);
    }
    for (auto target : targets) {
        append_gate(circuit, max_qubit, gate, std::vector<int>{target});
    }
}

void append_measure_ops(std::vector<QuantumGate>& circuit, int& max_qubit,
                        const std::vector<std::string>& tokens, const std::vector<double>& args,
                        const std::string& context) {
    auto probability_args = parse_optional_probability_args(args, context);
    if (tokens.empty()) {
        throw std::runtime_error("Measurement instruction has no targets in " + context);
    }
    for (auto& token : tokens) {
        bool invert = false;
        int target = parse_target_token(token, invert, context);
        if (target < 0) {
            throw std::runtime_error("Expected a qubit target for measurement in " + context);
        }
        append_gate(circuit, max_qubit, "M", std::vector<int>{target}, probability_args, invert);
    }
}

void append_mx_ops(std::vector<QuantumGate>& circuit, int& max_qubit,
                   const std::vector<std::string>& tokens, const std::vector<double>& args,
                   const std::string& context) {
    auto probability_args = parse_optional_probability_args(args, context);
    if (tokens.empty()) {
        throw std::runtime_error("MX instruction has no targets in " + context);
    }
    for (auto& token : tokens) {
        bool invert = false;
        int target = parse_target_token(token, invert, context);
        if (target < 0) {
            throw std::runtime_error("Expected a qubit target for MX in " + context);
        }
        append_gate(circuit, max_qubit, "H", std::vector<int>{target});
        append_gate(circuit, max_qubit, "M", std::vector<int>{target}, probability_args, invert);
        append_gate(circuit, max_qubit, "H", std::vector<int>{target});
    }
}

void append_my_ops(std::vector<QuantumGate>& circuit, int& max_qubit,
                   const std::vector<std::string>& tokens, const std::vector<double>& args,
                   const std::string& context) {
    auto probability_args = parse_optional_probability_args(args, context);
    if (tokens.empty()) {
        throw std::runtime_error("MY instruction has no targets in " + context);
    }
    for (auto& token : tokens) {
        bool invert = false;
        int target = parse_target_token(token, invert, context);
        if (target < 0) {
            throw std::runtime_error("Expected a qubit target for MY in " + context);
        }
        append_gate(circuit, max_qubit, "SDG", std::vector<int>{target});
        append_gate(circuit, max_qubit, "H", std::vector<int>{target});
        append_gate(circuit, max_qubit, "M", std::vector<int>{target}, probability_args, invert);
        append_gate(circuit, max_qubit, "H", std::vector<int>{target});
        append_gate(circuit, max_qubit, "S", std::vector<int>{target});
    }
}

void append_mpp_ops(std::vector<QuantumGate>& circuit, int& max_qubit,
                    const std::vector<std::string>& tokens, const std::string& context) {
    if (tokens.empty()) {
        throw std::runtime_error("MPP instruction has no targets in " + context);
    }
    for (auto& token : tokens) {
        bool invert = false;
        std::vector<int> targets;
        std::vector<double> paulis;
        size_t begin = 0;
        while (begin <= token.size()) {
            size_t star = token.find("*", begin);
            std::string factor = trim(token.substr(begin, star == std::string::npos ? std::string::npos : star - begin));
            if (factor.empty()) {
                throw std::runtime_error("Empty MPP factor in " + context);
            }
            while (!factor.empty() && factor[0] == 33) {
                invert = !invert;
                factor = factor.substr(1);
            }
            if (factor.size() < 2) {
                throw std::runtime_error("Invalid MPP factor " + factor + " in " + context);
            }
            std::string pauli = to_upper(factor.substr(0, 1));
            if (pauli != "X" && pauli != "Y" && pauli != "Z") {
                throw std::runtime_error("MPP factor must start with X, Y, or Z in " + context);
            }
            bool target_invert = false;
            int target = parse_target_token(factor.substr(1), target_invert, context);
            if (target_invert) {
                invert = !invert;
            }
            if (target < 0) {
                throw std::runtime_error("Expected a qubit target for MPP in " + context);
            }
            targets.push_back(target);
            paulis.push_back(pauli == "X" ? 1.0 : (pauli == "Y" ? 2.0 : 3.0));
            if (star == std::string::npos) {
                break;
            }
            begin = star + 1;
        }
        append_gate(circuit, max_qubit, "MPP", targets, paulis, invert);
    }
}

void append_reset_ops(std::vector<QuantumGate>& circuit, int& max_qubit, const std::string& basis,
                      const std::vector<std::string>& tokens, const std::string& context) {
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty()) {
        throw std::runtime_error("Reset instruction has no targets in " + context);
    }
    for (auto target : targets) {
        append_gate(circuit, max_qubit, "R", std::vector<int>{target});
        if (basis == "X") {
            append_gate(circuit, max_qubit, "H", std::vector<int>{target});
        } else if (basis == "Y") {
            append_gate(circuit, max_qubit, "H", std::vector<int>{target});
            append_gate(circuit, max_qubit, "S", std::vector<int>{target});
        }
    }
}

void append_controlled_pauli_pairs(std::vector<QuantumGate>& circuit, int& max_qubit, const std::string& gate,
                                   const std::vector<std::string>& tokens, const std::string& context) {
    if (tokens.empty() || tokens.size() % 2 != 0) {
        throw std::runtime_error(gate + " requires an even number of targets in " + context);
    }
    for (size_t i = 0; i < tokens.size(); i += 2) {
        bool control_invert = false;
        bool target_invert = false;
        int control = parse_target_token(tokens[i], control_invert, context);
        int target = parse_target_token(tokens[i + 1], target_invert, context);

        if (control < 0 && target >= 0) {
            if (target_invert) {
                throw std::runtime_error("Inverted qubit targets are not supported for feedback in " + context);
            }
            std::string feedback_gate = gate == "CX" ? "FEEDBACK_X" : (gate == "CY" ? "FEEDBACK_Y" : "FEEDBACK_Z");
            append_gate(circuit, max_qubit, feedback_gate, std::vector<int>{control, target}, std::vector<double>(), control_invert);
        } else if (control >= 0 && target >= 0) {
            if (control_invert || target_invert) {
                throw std::runtime_error("Inverted qubit targets are not supported for " + gate + " in " + context);
            }
            if (gate == "CX") {
                append_gate(circuit, max_qubit, "CX", std::vector<int>{control, target});
            } else if (gate == "CY") {
                append_gate(circuit, max_qubit, "SDG", std::vector<int>{target});
                append_gate(circuit, max_qubit, "CX", std::vector<int>{control, target});
                append_gate(circuit, max_qubit, "S", std::vector<int>{target});
            } else {
                append_gate(circuit, max_qubit, "H", std::vector<int>{target});
                append_gate(circuit, max_qubit, "CX", std::vector<int>{control, target});
                append_gate(circuit, max_qubit, "H", std::vector<int>{target});
            }
        } else if (control >= 0 && target < 0) {
            throw std::runtime_error("Measurement record targets are only supported as controls for " + gate + " in " + context);
        } else {
            throw std::runtime_error(gate + " cannot use two measurement record targets in " + context);
        }
    }
}

void append_two_qubit_pairs(std::vector<QuantumGate>& circuit, int& max_qubit, const std::string& gate,
                            const std::vector<std::string>& tokens, const std::string& context) {
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty() || targets.size() % 2 != 0) {
        throw std::runtime_error("Instruction '" + gate + "' requires an even number of targets in " + context);
    }
    for (size_t i = 0; i < targets.size(); i += 2) {
        append_gate(circuit, max_qubit, gate, std::vector<int>{targets[i], targets[i + 1]});
    }
}

void append_cz_pairs(std::vector<QuantumGate>& circuit, int& max_qubit,
                     const std::vector<std::string>& tokens, const std::string& context) {
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty() || targets.size() % 2 != 0) {
        throw std::runtime_error("CZ requires an even number of targets in " + context);
    }
    for (size_t i = 0; i < targets.size(); i += 2) {
        int c = targets[i];
        int t = targets[i + 1];
        append_gate(circuit, max_qubit, "H", std::vector<int>{t});
        append_gate(circuit, max_qubit, "CX", std::vector<int>{c, t});
        append_gate(circuit, max_qubit, "H", std::vector<int>{t});
    }
}

void append_swap_pairs(std::vector<QuantumGate>& circuit, int& max_qubit,
                       const std::vector<std::string>& tokens, const std::string& context) {
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty() || targets.size() % 2 != 0) {
        throw std::runtime_error("SWAP requires an even number of targets in " + context);
    }
    for (size_t i = 0; i < targets.size(); i += 2) {
        int a = targets[i];
        int b = targets[i + 1];
        append_gate(circuit, max_qubit, "CX", std::vector<int>{a, b});
        append_gate(circuit, max_qubit, "CX", std::vector<int>{b, a});
        append_gate(circuit, max_qubit, "CX", std::vector<int>{a, b});
    }
}

void append_single_qubit_noise_ops(std::vector<QuantumGate>& circuit, int& max_qubit, const std::string& gate,
                                   std::vector<std::string> tokens, const std::vector<double>& args,
                                   const std::string& context) {
    auto probability_args = parse_required_probability_args_or_token(args, tokens, context);
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty()) {
        throw std::runtime_error("Noise instruction '" + gate + "' has no targets in " + context);
    }
    for (auto target : targets) {
        append_gate(circuit, max_qubit, gate, std::vector<int>{target}, probability_args);
    }
}

void append_depolarize2_ops(std::vector<QuantumGate>& circuit, int& max_qubit,
                            std::vector<std::string> tokens, const std::vector<double>& args,
                            const std::string& context) {
    auto probability_args = parse_required_probability_args_or_token(args, tokens, context);
    auto targets = parse_qubit_targets(tokens, context);
    if (targets.empty() || targets.size() % 2 != 0) {
        throw std::runtime_error("DEPOLARIZE2 requires an even number of targets in " + context);
    }
    for (size_t i = 0; i < targets.size(); i += 2) {
        append_gate(circuit, max_qubit, "DEPOLARIZE2", std::vector<int>{targets[i], targets[i + 1]}, probability_args);
    }
}

std::vector<int> parse_record_targets(const std::vector<std::string>& tokens, bool& invert, const std::string& context) {
    std::vector<int> targets;
    for (auto& token : tokens) {
        bool target_invert = false;
        int target = parse_target_token(token, target_invert, context);
        invert = invert ^ target_invert;
        targets.push_back(target);
    }
    return targets;
}

void parse_stim_instruction(const std::string& line, int line_no, std::vector<QuantumGate>& circuit, int& max_qubit) {
    std::string context = "line " + std::to_string(line_no) + ": " + line;
    std::string rest;
    std::string head = read_instruction_head(line, rest);
    if (head.empty()) {
        return;
    }

    std::string name;
    std::vector<double> args;
    parse_instruction_head(head, name, args, context);
    auto tokens = split_ws(rest);

    if (name == "I" || name == "TICK" || name == "SHIFT_COORDS") {
        return;
    }
    if (name == "QUBIT_COORDS") {
        auto targets = parse_qubit_targets(tokens, context);
        for (auto target : targets) {
            update_max_qubit(target, max_qubit);
        }
        return;
    }
    if (name == "CNOT") {
        name = "CX";
    } else if (name == "MZ") {
        name = "M";
    } else if (name == "RZ") {
        name = "R";
    } else if (name == "S_DAG" || name == "S_DAGGER") {
        name = "SDG";
    } else if (name == "T_DAG" || name == "T_DAGGER") {
        name = "TDG";
    } else if (name == "XERR") {
        name = "X_ERROR";
    } else if (name == "YERR") {
        name = "Y_ERROR";
    } else if (name == "ZERR") {
        name = "Z_ERROR";
    } else if (name == "DEP1") {
        name = "DEPOLARIZE1";
    } else if (name == "DEP2") {
        name = "DEPOLARIZE2";
    }

    if (name == "X" || name == "Y" || name == "Z" || name == "H" || name == "S" ||
        name == "SDG" || name == "T" || name == "TDG") {
        append_single_qubit_ops(circuit, max_qubit, name, tokens, context);
    } else if (name == "M") {
        append_measure_ops(circuit, max_qubit, tokens, args, context);
    } else if (name == "MX") {
        append_mx_ops(circuit, max_qubit, tokens, args, context);
    } else if (name == "MY") {
        append_my_ops(circuit, max_qubit, tokens, args, context);
    } else if (name == "MPP") {
        append_mpp_ops(circuit, max_qubit, tokens, context);
    } else if (name == "R") {
        append_reset_ops(circuit, max_qubit, "Z", tokens, context);
    } else if (name == "RX") {
        append_reset_ops(circuit, max_qubit, "X", tokens, context);
    } else if (name == "RY") {
        append_reset_ops(circuit, max_qubit, "Y", tokens, context);
    } else if (name == "MR") {
        append_measure_ops(circuit, max_qubit, tokens, args, context);
        append_reset_ops(circuit, max_qubit, "Z", strip_target_inversions(tokens), context);
    } else if (name == "MRX") {
        append_mx_ops(circuit, max_qubit, tokens, args, context);
        append_reset_ops(circuit, max_qubit, "X", strip_target_inversions(tokens), context);
    } else if (name == "MRY") {
        append_my_ops(circuit, max_qubit, tokens, args, context);
        append_reset_ops(circuit, max_qubit, "Y", strip_target_inversions(tokens), context);
    } else if (name == "CX" || name == "CY" || name == "CZ") {
        append_controlled_pauli_pairs(circuit, max_qubit, name, tokens, context);
    } else if (name == "SWAP") {
        append_swap_pairs(circuit, max_qubit, tokens, context);
    } else if (name == "X_ERROR" || name == "Y_ERROR" || name == "Z_ERROR" || name == "DEPOLARIZE1") {
        append_single_qubit_noise_ops(circuit, max_qubit, name, tokens, args, context);
    } else if (name == "DEPOLARIZE2") {
        append_depolarize2_ops(circuit, max_qubit, tokens, args, context);
    } else if (name == "DETECTOR") {
        bool invert = false;
        auto targets = parse_record_targets(tokens, invert, context);
        QuantumGate op;
        op.gate = "DETECTOR";
        op.targets = targets;
        op.args = args;
        op.invert = invert;
        circuit.push_back(op);
    } else if (name == "OBSERVABLE_INCLUDE") {
        bool invert = false;
        auto targets = parse_record_targets(tokens, invert, context);
        QuantumGate op;
        op.gate = "OBSERVABLE_INCLUDE";
        op.targets = targets;
        op.args = args;
        op.invert = invert;
        circuit.push_back(op);
    } else {
        throw std::runtime_error("Unsupported stim instruction '" + name + "' in " + context);
    }
}

void parse_stim_block(const std::vector<std::string>& lines, size_t& pos, std::vector<QuantumGate>& circuit,
                      int& max_qubit, bool in_block) {
    while (pos < lines.size()) {
        int line_no = static_cast<int>(pos + 1);
        std::string line = strip_comment(lines[pos]);
        pos++;
        if (line.empty()) {
            continue;
        }
        if (line == "}") {
            if (!in_block) {
                throw std::runtime_error("Unexpected '}' at line " + std::to_string(line_no));
            }
            return;
        }

        std::string upper = to_upper(line);
        if (upper.size() >= 6 && upper.substr(0, 6) == "REPEAT" && (upper.size() == 6 || std::isspace(static_cast<unsigned char>(upper[6])))) {
            std::istringstream iss(line.substr(6));
            long long repeat_count = 0;
            if (!(iss >> repeat_count) || repeat_count < 0) {
                throw std::runtime_error("Invalid REPEAT count at line " + std::to_string(line_no));
            }
            std::string tail;
            std::getline(iss, tail);
            tail = trim(tail);
            if (tail.empty()) {
                bool found_brace = false;
                while (pos < lines.size()) {
                    std::string brace_line = strip_comment(lines[pos]);
                    pos++;
                    if (brace_line.empty()) {
                        continue;
                    }
                    if (brace_line == "{") {
                        found_brace = true;
                        break;
                    }
                    throw std::runtime_error("Expected '{' after REPEAT at line " + std::to_string(line_no));
                }
                if (!found_brace) {
                    throw std::runtime_error("Missing '{' after REPEAT at line " + std::to_string(line_no));
                }
            } else if (tail != "{") {
                throw std::runtime_error("Expected '{' after REPEAT count at line " + std::to_string(line_no));
            }

            std::vector<QuantumGate> body;
            parse_stim_block(lines, pos, body, max_qubit, true);
            for (long long i = 0; i < repeat_count; i++) {
                circuit.insert(circuit.end(), body.begin(), body.end());
            }
            continue;
        }

        if (line == "{") {
            throw std::runtime_error("Unexpected '{' at line " + std::to_string(line_no));
        }
        parse_stim_instruction(line, line_no, circuit, max_qubit);
    }

    if (in_block) {
        throw std::runtime_error("Missing closing '}' for REPEAT block");
    }
}

bool measurement_ref_value(const std::vector<MeasureResults>& measurements, int ref, const std::string& context) {
    int idx = ref < 0 ? static_cast<int>(measurements.size()) + ref : ref;
    if (idx < 0 || idx >= static_cast<int>(measurements.size())) {
        throw std::runtime_error("Measurement record reference " + std::to_string(ref) + " is out of range in " + context);
    }
    return measurements[idx].reg == 1;
}

std::string bit_string(const std::vector<MeasureResults>& measurements) {
    std::string out;
    out.reserve(measurements.size());
    for (auto& m : measurements) {
        out.push_back(m.reg ? '1' : '0');
    }
    return out;
}

std::string bool_bit_string(const std::vector<DetectorResult>& detectors) {
    std::string out;
    out.reserve(detectors.size());
    for (auto& d : detectors) {
        out.push_back(d.value ? '1' : '0');
    }
    return out;
}

std::string int_bit_string(const std::vector<int>& bits) {
    std::string out;
    out.reserve(bits.size());
    for (auto bit : bits) {
        out.push_back(bit ? '1' : '0');
    }
    return out;
}

double required_probability_arg(const QuantumGate& op) {
    if (op.args.size() != 1) {
        throw std::runtime_error(op.gate + " requires exactly one probability argument");
    }
    return validate_probability(op.args[0], op.gate);
}

double optional_measurement_probability_arg(const QuantumGate& op) {
    if (op.args.empty()) {
        return 0.0;
    }
    if (op.args.size() != 1) {
        throw std::runtime_error(op.gate + " requires zero or one probability argument");
    }
    return validate_probability(op.args[0], op.gate);
}

bool sample_probability(double probability, std::mt19937& rng, std::uniform_real_distribution<double>& dist) {
    if (probability <= 0.0) {
        return false;
    }
    if (probability >= 1.0) {
        return true;
    }
    return dist(rng) < probability;
}

void apply_measurement_flip(MeasureResults& measurement) {
    measurement.reg = measurement.reg ? 0 : 1;
    std::swap(measurement.out0, measurement.out1);
    measurement.ev = -measurement.ev;
}

void apply_measurement_random_flip(MeasureResults& measurement, double probability,
                                   std::mt19937& rng, std::uniform_real_distribution<double>& dist) {
    double out0 = measurement.out0;
    double out1 = measurement.out1;
    if (sample_probability(probability, rng, dist)) {
        measurement.reg = measurement.reg ? 0 : 1;
    }
    measurement.out0 = (1.0 - probability) * out0 + probability * out1;
    measurement.out1 = (1.0 - probability) * out1 + probability * out0;
    measurement.ev = measurement.out0 - measurement.out1;
}

void apply_pauli_by_index(Stabilizer& stabilizer, int qubit, int pauli) {
    if (pauli == 1) {
        stabilizer._x(qubit);
    } else if (pauli == 2) {
        stabilizer._y(qubit);
    } else if (pauli == 3) {
        stabilizer._z(qubit);
    }
}

} // namespace

void Stabilizer::_x(int qubit){

    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= z_col;
    return;
}

void Stabilizer::_y(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= x_col ^ z_col;
    return;
}

void Stabilizer::_z(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= x_col;
    return;
}

void Stabilizer::_h(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= (x_col & z_col);
    auto tmp = xt::eval(x_col);
    x_col = z_col;
    z_col = tmp;
    return;
}

void Stabilizer::_s(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= (x_col & z_col);
    z_col ^= x_col;
    return;
}

void Stabilizer::_sdg(int qubit){

    auto x_col = xt::col(this->tableau, qubit);
    auto z_col = xt::col(this->tableau, this->num_qubits + qubit);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);
    phase ^= (x_col & ~z_col);
    z_col ^= x_col;
    return;
}

void Stabilizer::_t(int qubit){

    std::vector<complex_t> coefs;
    std::vector<xt::xtensor<bool, 1>> destab_list;
    std::vector<xt::xtensor<bool, 1>> stab_list;
    this->tgate_decomp(coefs, destab_list, stab_list, qubit, false);
    /*
    std::cout << "coefs:\n";
    for (auto &k: coefs){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "destab list:\n";
    for (auto &k: destab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "stab list:\n";
    for (auto &k: stab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "========" << std::endl;*/

    this->update_xvec(coefs, destab_list, stab_list);
    
}

void Stabilizer::_tdg(int qubit){

    std::vector<complex_t> coefs;
    std::vector<xt::xtensor<bool, 1>> destab_list;
    std::vector<xt::xtensor<bool, 1>> stab_list;
    this->tgate_decomp(coefs, destab_list, stab_list, qubit, true);
    /*
    std::cout << "coefs:\n";
    for (auto &k: coefs){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "destab list:\n";
    for (auto &k: destab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    std::cout << "stab list:\n";
    for (auto &k: stab_list){
        std::cout << k << ' ';
    }
    std::cout << "\n";
    */
    this->update_xvec(coefs, destab_list, stab_list);
}

void Stabilizer::_cx(int c, int t){
    auto x0 = xt::col(this->tableau, c);
    auto z0 = xt::col(this->tableau, this->num_qubits + c);
    auto x1 = xt::col(this->tableau, t);
    auto z1 = xt::col(this->tableau, this->num_qubits + t);
    auto phase = xt::col(this->tableau, this->num_qubits * 2);

    phase ^= (x1 ^ z0 ^ true) & z1 & x0;
    x1 ^= x0;
    z0 ^= z1;
    return;
}


void Stabilizer::_reset(int qubit){

    auto res = this->_measure(qubit, true);
    if (res.reg == 1){
        this->_x(qubit);
    }
}

int Stabilizer::calc_g(bool x1, bool z1, bool x2, bool z2){
    
    if (!x1 && !z1){
        return 0;
    }else if (x1 && z1){
        return z2 - x2;
    }else if (x1 && !z1){
        return z2 * (2 * x2 - 1);
    }else{
        return x2 * (1 - 2 * z2);
    }
}


void Stabilizer::rowsum(int row1, int row2){
    auto h = xt::row(this->tableau, row1);
    auto i = xt::row(this->tableau, row2);
    int g_cnt = 0;
    int n = this->num_qubits;
    for (int j = 0; j < n; j++){
        g_cnt += calc_g(i(j), i(j+n), h(j), h(j+n));
    }
    bool res = ((2*h(2*n) + 2*i(2*n) + g_cnt) % 4 == 0);
    h ^= i;
    h(n * 2) = !res;
    return;
}

void Stabilizer::multiply_bool_pauli(complex_t&  phase, xt::xtensor<bool, 1>& pauli1, const xt::xtensor<bool, 1> pauli2){

    xt::xarray<complex_t> phase_mat = {
        { complex_t(1,0), complex_t(1,0),  complex_t(1,0),  complex_t(1,0) },
        { complex_t(1,0), complex_t(1,0),  complex_t(0,1),  complex_t(0,-1) },
        { complex_t(1,0), complex_t(0,-1), complex_t(1,0),  complex_t(0,1) },
        { complex_t(1,0), complex_t(0,1),  complex_t(0,-1), complex_t(1,0) }
    };

    for (int i = 0; i < this->num_qubits; i++){
        auto idx1 = pauli1(i) * 2 + pauli1(i+this->num_qubits);
        auto idx2 = pauli2(i) * 2 + pauli2(i+this->num_qubits);
        phase *= phase_mat(idx1, idx2);
    }
    phase *= (pauli2(2*this->num_qubits) ? complex_t(-1, 0) : complex_t(1,0));
    pauli1 ^= pauli2;
}

int Stabilizer::check_comm(xt::xtensor<bool, 1>& gate, xt::xtensor<bool, 1> entry, xt::xtensor<bool, 1> complement, complex_t& phase, xt::xtensor<bool, 1>& pauli, int qubit){

    int comm = 1;
    if (gate(qubit) == false && gate(qubit+this->num_qubits) == false) return 0;
    if (entry(qubit) == false && entry(qubit+this->num_qubits) == false) return 0;

    if (gate(qubit) != entry(qubit) || gate(qubit+this->num_qubits) != entry(qubit+this->num_qubits)){
        comm = -1;
    }

    if (comm > 0) return 0;

    this->multiply_bool_pauli(phase, pauli, complement);
    return 1;

}

void Stabilizer::gate_decomposition(xt::xtensor<bool, 1>& gate, xt::xtensor<bool, 1>& destab, xt::xtensor<bool, 1>& stab, complex_t& phase, int qubit){
    if (!xt::any(gate)) {
        return;
    }
    
    xt::xtensor<bool, 1> accum = xt::zeros<bool>({this->num_qubits * 2 + 1});
    for (int i = 0; i < this->num_qubits; i++){
        int res = this->check_comm(gate, xt::row(this->tableau, i), xt::row(this->tableau, i+this->num_qubits), phase, accum, qubit);
        stab(i) = bool(res);
    }
    for (int i = 0; i < this->num_qubits; i++){
        int res = this->check_comm(gate, xt::row(this->tableau, i+this->num_qubits), xt::row(this->tableau, i), phase, accum, qubit);
        destab(i) = bool(res);
    }

    return;
}

void Stabilizer::bin(long long value, xt::xtensor<bool, 1>& bin_v){

    int i = bin_v.size() - 1;
    while (value > 0){
        bin_v(i) = bool(value % 2);
        value /= 2;
        i -= 1;
    }
    return;
}

long long Stabilizer::convert(xt::xtensor<bool, 1>& bin_a, xt::xtensor<bool, 1>& bin_b){
    auto bin_t = bin_a ^ bin_b;
    int m = bin_a.size();
    long long res = 0;
    for (int i = 0; i < m; i++){
        res *= 2;
        if (bin_t(i)){
            res += 1;
        }
    }
    return res;
}

void Stabilizer::meas_tableau(xt::xtensor<bool, 1>& obs, xt::xtensor<bool, 1>& destab, xt::xtensor<bool, 1>& stab, int sign){
    int k = xt::from_indices(xt::argwhere(destab))(0, 0);
    
    for (int i = 0; i < this->num_qubits; i++){
        if ( i == k ) continue;
        if (destab(i)){
            this->rowsum(i+this->num_qubits, k+this->num_qubits);
        }
    }
    for ( int i = 0; i < this->num_qubits; i++){
        if (stab(i)){
            this->rowsum(i, k+this->num_qubits);
        }
        if (i == k){
            auto h = xt::row(this->tableau, i);
            h = xt::eval(xt::row(this->tableau, k+this->num_qubits));
        }
    }
    auto h = xt::row(this->tableau, k+this->num_qubits);
    h = xt::eval(obs);
    h(this->num_qubits * 2) = bool(sign); 
}

void Stabilizer::renorm(){

    double total = 0.;
    for (auto &kv : this->xvec) {
        total += (kv.second * std::conj(kv.second)).real();
    }
    total = std::sqrt(total);
    for (auto &kv : this->xvec) {
        kv.second /= total;
    }
}

MeasureResults Stabilizer::_measure(int qubit, bool reset_flag){

    xt::xtensor<bool, 1> obs = xt::zeros<bool>({this->num_qubits * 2 +1});
    obs(qubit+this->num_qubits) = true;
    xt::xtensor<bool, 1> destab = xt::zeros<bool>({this->num_qubits});
    xt::xtensor<bool, 1> stab = xt::zeros<bool>({this->num_qubits});

    complex_t phase = complex_t(1, 0);
    this->gate_decomposition(obs, destab, stab, phase, qubit);

    /*
    if (!reset_flag){
        std::cout << phase << std::endl;
        std::cout << destab << std::endl;
        std::cout << stab << std::endl;
    }*/

    std::unordered_map<long long, std::complex<double>> new_xvec_0;
    std::unordered_map<long long, std::complex<double>> new_xvec_1;

    std::vector<long long> keys;
    keys.reserve(this->xvec.size());
    for (auto &kv : this->xvec) { 
        keys.push_back(kv.first);
    }
    int keys_size = keys.size();
    
    std::mt19937 rng;                 // 默认使用 random_device 或固定种子均可
    rng.seed(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    double out0 = 0.0;
    double out1 = 0.0;
    double ev   = 0.0;
    int outcome = 0;
    omp_threshold = this->omp_threshold;
    if (!xt::any(destab)){
        
        for (int i = 0; i < keys_size; i++) {
            long long c = keys[i];
            xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
            bin(c, c_bin);
            complex_t val = this->xvec[c];
            assert(std::abs(phase.imag()) < 1e-5);
            int tmp_v = ( xt::sum<int>(stab * c_bin)(0) % 2 == 0) ? 1 : -1;
            if (phase.real() * tmp_v > 0) 
            {
                new_xvec_0[c] += val;
                new_xvec_1[c] = 0;
                out0 += (val * std::conj(val)).real();
            } else {
                new_xvec_1[c] += val;
                new_xvec_0[c] = 0;
                out1 += (val * std::conj(val)).real();
            }
        }
        auto total = out0 + out1;
        if (std::abs(total - 1) > 1e6){
            out0 /= total;
            out1 /= total;
        }
        ev = out0 - out1;
        outcome = dist(rng) > out0 ? 1: 0;
        if (reset_flag){
            if (out0 > 1e-5) outcome = 0;
        }

    }else{
        
        xt::xtensor<bool, 1> k = xt::zeros<bool>({this->num_qubits});
        int idx = xt::from_indices(xt::argwhere(destab))(0, 0);
        k(idx) = true;
        
        /*
        #pragma omp parallel if( keys_size > omp_threshold)
        {
            std::unordered_map<long long, std::complex<double>> local_xvec_0;
                std::unordered_map<long long, std::complex<double>> local_xvec_1;
            double local_ev = 0.0;
            #pragma omp for nowait
            for (int i = 0; i < keys_size; i++) {
                long long c = keys[i];
                complex_t val = this->xvec[c];
                double coef = 1.0 / std::sqrt(2);
                complex_t coef0, coef1;
                xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
                bin(c, c_bin);
                double tmp_v = ( xt::sum<int>(stab * c_bin)(0) % 2 == 0) ? 1. : -1.;
                long long target_ind = c;
                if (xt::sum<int>(k * c_bin)(0) % 2 == 1){
                    coef0 = coef * phase * tmp_v;
                    coef1 = coef0 * -1.0;
                    target_ind = this->convert(c_bin, destab);
                }else{
                    coef0 = coef;
                    coef1 = coef;
                    target_ind = c;
                }
                local_xvec_0[target_ind] += val * coef0;
                local_xvec_1[target_ind] += val * coef1;
                std::cout << "update ev" << std::endl;
                if (this->xvec.find(target_ind) != this->xvec.end()){
                    std::cout << val << " " << std::conj(this->xvec[target_ind]) << std::endl;
                    auto v = phase * val * std::conj(this->xvec[target_ind]);
                    assert(abs(v.imag()) < 1e-5);
                    local_ev += v.real() * tmp_v;
                }
            } // end for
            // Now we do a critical section or parallel reduction:
            #pragma omp critical
            {
                ev += local_ev;
                for (auto &kv : local_xvec_0) {
                    new_xvec_0[kv.first] += kv.second;
                }
                for (auto &kv : local_xvec_1) {
                    new_xvec_1[kv.first] += kv.second;
                }
            }  
        } // end parallel
        */
        for (int i = 0; i < keys_size; i++) {
            long long c = keys[i];
            complex_t val = this->xvec[c];
            double coef = 1.0 / std::sqrt(2);
            complex_t coef0, coef1;
            xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
            bin(c, c_bin);
            double tmp_v = ( xt::sum<int>(stab * c_bin)(0) % 2 == 0) ? 1. : -1.;
            long long target_ind = c;
            if (xt::sum<int>(k * c_bin)(0) % 2 == 1){
                coef0 = coef * phase * tmp_v;
                coef1 = coef0 * -1.0;
                target_ind = this->convert(c_bin, destab);
            }else{
                coef0 = coef;
                coef1 = coef;
                target_ind = c;
            }
            new_xvec_0[target_ind] += val * coef0;
            new_xvec_1[target_ind] += val * coef1;
            auto t_ind = this->convert(c_bin, destab);
            if (this->xvec.find(t_ind) != this->xvec.end()){
                auto v = phase * val * std::conj(this->xvec[t_ind]);
                assert(abs(v.imag()) < 1e-5);
                ev += v.real() * tmp_v;
            }
        }
        out0 = (1+ev) / 2;
        out1 = (1-ev) / 2;
        outcome = dist(rng) > out0 ? 1: 0;
        if (reset_flag){
            if (out0 > 1e-5) outcome = 0;
        }
        this->meas_tableau(obs, destab, stab, outcome);
    }

    if (outcome){
        this->xvec = std::move(new_xvec_1);
    }else{
        this->xvec = std::move(new_xvec_0);
    }
    this->renorm();
    
    MeasureResults res;
    res.out0 = out0;
    res.out1 = out1;
    res.ev = out0 - out1;
    res.reg = outcome;
    /*
    if (!reset_flag){
        std::cout << "ev: " << res.ev << ", out0: " << res.out0 << ", out1: " << res.out1 << std::endl;
    }*/
    return res;
}

MeasureResults Stabilizer::_measure_pauli_product(const std::vector<int>& targets, const std::vector<double>& paulis, bool invert){
    if (targets.empty()) {
        throw std::runtime_error("MPP requires at least one target");
    }
    if (targets.size() != paulis.size()) {
        throw std::runtime_error("MPP target and Pauli counts do not match");
    }

    std::vector<int> pauli_codes;
    pauli_codes.reserve(paulis.size());
    std::vector<bool> seen(this->num_qubits, false);
    for (size_t i = 0; i < targets.size(); i++) {
        int qubit = targets[i];
        if (qubit < 0 || qubit >= this->num_qubits) {
            throw std::runtime_error("Qubit target " + std::to_string(qubit) + " for MPP is outside simulator range");
        }
        if (seen[qubit]) {
            throw std::runtime_error("MPP contains the same qubit more than once");
        }
        seen[qubit] = true;
        int pauli = static_cast<int>(std::round(paulis[i]));
        if (pauli < 1 || pauli > 3) {
            throw std::runtime_error("MPP Pauli code must be 1, 2, or 3");
        }
        pauli_codes.push_back(pauli);
    }

    for (size_t i = 0; i < targets.size(); i++) {
        int qubit = targets[i];
        if (pauli_codes[i] == 1) {
            this->_h(qubit);
        } else if (pauli_codes[i] == 2) {
            this->_sdg(qubit);
            this->_h(qubit);
        }
    }

    int pivot = targets[0];
    for (size_t i = 1; i < targets.size(); i++) {
        this->_cx(targets[i], pivot);
    }

    auto measurement = this->_measure(pivot, false);

    for (int i = static_cast<int>(targets.size()) - 1; i >= 1; i--) {
        this->_cx(targets[i], pivot);
    }
    for (int i = static_cast<int>(targets.size()) - 1; i >= 0; i--) {
        int qubit = targets[i];
        if (pauli_codes[i] == 1) {
            this->_h(qubit);
        } else if (pauli_codes[i] == 2) {
            this->_h(qubit);
            this->_s(qubit);
        }
    }

    if (invert) {
        apply_measurement_flip(measurement);
    }
    return measurement;
}

void Stabilizer::tgate_decomp(std::vector<complex_t>& coefs, std::vector<xt::xtensor<bool, 1>>& destab_list, std::vector<xt::xtensor<bool, 1>>& stab_list, int qubit, bool dag){

    coefs.push_back(complex_t(std::cos(PI/8)));
    destab_list.push_back(xt::zeros<bool>({this->num_qubits}));
    stab_list.push_back(xt::zeros<bool>({this->num_qubits}));
    
    double sign = dag ? -1.: 1.;
    
    xt::xtensor<bool, 1> gate = xt::zeros<bool>({this->num_qubits * 2 + 1});
    gate(qubit + this->num_qubits) = true;
    xt::xtensor<bool, 1> destab = xt::zeros<bool>({this->num_qubits});
    xt::xtensor<bool, 1> stab = xt::zeros<bool>({this->num_qubits});
    complex_t phase = complex_t(1, 0);
    this->gate_decomposition(gate, destab, stab, phase, qubit);
    coefs.push_back(complex_t(0, -1*std::sin(PI/8)) * phase * sign);
    destab_list.push_back(destab);
    stab_list.push_back(stab);
}

void Stabilizer::update_xvec(std::vector<complex_t>& coefs, std::vector<xt::xtensor<bool, 1>>& destab_list, std::vector<xt::xtensor<bool, 1>>& stab_list){
    
    int m = coefs.size();
    std::vector<long long> keys;
    keys.reserve(this->xvec.size());
    for (auto &kv : this->xvec) { 
        keys.push_back(kv.first);
    }
    int keys_size = keys.size();
    int omp_threshold = this->omp_threshold;
    std::unordered_map<long long, std::complex<double>> new_xvec;

    for (int i = 0; i < m; i++){
        auto co = coefs[i];
        if (std::abs(coefs[i]) < 1e-6) continue;
        /*
        #pragma omp parallel if( keys_size > omp_threshold)
        {
            std::unordered_map<long long, std::complex<double>> local_xvec;
            #pragma omp for nowait
            for (int j = 0; j < keys_size; j++) {
                long long c = keys[j];
                complex_t val = this->xvec[c];
                xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
                bin(c, c_bin);
                long long target_ind = this->convert(c_bin, destab_list[i]);
                int tmp_v = ( xt::sum<int>(stab_list[i] * c_bin)(0) % 2 == 0) ? 1 : -1;
                local_xvec[target_ind] += co * complex_t(tmp_v, 0) * val;
            } // end for
            // Now we do a critical section or parallel reduction:
            #pragma omp critical
            {
                for (auto &kv : local_xvec) {
                    new_xvec[kv.first] += kv.second;
                }
            }  
        } // end parallel
        */
        for (int j = 0; j < keys_size; j++) {
            long long c = keys[j];
            complex_t val = this->xvec[c];
            xt::xtensor<bool, 1> c_bin = xt::zeros<bool>({this->num_qubits});
            bin(c, c_bin);
            long long target_ind = this->convert(c_bin, destab_list[i]);
            auto tmp_v = ( xt::sum<int>(stab_list[i] * c_bin)(0) % 2 == 0) ? 1. : -1.;
            new_xvec[target_ind] += co * tmp_v * val;
            //std::cout << c << ' ' << target_ind << ' ' << new_xvec[target_ind] << std::endl;
        } // end for
    }

    this->xvec = std::move(new_xvec);
    this->renorm();
}


void Stabilizer::init(){
    this->tableau = xt::zeros<bool>({num_qubits * 2, num_qubits * 2 + 1});
    for(int i = 0; i < 2 * num_qubits; i++){
        this->tableau(i, i) = true;
    }
    this->xvec[0] = complex_t(1, 0);
}

std::vector<MeasureResults> Stabilizer::sim(std::vector<QuantumGate>& circuit){
    return this->run(circuit, true, true).measurements;
}

SimulationResult Stabilizer::run(std::vector<QuantumGate>& circuit, bool stop_on_detector, bool print_detector_errors){

    SimulationResult result;

    auto require_qubit = [this](int qubit, const std::string& gate) {
        if (qubit < 0 || qubit >= this->num_qubits) {
            throw std::runtime_error("Qubit target " + std::to_string(qubit) + " for gate '" + gate +
                                     "' is outside simulator range [0, " + std::to_string(this->num_qubits) + ")");
        }
    };

    std::mt19937 rng;
    rng.seed(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> depolarize1_dist(1, 3);
    std::uniform_int_distribution<int> depolarize2_dist(1, 15);

    auto apply_single_qubit_noise = [this, &require_qubit, &rng, &dist](const QuantumGate& op, int pauli) {
        require_qubit(op.targets.at(0), op.gate);
        double probability = required_probability_arg(op);
        if (sample_probability(probability, rng, dist)) {
            apply_pauli_by_index(*this, op.targets[0], pauli);
        }
    };

    auto apply_depolarize1 = [this, &require_qubit, &rng, &dist, &depolarize1_dist](const QuantumGate& op) {
        require_qubit(op.targets.at(0), op.gate);
        double probability = required_probability_arg(op);
        if (sample_probability(probability, rng, dist)) {
            apply_pauli_by_index(*this, op.targets[0], depolarize1_dist(rng));
        }
    };

    auto apply_depolarize2 = [this, &require_qubit, &rng, &dist, &depolarize2_dist](const QuantumGate& op) {
        if (op.targets.size() != 2) {
            throw std::runtime_error(op.gate + " requires exactly two targets");
        }
        require_qubit(op.targets.at(0), op.gate);
        require_qubit(op.targets.at(1), op.gate);
        double probability = required_probability_arg(op);
        if (sample_probability(probability, rng, dist)) {
            int value = depolarize2_dist(rng);
            apply_pauli_by_index(*this, op.targets[0], value % 4);
            apply_pauli_by_index(*this, op.targets[1], (value / 4) % 4);
        }
    };

    for (size_t op_index = 0; op_index < circuit.size(); op_index++) {
        const auto &op = circuit[op_index];
        result.executed_operations = static_cast<int>(op_index + 1);
        if ( op.gate == "X"){
            require_qubit(op.targets.at(0), op.gate);
            this->_x(op.targets[0]);
        }else if ( op.gate == "Y"){
            require_qubit(op.targets.at(0), op.gate);
            this->_y(op.targets[0]);
        }else if ( op.gate == "Z"){
            require_qubit(op.targets.at(0), op.gate);
            this->_z(op.targets[0]);
        }else if ( op.gate == "H"){
            require_qubit(op.targets.at(0), op.gate);
            this->_h(op.targets[0]);
        }else if ( op.gate == "S"){
            require_qubit(op.targets.at(0), op.gate);
            this->_s(op.targets[0]);
        }else if ( op.gate == "SDG"){
            require_qubit(op.targets.at(0), op.gate);
            this->_sdg(op.targets[0]);
        }else if ( op.gate == "T"){
            require_qubit(op.targets.at(0), op.gate);
            this->_t(op.targets[0]);
        }else if ( op.gate == "TDG"){
            require_qubit(op.targets.at(0), op.gate);
            this->_tdg(op.targets[0]);
        }else if ( op.gate == "R"){
            require_qubit(op.targets.at(0), op.gate);
            this->_reset(op.targets[0]);
        }else if (op.gate == "CX"){
            require_qubit(op.targets.at(0), op.gate);
            require_qubit(op.targets.at(1), op.gate);
            this->_cx(op.targets[0], op.targets[1]);
        }else if (op.gate == "FEEDBACK_X" || op.gate == "FEEDBACK_Y" || op.gate == "FEEDBACK_Z"){
            if (op.targets.size() != 2) {
                throw std::runtime_error(op.gate + " requires a measurement record control and a qubit target");
            }
            require_qubit(op.targets.at(1), op.gate);
            bool enabled = measurement_ref_value(result.measurements, op.targets.at(0), op.gate);
            if (op.invert) {
                enabled = !enabled;
            }
            if (enabled) {
                if (op.gate == "FEEDBACK_X") {
                    this->_x(op.targets[1]);
                } else if (op.gate == "FEEDBACK_Y") {
                    this->_y(op.targets[1]);
                } else {
                    this->_z(op.targets[1]);
                }
            }
        }else if (op.gate == "X_ERROR" || op.gate == "XERR"){
            apply_single_qubit_noise(op, 1);
        }else if (op.gate == "Y_ERROR" || op.gate == "YERR"){
            apply_single_qubit_noise(op, 2);
        }else if (op.gate == "Z_ERROR" || op.gate == "ZERR"){
            apply_single_qubit_noise(op, 3);
        }else if (op.gate == "DEPOLARIZE1" || op.gate == "DEP1"){
            apply_depolarize1(op);
        }else if (op.gate == "DEPOLARIZE2" || op.gate == "DEP2"){
            apply_depolarize2(op);
        }else if (op.gate == "M"){
            require_qubit(op.targets.at(0), op.gate);
            double measurement_flip_probability = optional_measurement_probability_arg(op);
            auto measurement = this->_measure(op.targets[0], false);
            if (op.invert) {
                apply_measurement_flip(measurement);
            }
            if (measurement_flip_probability > 0.0) {
                apply_measurement_random_flip(measurement, measurement_flip_probability, rng, dist);
            }
            result.measurements.push_back(measurement);
        }else if (op.gate == "MPP"){
            for (auto target : op.targets) {
                require_qubit(target, op.gate);
            }
            auto measurement = this->_measure_pauli_product(op.targets, op.args, op.invert);
            result.measurements.push_back(measurement);
        }else if (op.gate == "DETECTOR"){
            DetectorResult detector;
            detector.targets = op.targets;
            detector.value = op.invert;
            for (auto ref: op.targets){
                detector.value = detector.value ^ measurement_ref_value(result.measurements, ref, op.gate);
            }
            result.detectors.push_back(detector);
            if (detector.value){
                result.discarded = true;
                result.failed_detector_index = static_cast<int>(result.detectors.size()) - 1;
                if (print_detector_errors) {
                    std::cout << "detector error, discard the shot!\n";
                    for (auto ref: op.targets){
                        std::cout << ref << ' ';
                    }
                    std::cout << '\n';
                }
                if (stop_on_detector) {
                    break;
                }
            }
        }else if (op.gate == "OBSERVABLE_INCLUDE"){
            ObservableResult observable;
            observable.index = op.args.empty() ? 0 : static_cast<int>(op.args[0]);
            observable.targets = op.targets;
            observable.value = op.invert;
            if (observable.index < 0) {
                throw std::runtime_error("OBSERVABLE_INCLUDE index cannot be negative");
            }
            for (auto ref: op.targets){
                observable.value = observable.value ^ measurement_ref_value(result.measurements, ref, op.gate);
            }
            if (static_cast<int>(result.observable_bits.size()) <= observable.index) {
                result.observable_bits.resize(observable.index + 1, 0);
            }
            result.observable_bits[observable.index] ^= observable.value ? 1 : 0;
            result.observables.push_back(observable);
        }
        else{
            throw std::runtime_error(op.gate + " is not supported");
        }
    }

    return result;
}

void Stabilizer::print_xvec(){
    
    std::vector<std::tuple<long long, complex_t>> p;
    for (auto &k: this->xvec){
        if ((k.second * std::conj(k.second)).real() < 1e-5) continue;
        p.push_back({k.first, k.second});
    }

    // Define custom comparator
    auto tuple_comparator = [](const auto& a, const auto& b) {
        // First compare by the long long (first element)
        if (std::get<0>(a) != std::get<0>(b)) {
            return std::get<0>(a) < std::get<0>(b);
        }
        // If long longs are equal, compare by complex magnitude
        return std::abs(std::get<1>(a)) < std::abs(std::get<1>(b));
    };

    sort(p.begin(),p.end(), tuple_comparator);
    for(auto &i: p)
    {
        std::cout << std::get<0>(i) << ' ' << std::get<1>(i)  << std::endl;
    }
}

xt::xtensor<bool, 1> Stabilizer::phase(){
    return xt::col(this->tableau, this->num_qubits * 2);
}


CircuitInput load_stim_text(const std::string& text, const std::string& source_name){
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    CircuitInput input;
    input.source_path = source_name;
    int max_qubit = -1;
    size_t pos = 0;
    parse_stim_block(lines, pos, input.circuit, max_qubit, false);
    input.num_qubits = max_qubit + 1;
    return input;
}

CircuitInput load_stim_file(const std::string& path){
    std::ifstream infile(path.c_str());
    if (!infile) {
        throw std::runtime_error("Cannot open stim file: " + path);
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(infile, line)) {
        lines.push_back(line);
    }

    CircuitInput input;
    input.source_path = path;
    int max_qubit = -1;
    size_t pos = 0;
    parse_stim_block(lines, pos, input.circuit, max_qubit, false);
    input.num_qubits = max_qubit + 1;
    return input;
}

std::vector<SimulationResult> run_stim_file(const std::string& path, int shots, int num_qubits, bool stop_on_detector){
    if (shots < 0) {
        throw std::runtime_error("shots must be non-negative");
    }
    auto input = load_stim_file(path);
    int resolved_num_qubits = num_qubits > 0 ? num_qubits : input.num_qubits;
    if (resolved_num_qubits < input.num_qubits) {
        throw std::runtime_error("num_qubits override is smaller than the largest qubit referenced by the stim file");
    }

    std::vector<SimulationResult> results;
    results.reserve(shots);
    for (int shot = 0; shot < shots; shot++) {
        Stabilizer stb(resolved_num_qubits);
        results.push_back(stb.run(input.circuit, stop_on_detector, false));
    }
    return results;
}

std::vector<SimulationResult> run_stim_text(const std::string& text, int shots, int num_qubits, bool stop_on_detector){
    if (shots < 0) {
        throw std::runtime_error("shots must be non-negative");
    }
    auto input = load_stim_text(text);
    int resolved_num_qubits = num_qubits > 0 ? num_qubits : input.num_qubits;
    if (resolved_num_qubits < input.num_qubits) {
        throw std::runtime_error("num_qubits override is smaller than the largest qubit referenced by the stim text");
    }

    std::vector<SimulationResult> results;
    results.reserve(shots);
    for (int shot = 0; shot < shots; shot++) {
        Stabilizer stb(resolved_num_qubits);
        results.push_back(stb.run(input.circuit, stop_on_detector, false));
    }
    return results;
}

std::string shot_result_to_json(const SimulationResult& result){
    std::ostringstream oss;
    oss << "{"
        << "\"discarded\":" << (result.discarded ? "true" : "false")
        << ",\"failed_detector_index\":" << result.failed_detector_index
        << ",\"executed_operations\":" << result.executed_operations
        << ",\"measurement_count\":" << result.measurements.size()
        << ",\"detector_count\":" << result.detectors.size()
        << ",\"observable_count\":" << result.observable_bits.size()
        << ",\"measurements\":\"" << bit_string(result.measurements) << "\""
        << ",\"detectors\":\"" << bool_bit_string(result.detectors) << "\""
        << ",\"observables\":\"" << int_bit_string(result.observable_bits) << "\""
        << "}";
    return oss.str();
}

void write_shot_results_jsonl(const std::string& path, const std::vector<SimulationResult>& results){
    std::ofstream outfile(path.c_str());
    if (!outfile) {
        throw std::runtime_error("Cannot open output file: " + path);
    }
    for (auto& result : results) {
        outfile << shot_result_to_json(result) << '\n';
    }
}
