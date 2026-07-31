#include "core/frames.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace symft {
using namespace detail;

ConditionalPauliString::ConditionalPauliString(PauliString pauli_, int condition_)
    : pauli(std::move(pauli_)), condition(condition_) {
    if (condition <= 0) {
        fail("condition id must be positive");
    }
}

bool operator==(const ConditionalPauliString& lhs, const ConditionalPauliString& rhs) {
    return lhs.condition == rhs.condition && lhs.pauli == rhs.pauli;
}

SymbolicPauliString::SymbolicPauliString(PauliString pauli_) : pauli(std::move(pauli_)), sign(false) {}

SymbolicPauliString::SymbolicPauliString(PauliString pauli_, SymbolicBool sign_)
    : pauli(std::move(pauli_)), sign(std::move(sign_)) {}

bool operator==(const SymbolicPauliString& lhs, const SymbolicPauliString& rhs) {
    return lhs.pauli == rhs.pauli && lhs.sign == rhs.sign;
}

ActivePauliFrame::ActivePauliFrame(int k_) : ActivePauliFrame(k_, std::make_shared<SymbolicContext>()) {}

ActivePauliFrame::ActivePauliFrame(int k_, std::shared_ptr<SymbolicContext> context_)
    : k(checked_nqubits(k_)), context(std::move(context_)) {
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
}

ConditionalPauliString ActivePauliFrame::add_pauli(const PauliString& pauli) {
    return add_pauli(pauli, context->fresh_condition());
}

ConditionalPauliString ActivePauliFrame::add_pauli(const PauliString& pauli, int condition) {
    if (pauli.nqubits != k) {
        fail("Pauli string dimension does not match active Pauli frame");
    }
    const std::size_t term = terms.size();
    if ((term & 63u) == 0) {
        x_term_blocks.resize(x_term_blocks.size() + static_cast<std::size_t>(k), 0);
        z_term_blocks.resize(z_term_blocks.size() + static_cast<std::size_t>(k), 0);
    }
    const std::size_t block = term >> 6;
    const std::size_t base = block * static_cast<std::size_t>(k);
    const std::uint64_t mask = std::uint64_t{1} << (term & 63u);
    for (std::size_t word = 0; word < pauli.x.size(); ++word) {
        std::uint64_t x_bits = pauli.x[word];
        std::uint64_t z_bits = pauli.z[word];
        while (x_bits) {
            const int bit = trailing_zeros64(x_bits);
            const std::size_t q = word * 64 + static_cast<std::size_t>(bit);
            if (q < static_cast<std::size_t>(k)) {
                x_term_blocks[base + q] |= mask;
            }
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const int bit = trailing_zeros64(z_bits);
            const std::size_t q = word * 64 + static_cast<std::size_t>(bit);
            if (q < static_cast<std::size_t>(k)) {
                z_term_blocks[base + q] |= mask;
            }
            z_bits &= z_bits - 1;
        }
    }
    terms.emplace_back(pauli, condition);
    context->bump_next_condition(condition);
    return terms.back();
}

SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const PauliString& pauli) {
    SymbolicPauliString out(pauli);
    if (pauli_anticommutes(cp.pauli, pauli)) {
        out.sign = xor_bool(out.sign, symbolic_bool(cp.condition));
    }
    return out;
}

SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const SymbolicPauliString& pauli) {
    SymbolicPauliString out = pauli;
    if (pauli_anticommutes(cp.pauli, pauli.pauli)) {
        out.sign = xor_bool(out.sign, symbolic_bool(cp.condition));
    }
    return out;
}

SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.k) {
        fail("Pauli string dimension does not match active Pauli frame");
    }
    std::vector<int> x_qubits;
    std::vector<int> z_qubits;
    for (std::size_t word = 0; word < pauli.x.size(); ++word) {
        std::uint64_t x_bits = pauli.x[word];
        std::uint64_t z_bits = pauli.z[word];
        while (x_bits) {
            const int bit = trailing_zeros64(x_bits);
            x_qubits.push_back(static_cast<int>(word * 64 + static_cast<std::size_t>(bit)));
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const int bit = trailing_zeros64(z_bits);
            z_qubits.push_back(static_cast<int>(word * 64 + static_cast<std::size_t>(bit)));
            z_bits &= z_bits - 1;
        }
    }
    std::vector<int> conditions;
    const std::size_t blocks = (frame.terms.size() + 63) >> 6;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::size_t base = block * static_cast<std::size_t>(frame.k);
        std::uint64_t anticommuting = 0;
        for (int q : x_qubits) {
            anticommuting ^= frame.z_term_blocks[base + static_cast<std::size_t>(q)];
        }
        for (int q : z_qubits) {
            anticommuting ^= frame.x_term_blocks[base + static_cast<std::size_t>(q)];
        }
        while (anticommuting) {
            const int bit = trailing_zeros64(anticommuting);
            const std::size_t term = block * 64 + static_cast<std::size_t>(bit);
            if (term < frame.terms.size()) {
                conditions.push_back(frame.terms[term].condition);
            }
            anticommuting &= anticommuting - 1;
        }
    }
    std::sort(conditions.begin(), conditions.end());
    std::vector<int> normalized;
    normalized.reserve(conditions.size());
    for (std::size_t start = 0; start < conditions.size();) {
        std::size_t end = start + 1;
        while (end < conditions.size() && conditions[end] == conditions[start]) {
            ++end;
        }
        if ((end - start) & 1u) {
            normalized.push_back(conditions[start]);
        }
        start = end;
    }
    SymbolicBool sign;
    sign.conditions = std::move(normalized);
    return SymbolicPauliString(pauli, std::move(sign));
}

SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const SymbolicPauliString& pauli) {
    if (pauli.pauli.nqubits != frame.k) {
        fail("Pauli string dimension does not match active Pauli frame");
    }
    SymbolicPauliString out = conjugate_by(frame, pauli.pauli);
    out.sign = xor_bool(out.sign, pauli.sign);
    return out;
}

DormantState::DormantState(int d_) : DormantState(d_, std::make_shared<SymbolicContext>()) {}

DormantState::DormantState(int d_, std::shared_ptr<SymbolicContext> context_)
    : d(checked_nqubits(d_)), bits(static_cast<std::size_t>(d), SymbolicBool(false)), context(std::move(context_)) {
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
}

DormantState::DormantState(std::vector<SymbolicBool> bits_, std::shared_ptr<SymbolicContext> context_)
    : d(static_cast<int>(bits_.size())), bits(std::move(bits_)), context(std::move(context_)) {
    if (!context) {
        context = std::make_shared<SymbolicContext>();
    }
    for (const auto& bit : bits) {
        context->bump_next_condition(bit);
    }
}

SymbolicBool DormantState::dormant_bit(int q) const {
    check_qubit(d, q);
    return bits[static_cast<std::size_t>(q)];
}

void DormantState::set_dormant_bit(int q, const SymbolicBool& value) {
    check_qubit(d, q);
    bits[static_cast<std::size_t>(q)] = value;
    context->bump_next_condition(value);
}

SymbolicBool DormantState::assign_dormant_symbol(int q) {
    check_qubit(d, q);
    const SymbolicBool expr = symbolic_bool(context->fresh_condition());
    bits[static_cast<std::size_t>(q)] = expr;
    return expr;
}

CliffordFrame::CliffordFrame(int nqubits_) : nqubits(checked_nqubits(nqubits_)), nwords(nwords_for(nqubits)) {
    rows.assign(static_cast<std::size_t>(2 * nqubits), PauliString(nqubits));
    for (int q = 0; q < nqubits; ++q) {
        rows[static_cast<std::size_t>(xrow(q))].set_xbit(q);
        rows[static_cast<std::size_t>(zrow(q))].set_zbit(q);
    }
}

int CliffordFrame::xrow(int q) const {
    return check_qubit(nqubits, q);
}

int CliffordFrame::zrow(int q) const {
    return nqubits + check_qubit(nqubits, q);
}

void CliffordFrame::copy_pauli_to_row(int row, const PauliString& pauli) {
    if (pauli.nqubits != nqubits || row < 0 || row >= static_cast<int>(rows.size())) {
        fail("invalid Clifford frame row assignment");
    }
    rows[static_cast<std::size_t>(row)] = pauli;
    invalidate_support_cache();
}

void CliffordFrame::invalidate_support_cache() {
    support_words_valid = false;
    coordinate_columns_valid = false;
}

const std::vector<CliffordFrame::SupportWord>& CliffordFrame::support_for_row(int row) const {
    if (row < 0 || row >= static_cast<int>(rows.size())) {
        fail("invalid Clifford frame row");
    }
    if (!support_words_valid) {
        support_words.clear();
        support_words.resize(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r) {
            const auto& pauli = rows[r];
            auto& support = support_words[r];
            support.reserve(pauli.x.size());
            for (std::size_t word = 0; word < pauli.x.size(); ++word) {
                const std::uint64_t x_mask = pauli.x[word];
                const std::uint64_t z_mask = pauli.z[word];
                if (x_mask || z_mask) {
                    support.push_back(SupportWord{word, x_mask, z_mask});
                }
            }
        }
        support_words_valid = true;
    }
    return support_words[static_cast<std::size_t>(row)];
}

void CliffordFrame::ensure_coordinate_columns() const {
    if (coordinate_columns_valid) {
        return;
    }
    const std::size_t row_words = (rows.size() + 63) >> 6;
    x_coordinate_columns.assign(static_cast<std::size_t>(nqubits) * row_words, 0);
    z_coordinate_columns.assign(static_cast<std::size_t>(nqubits) * row_words, 0);
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto& support = support_for_row(static_cast<int>(row));
        for (const auto& word : support) {
            std::uint64_t x_bits = word.x_mask;
            std::uint64_t z_bits = word.z_mask;
            while (x_bits) {
                const std::size_t q = word.index * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
                if (q < static_cast<std::size_t>(nqubits)) {
                    x_coordinate_columns[q * row_words + (row >> 6)] |= std::uint64_t{1} << (row & 63);
                }
                x_bits &= x_bits - 1;
            }
            while (z_bits) {
                const std::size_t q = word.index * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
                if (q < static_cast<std::size_t>(nqubits)) {
                    z_coordinate_columns[q * row_words + (row >> 6)] |= std::uint64_t{1} << (row & 63);
                }
                z_bits &= z_bits - 1;
            }
        }
    }
    coordinate_columns_valid = true;
}

bool operator==(const CliffordFrame& lhs, const CliffordFrame& rhs) {
    return lhs.nqubits == rhs.nqubits && lhs.rows == rhs.rows;
}

namespace {

void swap_rows(CliffordFrame& frame, int a, int b) {
    if (a != b) {
        std::swap(frame.rows[static_cast<std::size_t>(a)], frame.rows[static_cast<std::size_t>(b)]);
        frame.invalidate_support_cache();
    }
}

void add_row_phase(CliffordFrame& frame, int row, int delta) {
    frame.rows[static_cast<std::size_t>(row)].phase_shift(delta);
}

void mul_rows(CliffordFrame& frame, int dst, int lhs, int rhs, int extra_phase = 0) {
    PauliString out = frame.rows[static_cast<std::size_t>(lhs)] * frame.rows[static_cast<std::size_t>(rhs)];
    out.phase_shift(extra_phase);
    frame.rows[static_cast<std::size_t>(dst)] = out;
    frame.invalidate_support_cache();
}

void check_two_qubit_gate(const CliffordFrame& frame, int a, int b) {
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    if (ai == bi) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
}

void right_apply_clifford(CliffordFrame& frame, const CliffordFrame& gate) {
    if (frame.nqubits != gate.nqubits) {
        fail("Clifford frames act on different numbers of qubits");
    }
    for (auto& row : frame.rows) {
        row = preimage(gate, row);
    }
    frame.invalidate_support_cache();
}

template <class Fn>
void right_apply_left_gate(CliffordFrame& frame, Fn&& fn) {
    CliffordFrame gate(frame.nqubits);
    fn(gate);
    right_apply_clifford(frame, gate);
}

} // namespace

PauliString preimage(const CliffordFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.nqubits) {
        fail("Pauli string and Clifford frame have different numbers of qubits");
    }
    PauliString out(frame.nqubits);
    out.set_phase(pauli.phase_exponent());
    const auto multiply_row = [&](int row_index) {
        const auto& row = frame.rows[static_cast<std::size_t>(row_index)];
        const auto& support = frame.support_for_row(row_index);
        int carry = 0;
        if (support.size() * 2 <= pauli.x.size()) {
            for (const auto& word : support) {
                carry += popcount64(out.z[word.index] & word.x_mask);
                out.x[word.index] ^= word.x_mask;
                out.z[word.index] ^= word.z_mask;
            }
        } else {
            for (std::size_t word = 0; word < row.x.size(); ++word) {
                carry += popcount64(out.z[word] & row.x[word]);
                out.x[word] ^= row.x[word];
                out.z[word] ^= row.z[word];
            }
        }
        out.set_phase(out.phase_exponent() + row.phase_exponent() + 2 * (carry & 1));
    };
    // The old implementation checked both bits on every qubit. Walking the
    // set bits keeps the original ascending-q, X-before-Z multiplication
    // order while avoiding an O(n) scan for sparse Paulis.
    for (std::size_t word = 0; word < pauli.x.size(); ++word) {
        std::uint64_t bits = pauli.x[word] | pauli.z[word];
        while (bits) {
            const int bit = trailing_zeros64(bits);
            const int q = static_cast<int>(word * 64 + static_cast<std::size_t>(bit));
            if (pauli.x[word] & (std::uint64_t{1} << bit)) {
                multiply_row(frame.xrow(q));
            }
            if (pauli.z[word] & (std::uint64_t{1} << bit)) {
                multiply_row(frame.zrow(q));
            }
            bits &= bits - 1;
        }
    }
    return out;
}

PauliString coordinates_in_frame(const CliffordFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.nqubits) {
        fail("Pauli string and Clifford frame have different numbers of qubits");
    }
    const std::size_t row_words = (frame.rows.size() + 63) >> 6;
    frame.ensure_coordinate_columns();
    std::vector<std::uint64_t> parity(row_words, 0);
    for (std::size_t word = 0; word < pauli.x.size(); ++word) {
        std::uint64_t x_bits = pauli.x[word];
        std::uint64_t z_bits = pauli.z[word];
        while (x_bits) {
            const std::size_t q = word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
            const std::size_t base = q * row_words;
            for (std::size_t row_word = 0; row_word < row_words; ++row_word) {
                parity[row_word] ^= frame.z_coordinate_columns[base + row_word];
            }
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const std::size_t q = word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
            const std::size_t base = q * row_words;
            for (std::size_t row_word = 0; row_word < row_words; ++row_word) {
                parity[row_word] ^= frame.x_coordinate_columns[base + row_word];
            }
            z_bits &= z_bits - 1;
        }
    }
    PauliString out(frame.nqubits);
    const std::size_t n = static_cast<std::size_t>(frame.nqubits);
    const std::size_t n_shift_words = n >> 6;
    const unsigned n_shift = static_cast<unsigned>(n & 63);
    for (std::size_t word = 0; word < out.z.size(); ++word) {
        out.z[word] = parity[word];
        if (word + 1 == out.z.size() && n_shift != 0) {
            out.z[word] &= (std::uint64_t{1} << n_shift) - 1;
        }
        std::uint64_t value = parity[n_shift_words + word] >> n_shift;
        if (n_shift != 0 && n_shift_words + word + 1 < parity.size()) {
            value |= parity[n_shift_words + word + 1] << (64 - n_shift);
        }
        out.x[word] = value;
    }
    const PauliString reconstructed = preimage(frame, out);
    if (!reconstructed.same_body(pauli)) {
        fail("frame rows do not span the Pauli body");
    }
    out.set_phase(pauli.phase_exponent() - reconstructed.phase_exponent());
    return out;
}

void left_H(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    swap_rows(frame, frame.xrow(qi), frame.zrow(qi));
}

void left_H_NXY(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, z, 2);
}

void left_H_NXZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

void left_H_NYZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 1);
}

void left_H_XY(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

void left_H_YZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 1);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
}

void left_C_NXYZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

void left_C_NZYX(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, z, 2);
}

void left_C_XNYZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
}

void left_C_XYNZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
}

void left_C_XYZ(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    mul_rows(frame, x, x, z, 1);
    swap_rows(frame, x, z);
}

void left_C_ZNYX(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 1);
}

void left_C_ZYNX(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
    add_row_phase(frame, x, 2);
    add_row_phase(frame, z, 2);
}

void left_C_ZYX(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
}

void left_S(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    mul_rows(frame, frame.xrow(qi), frame.xrow(qi), frame.zrow(qi), 3);
}

void left_SDG(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    mul_rows(frame, frame.xrow(qi), frame.xrow(qi), frame.zrow(qi), 1);
}

void left_SQRT_X(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 3);
    swap_rows(frame, x, z);
}

void left_SQRT_X_DAG(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    mul_rows(frame, x, x, z, 1);
    swap_rows(frame, x, z);
}

void left_SQRT_Y(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    add_row_phase(frame, z, 2);
}

void left_SQRT_Y_DAG(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    const int x = frame.xrow(qi);
    const int z = frame.zrow(qi);
    swap_rows(frame, x, z);
    add_row_phase(frame, x, 2);
}

void left_X(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    add_row_phase(frame, frame.zrow(qi), 2);
}

void left_Y(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    add_row_phase(frame, frame.xrow(qi), 2);
    add_row_phase(frame, frame.zrow(qi), 2);
}

void left_Z(CliffordFrame& frame, int q) {
    const int qi = check_qubit(frame.nqubits, q);
    add_row_phase(frame, frame.xrow(qi), 2);
}

void left_CX(CliffordFrame& frame, int control, int target) {
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    if (c == t) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    mul_rows(frame, frame.xrow(c), frame.xrow(c), frame.xrow(t));
    mul_rows(frame, frame.zrow(t), frame.zrow(c), frame.zrow(t));
}

void left_CY(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    mul_rows(frame, xc, xc, zc, 3);
    mul_rows(frame, xc, xc, zt);
    mul_rows(frame, xt, zc, xt);
    mul_rows(frame, xc, xc, xt);
    mul_rows(frame, zt, zc, zt);
}

void left_CZ(CliffordFrame& frame, int a, int b) {
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    if (ai == bi) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    mul_rows(frame, frame.xrow(ai), frame.xrow(ai), frame.zrow(bi));
    mul_rows(frame, frame.xrow(bi), frame.zrow(ai), frame.xrow(bi));
}

void left_SWAP(CliffordFrame& frame, int a, int b) {
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    if (ai == bi) {
        fail("two-qubit Clifford gate requires distinct qubits");
    }
    swap_rows(frame, frame.xrow(ai), frame.xrow(bi));
    swap_rows(frame, frame.zrow(ai), frame.zrow(bi));
}

void left_CXSWAP(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, xb);
    mul_rows(frame, zb, za, zb);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

void left_CZSWAP(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, zb);
    mul_rows(frame, xb, za, xb);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

void left_ISWAP(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xb, xb, zb, 3);
    mul_rows(frame, xa, xa, zb);
    mul_rows(frame, xb, za, xb);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

void left_ISWAP_DAG(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xb, xb, zb, 1);
    mul_rows(frame, xa, xa, zb);
    mul_rows(frame, xb, za, xb);
    swap_rows(frame, xa, xb);
    swap_rows(frame, za, zb);
}

void left_SQRT_XX(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xa, xa, xb);
    mul_rows(frame, zb, za, zb);
    swap_rows(frame, xa, za);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xa, xa, xb);
    mul_rows(frame, zb, za, zb);
}

void left_SQRT_XX_DAG(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xa, xa, xb);
    mul_rows(frame, zb, za, zb);
    swap_rows(frame, xa, za);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xa, xa, xb);
    mul_rows(frame, zb, za, zb);
}

void left_SQRT_YY(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xb, xb, xa);
    mul_rows(frame, za, zb, za);
    swap_rows(frame, xb, zb);
    add_row_phase(frame, xa, 2);
    mul_rows(frame, xb, xb, xa);
    mul_rows(frame, za, zb, za);
    mul_rows(frame, xa, xa, za, 3);
}

void left_SQRT_YY_DAG(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 3);
    add_row_phase(frame, xb, 2);
    mul_rows(frame, xb, xb, xa);
    mul_rows(frame, za, zb, za);
    swap_rows(frame, xb, zb);
    mul_rows(frame, xb, xb, xa);
    mul_rows(frame, za, zb, za);
    mul_rows(frame, xa, xa, za, 1);
}

void left_SQRT_ZZ(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 3);
    mul_rows(frame, xb, xb, zb, 3);
    mul_rows(frame, xa, xa, zb);
    mul_rows(frame, xb, za, xb);
}

void left_SQRT_ZZ_DAG(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, za, 1);
    mul_rows(frame, xb, xb, zb, 1);
    mul_rows(frame, xa, xa, zb);
    mul_rows(frame, xb, za, xb);
}

void left_SWAPCX(CliffordFrame& frame, int a, int b) {
    check_two_qubit_gate(frame, a, b);
    const int ai = check_qubit(frame.nqubits, a);
    const int bi = check_qubit(frame.nqubits, b);
    const int xa = frame.xrow(ai);
    const int za = frame.zrow(ai);
    const int xb = frame.xrow(bi);
    const int zb = frame.zrow(bi);
    mul_rows(frame, xa, xa, xb);
    mul_rows(frame, zb, za, zb);
    mul_rows(frame, xb, xb, xa);
    mul_rows(frame, za, zb, za);
}

void left_XCX(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    swap_rows(frame, xc, zc);
    mul_rows(frame, xc, xc, xt);
    mul_rows(frame, zt, zc, zt);
    swap_rows(frame, xc, zc);
}

void left_XCY(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    swap_rows(frame, xc, zc);
    mul_rows(frame, xc, xc, zc, 3);
    mul_rows(frame, xc, xc, zt);
    mul_rows(frame, xt, zc, xt);
    mul_rows(frame, xc, xc, xt);
    mul_rows(frame, zt, zc, zt);
    swap_rows(frame, xc, zc);
}

void left_XCZ(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    mul_rows(frame, xt, xt, xc);
    mul_rows(frame, zc, zt, zc);
}

void left_YCX(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    swap_rows(frame, xt, zt);
    mul_rows(frame, xt, xt, zt, 3);
    mul_rows(frame, xc, xc, zt);
    mul_rows(frame, xt, zc, xt);
    mul_rows(frame, xt, xt, xc);
    mul_rows(frame, zc, zt, zc);
    swap_rows(frame, xt, zt);
}

void left_YCY(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    swap_rows(frame, xc, zc);
    swap_rows(frame, xt, zt);
    mul_rows(frame, xc, xc, zc, 3);
    mul_rows(frame, xt, xt, xc);
    mul_rows(frame, zc, zt, zc);
    swap_rows(frame, xt, zt);
    mul_rows(frame, xt, xt, xc);
    mul_rows(frame, zc, zt, zc);
    mul_rows(frame, xc, xc, zc, 3);
}

void left_YCZ(CliffordFrame& frame, int control, int target) {
    check_two_qubit_gate(frame, control, target);
    const int c = check_qubit(frame.nqubits, control);
    const int t = check_qubit(frame.nqubits, target);
    const int xc = frame.xrow(c);
    const int zc = frame.zrow(c);
    const int xt = frame.xrow(t);
    const int zt = frame.zrow(t);
    mul_rows(frame, xt, xt, zt, 3);
    mul_rows(frame, xc, xc, zt);
    mul_rows(frame, xt, zc, xt);
    mul_rows(frame, xt, xt, xc);
    mul_rows(frame, zc, zt, zc);
}

void right_H(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_H(gate, q); });
}

void right_S(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_S(gate, q); });
}

void right_SDG(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_SDG(gate, q); });
}

void right_X(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_X(gate, q); });
}

void right_Z(CliffordFrame& frame, int q) {
    right_apply_left_gate(frame, [q](CliffordFrame& gate) { left_Z(gate, q); });
}

void right_CX(CliffordFrame& frame, int control, int target) {
    right_apply_left_gate(frame, [control, target](CliffordFrame& gate) { left_CX(gate, control, target); });
}

void right_CZ(CliffordFrame& frame, int a, int b) {
    right_apply_left_gate(frame, [a, b](CliffordFrame& gate) { left_CZ(gate, a, b); });
}

void right_SWAP(CliffordFrame& frame, int a, int b) {
    right_apply_left_gate(frame, [a, b](CliffordFrame& gate) { left_SWAP(gate, a, b); });
}

} // namespace symft
