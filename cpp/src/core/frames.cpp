#include "core/frames.hpp"

#include "core/internal.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace symft {
using namespace detail;

CliffordFrame::CliffordFrame(int nqubits_)
    : nqubits(checked_nqubits(nqubits_)),
      tableau_core(nqubits) {}

int CliffordFrame::xrow(int q) const {
    return tableau_core.xrow(q);
}

int CliffordFrame::zrow(int q) const {
    return tableau_core.zrow(q);
}

const PauliString& CliffordFrame::generator(int row) const {
    if (row < 0) {
        fail("invalid Clifford frame row");
    }
    return tableau_core.generator(static_cast<std::size_t>(row));
}

const std::vector<PauliString>& CliffordFrame::generators() const {
    return tableau_core.generators();
}

void CliffordFrame::copy_pauli_to_row(int row, const PauliString& pauli) {
    if (pauli.nqubits != nqubits || row < 0 ||
        static_cast<std::size_t>(row) >= tableau_core.row_count()) {
        fail("invalid Clifford frame row assignment");
    }
    tableau_core.assign_generator(static_cast<std::size_t>(row), pauli);
}

const std::vector<CliffordFrame::SupportWord>& CliffordFrame::support_for_row(int row) const {
    if (row < 0 || static_cast<std::size_t>(row) >= tableau_core.row_count()) {
        fail("invalid Clifford frame row");
    }
    if (support_generation != tableau_core.body_generation()) {
        support_words.clear();
        support_words.resize(tableau_core.row_count());
        for (std::size_t r = 0; r < tableau_core.row_count(); ++r) {
            const auto& pauli = tableau_core.generator(r);
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
        support_generation = tableau_core.body_generation();
    }
    return support_words[static_cast<std::size_t>(row)];
}

bool operator==(const CliffordFrame& lhs, const CliffordFrame& rhs) {
    return lhs.nqubits == rhs.nqubits && lhs.generators() == rhs.generators();
}

SymbolicPauliString::SymbolicPauliString(PauliString pauli_) : pauli(std::move(pauli_)), sign(false) {}

SymbolicPauliString::SymbolicPauliString(PauliString pauli_, SymbolicBool sign_)
    : pauli(std::move(pauli_)), sign(std::move(sign_)) {}

bool operator==(const SymbolicPauliString& lhs, const SymbolicPauliString& rhs) {
    return lhs.pauli == rhs.pauli && lhs.sign == rhs.sign;
}

SymbolicPauliFrame::SymbolicPauliFrame(int nqubits_)
    : nqubits(checked_nqubits(nqubits_)) {}

void SymbolicPauliFrame::add_pauli(const PauliString& pauli, int condition) {
    if (pauli.nqubits != nqubits) {
        fail("Pauli string dimension does not match symbolic Pauli frame");
    }
    if (condition <= 0) {
        fail("condition id must be positive");
    }
    const std::size_t term = term_conditions.size();
    if ((term & 63u) == 0) {
        x_term_blocks.resize(x_term_blocks.size() + static_cast<std::size_t>(nqubits), 0);
        z_term_blocks.resize(z_term_blocks.size() + static_cast<std::size_t>(nqubits), 0);
    }
    const std::size_t block = term >> 6;
    const std::size_t base = block * static_cast<std::size_t>(nqubits);
    const std::uint64_t mask = std::uint64_t{1} << (term & 63u);
    for (std::size_t word = 0; word < pauli.x.size(); ++word) {
        std::uint64_t x_bits = pauli.x[word];
        std::uint64_t z_bits = pauli.z[word];
        while (x_bits) {
            const int bit = trailing_zeros64(x_bits);
            const std::size_t q = word * 64 + static_cast<std::size_t>(bit);
            if (q < static_cast<std::size_t>(nqubits)) {
                x_term_blocks[base + q] |= mask;
            }
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const int bit = trailing_zeros64(z_bits);
            const std::size_t q = word * 64 + static_cast<std::size_t>(bit);
            if (q < static_cast<std::size_t>(nqubits)) {
                z_term_blocks[base + q] |= mask;
            }
            z_bits &= z_bits - 1;
        }
    }
    term_conditions.push_back(condition);
}

SymbolicPauliString conjugate_by(const SymbolicPauliFrame& frame, const PauliString& pauli) {
    if (pauli.nqubits != frame.nqubits) {
        fail("Pauli string dimension does not match symbolic Pauli frame");
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
    const std::size_t blocks = (frame.term_conditions.size() + 63) >> 6;
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::size_t base = block * static_cast<std::size_t>(frame.nqubits);
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
            if (term < frame.term_conditions.size()) {
                conditions.push_back(frame.term_conditions[term]);
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

SymbolicPauliString conjugate_by(const SymbolicPauliFrame& frame, const SymbolicPauliString& pauli) {
    if (pauli.pauli.nqubits != frame.nqubits) {
        fail("Pauli string dimension does not match symbolic Pauli frame");
    }
    SymbolicPauliString out = conjugate_by(frame, pauli.pauli);
    out.sign = xor_bool(out.sign, pauli.sign);
    return out;
}

} // namespace symft
