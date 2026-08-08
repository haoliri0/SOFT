#pragma once

#include "core/pauli.hpp"
#include "core/symbolic.hpp"
#include "core/tableau.hpp"

namespace symft {

struct CliffordFrame {
    struct SupportWord {
        std::size_t index = 0;
        std::uint64_t x_mask = 0;
        std::uint64_t z_mask = 0;
    };

    int nqubits = 0;
    int nwords = 0;
    // For the represented Clifford U, the shared tableau stores
    // U^\dagger X_q U followed by U^\dagger Z_q U with complete phases.
    detail::TableauCore tableau_core;
    mutable std::vector<std::vector<SupportWord>> support_words;
    mutable bool support_words_valid = false;

    CliffordFrame() = default;
    explicit CliffordFrame(int nqubits);

    int xrow(int q) const;
    int zrow(int q) const;
    const PauliString& generator(int row) const;
    const std::vector<PauliString>& generators() const;
    void copy_pauli_to_row(int row, const PauliString& pauli);
    void invalidate_support_cache();
    const std::vector<SupportWord>& support_for_row(int row) const;
    void ensure_coordinate_columns() const;
};

bool operator==(const CliffordFrame& lhs, const CliffordFrame& rhs);

struct SymbolicPauliString {
    PauliString pauli;
    SymbolicBool sign;

    SymbolicPauliString() = default;
    explicit SymbolicPauliString(PauliString pauli);
    SymbolicPauliString(PauliString pauli, SymbolicBool sign);
};

bool operator==(const SymbolicPauliString& lhs, const SymbolicPauliString& rhs);

struct SymbolicPauliFrame {
    int nqubits = 0;
    std::vector<int> term_conditions;
    std::vector<std::uint64_t> x_term_blocks;
    std::vector<std::uint64_t> z_term_blocks;

    SymbolicPauliFrame() = default;
    explicit SymbolicPauliFrame(int nqubits);

    void add_pauli(const PauliString& pauli, int condition);
};

SymbolicPauliString conjugate_by(const SymbolicPauliFrame& frame, const PauliString& pauli);
SymbolicPauliString conjugate_by(const SymbolicPauliFrame& frame, const SymbolicPauliString& pauli);

} // namespace symft
