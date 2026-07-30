#pragma once

#include "core/common.hpp"

#include <iosfwd>
#include <string>

namespace symft {

struct PauliString {
    int nqubits = 0;
    std::vector<std::uint64_t> x;
    std::vector<std::uint64_t> z;
    std::uint8_t phase = 0;

    PauliString() = default;
    explicit PauliString(int nqubits);

    bool xbit(int q) const;
    bool zbit(int q) const;
    void set_xbit(int q, bool value = true);
    void set_zbit(int q, bool value = true);
    int phase_exponent() const;
    void set_phase(int phase_exponent);
    void phase_shift(int delta);
    bool has_nonidentity_body() const;
    bool same_body(const PauliString& other) const;
    std::string str() const;
};

bool operator==(const PauliString& lhs, const PauliString& rhs);
bool operator!=(const PauliString& lhs, const PauliString& rhs);
PauliString operator*(const PauliString& lhs, const PauliString& rhs);
std::ostream& operator<<(std::ostream& out, const PauliString& pauli);

PauliString pauli_identity(int nqubits);
PauliString pauli_x(int nqubits, int q);
PauliString pauli_y(int nqubits, int q);
PauliString pauli_z(int nqubits, int q);
PauliString pauli_string(const std::string& ops);
PauliString neg(PauliString pauli);
bool pauli_anticommutes(const PauliString& a, const PauliString& b);
bool pauli_squares_to_identity(const PauliString& pauli);
int pauli_body_y_count(const PauliString& pauli);
bool measurement_phase_sign(const PauliString& pauli);

} // namespace symft
