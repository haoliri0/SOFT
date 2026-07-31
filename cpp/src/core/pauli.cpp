#include "core/pauli.hpp"

#include "core/internal.hpp"

#include <cctype>
#include <ostream>

namespace symft {
using namespace detail;

PauliString::PauliString(int nqubits_) : nqubits(checked_nqubits(nqubits_)) {
    const int nw = nwords_for(nqubits);
    x.assign(nw, 0);
    z.assign(nw, 0);
}

bool PauliString::xbit(int q) const {
    check_qubit(nqubits, q);
    return (x[word_index(q)] & bit_mask(q)) != 0;
}

bool PauliString::zbit(int q) const {
    check_qubit(nqubits, q);
    return (z[word_index(q)] & bit_mask(q)) != 0;
}

void PauliString::set_xbit(int q, bool value) {
    check_qubit(nqubits, q);
    auto& word = x[word_index(q)];
    if (value) {
        word |= bit_mask(q);
    } else {
        word &= ~bit_mask(q);
    }
}

void PauliString::set_zbit(int q, bool value) {
    check_qubit(nqubits, q);
    auto& word = z[word_index(q)];
    if (value) {
        word |= bit_mask(q);
    } else {
        word &= ~bit_mask(q);
    }
}

int PauliString::phase_exponent() const {
    return int(phase & 3u);
}

void PauliString::set_phase(int phase_exponent_) {
    phase = static_cast<std::uint8_t>(phase_exponent_ & 3);
}

void PauliString::phase_shift(int delta) {
    set_phase(phase_exponent() + delta);
}

bool PauliString::has_nonidentity_body() const {
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i] != 0 || z[i] != 0) {
            return true;
        }
    }
    return false;
}

bool PauliString::same_body(const PauliString& other) const {
    return nqubits == other.nqubits && x == other.x && z == other.z;
}

std::string PauliString::str() const {
    int ny = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        ny += popcount64(x[i] & z[i]);
    }
    const int coeff_phase = (phase_exponent() - ny) & 3;
    std::string out;
    if (coeff_phase == 1) {
        out = "i*";
    } else if (coeff_phase == 2) {
        out = "-";
    } else if (coeff_phase == 3) {
        out = "-i*";
    }
    if (nqubits == 0) {
        out += "I";
        return out;
    }
    for (int q = 0; q < nqubits; ++q) {
        const bool xb = xbit(q);
        const bool zb = zbit(q);
        out.push_back(xb ? (zb ? 'Y' : 'X') : (zb ? 'Z' : 'I'));
    }
    return out;
}

bool operator==(const PauliString& lhs, const PauliString& rhs) {
    return lhs.nqubits == rhs.nqubits && lhs.phase == rhs.phase && lhs.x == rhs.x && lhs.z == rhs.z;
}

bool operator!=(const PauliString& lhs, const PauliString& rhs) {
    return !(lhs == rhs);
}

PauliString operator*(const PauliString& lhs, const PauliString& rhs) {
    check_same_nqubits(lhs, rhs);
    PauliString out(lhs.nqubits);
    int carry = 0;
    for (std::size_t i = 0; i < lhs.x.size(); ++i) {
        carry += popcount64(lhs.z[i] & rhs.x[i]);
    }
    out.set_phase(lhs.phase_exponent() + rhs.phase_exponent() + 2 * (carry & 1));
    for (std::size_t i = 0; i < lhs.x.size(); ++i) {
        out.x[i] = lhs.x[i] ^ rhs.x[i];
        out.z[i] = lhs.z[i] ^ rhs.z[i];
    }
    return out;
}

std::ostream& operator<<(std::ostream& out, const PauliString& pauli) {
    out << pauli.str();
    return out;
}

PauliString pauli_identity(int nqubits) {
    return PauliString(nqubits);
}

PauliString pauli_x(int nqubits, int q) {
    PauliString out(nqubits);
    out.set_xbit(q);
    return out;
}

PauliString pauli_y(int nqubits, int q) {
    PauliString out(nqubits);
    out.set_xbit(q);
    out.set_zbit(q);
    out.set_phase(1);
    return out;
}

PauliString pauli_z(int nqubits, int q) {
    PauliString out(nqubits);
    out.set_zbit(q);
    return out;
}

PauliString pauli_string(const std::string& ops) {
    PauliString out(static_cast<int>(ops.size()));
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const char ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ops[i])));
        const int q = static_cast<int>(i);
        if (ch == 'I' || ch == '_') {
            continue;
        }
        if (ch == 'X') {
            out.set_xbit(q);
        } else if (ch == 'Z') {
            out.set_zbit(q);
        } else if (ch == 'Y') {
            out.set_xbit(q);
            out.set_zbit(q);
            out.phase_shift(1);
        } else {
            fail("unsupported Pauli character");
        }
    }
    return out;
}

PauliString neg(PauliString pauli) {
    pauli.phase_shift(2);
    return pauli;
}

bool pauli_anticommutes(const PauliString& a, const PauliString& b) {
    check_same_nqubits(a, b);
    bool parity = false;
    for (std::size_t i = 0; i < a.x.size(); ++i) {
        parity ^= is_odd_popcount(a.x[i] & b.z[i]);
        parity ^= is_odd_popcount(a.z[i] & b.x[i]);
    }
    return parity;
}

int pauli_body_y_count(const PauliString& pauli) {
    int count = 0;
    for (std::size_t i = 0; i < pauli.x.size(); ++i) {
        count += popcount64(pauli.x[i] & pauli.z[i]);
    }
    return count;
}

bool pauli_squares_to_identity(const PauliString& pauli) {
    return ((pauli.phase_exponent() - pauli_body_y_count(pauli)) & 1) == 0;
}

bool measurement_phase_sign(const PauliString& pauli) {
    const int coeff_phase = (pauli.phase_exponent() - pauli_body_y_count(pauli)) & 3;
    if (coeff_phase == 0) {
        return false;
    }
    if (coeff_phase == 2) {
        return true;
    }
    fail("Pauli measurement requires a Hermitian Pauli with real coefficient");
}

} // namespace symft
