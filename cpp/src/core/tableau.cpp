#include "core/tableau.hpp"

#include "core/frames.hpp"
#include "core/internal.hpp"

#include <algorithm>
#include <utility>

namespace symft::detail {

TableauCore::TableauCore(int nqubits, CoordinateIndexMode index_mode)
    : nqubits_(checked_nqubits(nqubits)),
      row_words_((2 * static_cast<std::size_t>(nqubits_) + 63) >> 6),
      index_mode_(index_mode),
      generators_(row_count(), PauliString(nqubits_)),
      x_coordinate_columns_(static_cast<std::size_t>(nqubits_) * row_words_, 0),
      z_coordinate_columns_(static_cast<std::size_t>(nqubits_) * row_words_, 0) {
    for (int q = 0; q < nqubits_; ++q) {
        const std::size_t x_row = static_cast<std::size_t>(q);
        const std::size_t z_row = static_cast<std::size_t>(nqubits_ + q);
        generators_[x_row].set_xbit(q);
        generators_[z_row].set_zbit(q);
        const std::size_t base = static_cast<std::size_t>(q) * row_words_;
        x_coordinate_columns_[base + (x_row >> 6)] |=
            std::uint64_t{1} << (x_row & 63);
        z_coordinate_columns_[base + (z_row >> 6)] |=
            std::uint64_t{1} << (z_row & 63);
    }
}

int TableauCore::nqubits() const {
    return nqubits_;
}

int TableauCore::xrow(int q) const {
    return check_qubit(nqubits_, q);
}

int TableauCore::zrow(int q) const {
    return nqubits_ + check_qubit(nqubits_, q);
}

std::size_t TableauCore::row_count() const {
    return 2 * static_cast<std::size_t>(nqubits_);
}

std::size_t TableauCore::row_words() const {
    return row_words_;
}

std::uint64_t TableauCore::body_generation() const {
    return body_generation_;
}

const PauliString& TableauCore::generator(std::size_t row) const {
    require_row(row);
    return generators_[row];
}

const std::vector<PauliString>& TableauCore::generators() const {
    return generators_;
}

void TableauCore::assign_generator(std::size_t row, PauliString generator) {
    require_row(row);
    if (generator.nqubits != nqubits_) {
        fail("Pauli tableau row has the wrong number of qubits");
    }
    if (index_mode_ == CoordinateIndexMode::Incremental) {
        assign_row_body(row, generators_[row], generator);
    } else {
        invalidate_coordinate_columns();
    }
    generators_[row] = std::move(generator);
    ++body_generation_;
}

void TableauCore::replace_generators(std::vector<PauliString> generators) {
    if (generators.size() != row_count()) {
        fail("Pauli tableau has the wrong number of rows");
    }
    for (const auto& generator : generators) {
        if (generator.nqubits != nqubits_) {
            fail("Pauli tableau row has the wrong number of qubits");
        }
    }
    generators_ = std::move(generators);
    ++body_generation_;
    invalidate_coordinate_columns();
    if (index_mode_ == CoordinateIndexMode::Incremental) {
        ensure_coordinate_columns();
    }
}

void TableauCore::phase_shift_generator(std::size_t row, int delta) {
    require_row(row);
    generators_[row].phase_shift(delta);
}

void TableauCore::swap_generators(std::size_t row_a, std::size_t row_b) {
    require_row(row_a);
    require_row(row_b);
    if (row_a == row_b) {
        return;
    }
    if (index_mode_ == CoordinateIndexMode::Incremental) {
        swap_row_columns(row_a, row_b);
    } else {
        invalidate_coordinate_columns();
    }
    std::swap(generators_[row_a], generators_[row_b]);
    ++body_generation_;
}

void TableauCore::multiply_selected_generator_bodies(
    const PauliString& factor,
    const std::vector<std::uint64_t>& selected_rows) {
    if (factor.nqubits != nqubits_ || selected_rows.size() != row_words_) {
        fail("invalid Pauli tableau row update");
    }
    ensure_coordinate_columns();
    for (std::size_t row_word = 0; row_word < selected_rows.size(); ++row_word) {
        std::uint64_t rows = selected_rows[row_word];
        while (rows) {
            const std::size_t row =
                row_word * 64 + static_cast<std::size_t>(trailing_zeros64(rows));
            require_row(row);
            auto& generator = generators_[row];
            int y_count = 0;
            for (std::size_t word = 0; word < generator.x.size(); ++word) {
                generator.x[word] ^= factor.x[word];
                generator.z[word] ^= factor.z[word];
                y_count += popcount64(generator.x[word] & generator.z[word]);
            }
            generator.set_phase(y_count);
            rows &= rows - 1;
        }
    }
    xor_selected_rows(factor, selected_rows);
    ++body_generation_;
}

void TableauCore::invalidate_coordinate_columns() const {
    coordinate_columns_valid_ = false;
}

void TableauCore::ensure_coordinate_columns() const {
    if (coordinate_columns_valid_) {
        return;
    }
    std::fill(x_coordinate_columns_.begin(), x_coordinate_columns_.end(), 0);
    std::fill(z_coordinate_columns_.begin(), z_coordinate_columns_.end(), 0);
    for (std::size_t row = 0; row < generators_.size(); ++row) {
        const PauliString& pauli = generators_[row];
        for (std::size_t word = 0; word < pauli.x.size(); ++word) {
            std::uint64_t x_bits = pauli.x[word];
            std::uint64_t z_bits = pauli.z[word];
            while (x_bits) {
                const std::size_t q =
                    word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
                const std::size_t base = q * row_words_;
                x_coordinate_columns_[base + (row >> 6)] |=
                    std::uint64_t{1} << (row & 63);
                x_bits &= x_bits - 1;
            }
            while (z_bits) {
                const std::size_t q =
                    word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
                const std::size_t base = q * row_words_;
                z_coordinate_columns_[base + (row >> 6)] |=
                    std::uint64_t{1} << (row & 63);
                z_bits &= z_bits - 1;
            }
        }
    }
    coordinate_columns_valid_ = true;
}

PauliString TableauCore::decompose_body(const PauliString& physical_pauli) const {
    std::vector<std::uint64_t> parity_scratch(row_words_, 0);
    return decompose_body(physical_pauli, parity_scratch);
}

std::span<const std::uint64_t> TableauCore::x_column(int q) const {
    const std::size_t base = static_cast<std::size_t>(check_qubit(nqubits_, q)) * row_words_;
    ensure_coordinate_columns();
    return std::span<const std::uint64_t>(x_coordinate_columns_.data() + base, row_words_);
}

std::span<const std::uint64_t> TableauCore::z_column(int q) const {
    const std::size_t base = static_cast<std::size_t>(check_qubit(nqubits_, q)) * row_words_;
    ensure_coordinate_columns();
    return std::span<const std::uint64_t>(z_coordinate_columns_.data() + base, row_words_);
}

void TableauCore::xor_selected_rows(
    const PauliString& factor,
    const std::vector<std::uint64_t>& selected_rows) {
    if (factor.nqubits != nqubits_ || selected_rows.size() != row_words_) {
        fail("invalid Pauli tableau row update");
    }
    ensure_coordinate_columns();
    const auto& packed_kernels = simd::dispatch_table();
    const bool use_simd = packed_kernels.packed_word_lanes > 1 &&
                          row_words_ >= packed_kernels.packed_word_lanes;
    const auto xor_column = [&](std::vector<std::uint64_t>& columns, std::size_t base) {
        if (use_simd) {
            packed_kernels.xor_packed_words(
                columns.data() + base,
                selected_rows.data(),
                row_words_);
            return;
        }
        for (std::size_t row_word = 0; row_word < row_words_; ++row_word) {
            columns[base + row_word] ^= selected_rows[row_word];
        }
    };
    for (std::size_t word = 0; word < factor.x.size(); ++word) {
        std::uint64_t x_bits = factor.x[word];
        std::uint64_t z_bits = factor.z[word];
        while (x_bits) {
            const std::size_t q =
                word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
            const std::size_t base = q * row_words_;
            xor_column(x_coordinate_columns_, base);
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const std::size_t q =
                word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
            const std::size_t base = q * row_words_;
            xor_column(z_coordinate_columns_, base);
            z_bits &= z_bits - 1;
        }
    }
}

void TableauCore::assign_row_body(
    std::size_t row,
    const PauliString& old_body,
    const PauliString& new_body) {
    require_row(row);
    if (old_body.nqubits != nqubits_ || new_body.nqubits != nqubits_) {
        fail("Pauli tableau row has the wrong number of qubits");
    }
    ensure_coordinate_columns();
    const std::size_t row_word = row >> 6;
    const std::uint64_t row_mask = std::uint64_t{1} << (row & 63);
    for (std::size_t word = 0; word < old_body.x.size(); ++word) {
        std::uint64_t x_bits = old_body.x[word] ^ new_body.x[word];
        std::uint64_t z_bits = old_body.z[word] ^ new_body.z[word];
        while (x_bits) {
            const std::size_t q =
                word * 64 + static_cast<std::size_t>(trailing_zeros64(x_bits));
            x_coordinate_columns_[q * row_words_ + row_word] ^= row_mask;
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const std::size_t q =
                word * 64 + static_cast<std::size_t>(trailing_zeros64(z_bits));
            z_coordinate_columns_[q * row_words_ + row_word] ^= row_mask;
            z_bits &= z_bits - 1;
        }
    }
}

void TableauCore::swap_row_columns(std::size_t row_a, std::size_t row_b) {
    require_row(row_a);
    require_row(row_b);
    ensure_coordinate_columns();
    if (row_a == row_b) {
        return;
    }
    const std::size_t word_a = row_a >> 6;
    const std::size_t word_b = row_b >> 6;
    const std::uint64_t mask_a = std::uint64_t{1} << (row_a & 63);
    const std::uint64_t mask_b = std::uint64_t{1} << (row_b & 63);
    const auto swap_bits = [&](std::vector<std::uint64_t>& columns, std::size_t base) {
        const bool bit_a = (columns[base + word_a] & mask_a) != 0;
        const bool bit_b = (columns[base + word_b] & mask_b) != 0;
        if (bit_a != bit_b) {
            columns[base + word_a] ^= mask_a;
            columns[base + word_b] ^= mask_b;
        }
    };
    for (int q = 0; q < nqubits_; ++q) {
        const std::size_t base = static_cast<std::size_t>(q) * row_words_;
        swap_bits(x_coordinate_columns_, base);
        swap_bits(z_coordinate_columns_, base);
    }
}

void TableauCore::require_row(std::size_t row) const {
    if (row >= row_count()) {
        fail("Pauli tableau row index out of range");
    }
}

} // namespace symft::detail

namespace symft {
using namespace detail;

namespace {

void swap_rows(CliffordFrame& frame, int a, int b) {
    if (a != b) {
        frame.tableau_core.swap_generators(
            static_cast<std::size_t>(a),
            static_cast<std::size_t>(b));
    }
}

void add_row_phase(CliffordFrame& frame, int row, int delta) {
    frame.tableau_core.phase_shift_generator(static_cast<std::size_t>(row), delta);
}

void mul_rows(CliffordFrame& frame, int dst, int lhs, int rhs, int extra_phase = 0) {
    PauliString out = frame.generator(lhs) * frame.generator(rhs);
    out.phase_shift(extra_phase);
    frame.tableau_core.assign_generator(
        static_cast<std::size_t>(dst),
        std::move(out));
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
    std::vector<PauliString> generators;
    generators.reserve(frame.tableau_core.row_count());
    for (const auto& generator : frame.generators()) {
        generators.push_back(preimage(gate, generator));
    }
    frame.tableau_core.replace_generators(std::move(generators));
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
        const auto& row = frame.generator(row_index);
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
    PauliString out = frame.tableau_core.decompose_body(pauli);
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

namespace symft {
using namespace detail;

namespace {

void multiply_into(PauliString& product, const PauliString& factor) {
    int carry = 0;
    for (std::size_t word = 0; word < product.x.size(); ++word) {
        carry += popcount64(product.z[word] & factor.x[word]);
        product.x[word] ^= factor.x[word];
        product.z[word] ^= factor.z[word];
    }
    product.set_phase(product.phase_exponent() + factor.phase_exponent() + 2 * (carry & 1));
}

} // namespace

PlanningTableau::PlanningTableau(int nqubits)
    : tableau_core_(nqubits, CoordinateIndexMode::Incremental),
      generator_signs_(tableau_core_.row_words(), 0),
      coordinate_parity_scratch_(tableau_core_.row_words(), 0),
      selected_rows_scratch_(tableau_core_.row_words(), 0),
      product_phase_low_scratch_(tableau_core_.row_words(), 0),
      product_phase_high_scratch_(tableau_core_.row_words(), 0) {}

int PlanningTableau::nqubits() const {
    return tableau_core_.nqubits();
}

PauliString PlanningTableau::stabilizer(int q) const {
    const int n = nqubits();
    return generator(static_cast<std::size_t>(n + check_qubit(n, q)));
}

PauliString PlanningTableau::destabilizer(int q) const {
    const int n = nqubits();
    return generator(static_cast<std::size_t>(check_qubit(n, q)));
}

bool PlanningTableau::generator_sign(std::size_t row) const {
    return (generator_signs_[row >> 6] & (std::uint64_t{1} << (row & 63))) != 0;
}

void PlanningTableau::set_generator_sign(std::size_t row, bool sign) {
    const std::uint64_t mask = std::uint64_t{1} << (row & 63);
    auto& word = generator_signs_[row >> 6];
    if (sign) {
        word |= mask;
    } else {
        word &= ~mask;
    }
}

PauliString PlanningTableau::generator(std::size_t row) const {
    PauliString result = tableau_core_.generator(row);
    if (generator_sign(row)) {
        result.phase_shift(2);
    }
    return result;
}

void PlanningTableau::assign_generator(
    std::size_t row,
    const PauliString& new_generator) {
    if (!pauli_squares_to_identity(new_generator)) {
        fail("planning tableau generators must be Hermitian");
    }
    set_generator_sign(row, measurement_phase_sign(new_generator));
    PauliString body = new_generator;
    body.set_phase(pauli_body_y_count(body));
    tableau_core_.assign_generator(
        row,
        std::move(body));
}

void PlanningTableau::update_selected_generator_signs(
    const PauliString& factor,
    const std::vector<std::uint64_t>& selected_rows) {
    const std::size_t row_words = generator_signs_.size();
    std::fill(product_phase_low_scratch_.begin(), product_phase_low_scratch_.end(), 0);
    std::fill(product_phase_high_scratch_.begin(), product_phase_high_scratch_.end(), 0);
    const auto& packed_kernels = simd::dispatch_table();
    const bool use_simd = packed_kernels.packed_word_lanes > 1 &&
                          row_words >= packed_kernels.packed_word_lanes;

    // For each selected row A and the fixed factor B, accumulate the local
    // phase in A_q B_q = i^e (A_q xor B_q), modulo four. Canonical tableau
    // updates only multiply commuting generators, so the low phase bit must
    // vanish and the high bit is the sign change. Each uint64_t handles 64
    // generator rows together.
    for (std::size_t word = 0; word < factor.x.size(); ++word) {
        std::uint64_t support = factor.x[word] | factor.z[word];
        while (support) {
            const int bit = trailing_zeros64(support);
            const std::size_t q = word * 64 + static_cast<std::size_t>(bit);
            const auto x_column = tableau_core_.x_column(static_cast<int>(q));
            const auto z_column = tableau_core_.z_column(static_cast<int>(q));
            const bool factor_x = (factor.x[word] & (std::uint64_t{1} << bit)) != 0;
            const bool factor_z = (factor.z[word] & (std::uint64_t{1} << bit)) != 0;
            const simd::PackedPauliAxis factor_axis = factor_x
                ? (factor_z ? simd::PackedPauliAxis::Y : simd::PackedPauliAxis::X)
                : simd::PackedPauliAxis::Z;
            if (use_simd) {
                packed_kernels.accumulate_tableau_phase(
                    product_phase_low_scratch_.data(),
                    product_phase_high_scratch_.data(),
                    x_column.data(),
                    z_column.data(),
                    selected_rows.data(),
                    row_words,
                    factor_axis);
                support &= support - 1;
                continue;
            }
            for (std::size_t row_word = 0; row_word < row_words; ++row_word) {
                const std::uint64_t row_x = x_column[row_word];
                const std::uint64_t row_z = z_column[row_word];
                std::uint64_t phase_low;
                std::uint64_t phase_high;
                if (factor_x && factor_z) {
                    phase_low = row_x ^ row_z;
                    phase_high = ~row_x & row_z;
                } else if (factor_x) {
                    phase_low = row_z;
                    phase_high = row_x & row_z;
                } else {
                    phase_low = row_x;
                    phase_high = row_x & ~row_z;
                }
                phase_low &= selected_rows[row_word];
                phase_high &= selected_rows[row_word];
                const std::uint64_t carry =
                    product_phase_low_scratch_[row_word] & phase_low;
                product_phase_low_scratch_[row_word] ^= phase_low;
                product_phase_high_scratch_[row_word] ^= phase_high ^ carry;
            }
            support &= support - 1;
        }
    }

    const bool factor_sign = measurement_phase_sign(factor);
    for (std::size_t row_word = 0; row_word < row_words; ++row_word) {
        if ((product_phase_low_scratch_[row_word] & selected_rows[row_word]) != 0) {
            fail("planning tableau update multiplied anticommuting generators");
        }
        std::uint64_t sign_changes = product_phase_high_scratch_[row_word];
        if (factor_sign) {
            sign_changes ^= selected_rows[row_word];
        }
        generator_signs_[row_word] ^= sign_changes & selected_rows[row_word];
    }
}

void PlanningTableau::swap_generator_signs(std::size_t row_a, std::size_t row_b) {
    if (row_a == row_b) {
        return;
    }
    const bool sign_a = generator_sign(row_a);
    const bool sign_b = generator_sign(row_b);
    set_generator_sign(row_a, sign_b);
    set_generator_sign(row_b, sign_a);
}

void PlanningTableau::swap_generator_pairs(int q_a, int q_b) {
    if (q_a == q_b) {
        return;
    }
    const int n = nqubits();
    tableau_core_.swap_generators(
        static_cast<std::size_t>(q_a),
        static_cast<std::size_t>(q_b));
    tableau_core_.swap_generators(
        static_cast<std::size_t>(n + q_a),
        static_cast<std::size_t>(n + q_b));
    swap_generator_signs(static_cast<std::size_t>(q_a), static_cast<std::size_t>(q_b));
    swap_generator_signs(
        static_cast<std::size_t>(n + q_a),
        static_cast<std::size_t>(n + q_b));
}

PauliString PlanningTableau::reconstruct(const PauliString& coordinates) const {
    const int total_qubits = nqubits();
    if (coordinates.nqubits != total_qubits) {
        fail("Pauli coordinates and planning tableau have different numbers of qubits");
    }
    if (tableau_core_.body_generation() == 0) {
        return coordinates;
    }
    PauliString physical(total_qubits);
    physical.set_phase(coordinates.phase_exponent());
    int sign_parity = 0;
    const std::size_t n = static_cast<std::size_t>(total_qubits);
    const std::size_t shift_words = n >> 6;
    const unsigned shift = static_cast<unsigned>(n & 63);
    for (std::size_t word = 0; word < coordinates.x.size(); ++word) {
        const std::uint64_t destabilizer_signs = generator_signs_[word];
        std::uint64_t stabilizer_signs = generator_signs_[shift_words + word] >> shift;
        if (shift != 0 && shift_words + word + 1 < generator_signs_.size()) {
            stabilizer_signs |= generator_signs_[shift_words + word + 1] << (64 - shift);
        }
        sign_parity ^= popcount64(coordinates.x[word] & destabilizer_signs) & 1;
        sign_parity ^= popcount64(coordinates.z[word] & stabilizer_signs) & 1;
        std::uint64_t bits = coordinates.x[word] | coordinates.z[word];
        while (bits) {
            const int bit = trailing_zeros64(bits);
            const std::size_t q = word * 64 + static_cast<std::size_t>(bit);
            const std::uint64_t mask = std::uint64_t{1} << bit;
            if ((coordinates.x[word] & mask) != 0) {
                multiply_into(
                    physical,
                    tableau_core_.generator(static_cast<std::size_t>(q)));
            }
            if ((coordinates.z[word] & mask) != 0) {
                multiply_into(
                    physical,
                    tableau_core_.generator(static_cast<std::size_t>(total_qubits) + q));
            }
            bits &= bits - 1;
        }
    }
    physical.phase_shift(2 * sign_parity);
    return physical;
}

PauliString PlanningTableau::decompose(const PauliString& physical_pauli) const {
    if (physical_pauli.nqubits != nqubits()) {
        fail("Pauli string and planning tableau have different numbers of qubits");
    }
    if (tableau_core_.body_generation() == 0) {
        return physical_pauli;
    }
    PauliString coordinates =
        tableau_core_.decompose_body(physical_pauli, coordinate_parity_scratch_);
    const PauliString reconstructed = reconstruct(coordinates);
    if (!reconstructed.same_body(physical_pauli)) {
        fail("planning tableau generators do not span the Pauli body");
    }
    coordinates.set_phase(physical_pauli.phase_exponent() - reconstructed.phase_exponent());
    return coordinates;
}

PauliString PlanningTableau::positive_physical_body(const PauliString& coordinates) const {
    if (!pauli_squares_to_identity(coordinates)) {
        fail("planning tableau update requires a Hermitian Pauli");
    }
    PauliString positive_coordinates = coordinates;
    positive_coordinates.set_phase(pauli_body_y_count(positive_coordinates));
    return reconstruct(positive_coordinates);
}

void PlanningTableau::multiply_nonpivot_generators(
    const PauliString& coordinates,
    int pivot,
    const PauliString& pivot_generator) {
    std::fill(selected_rows_scratch_.begin(), selected_rows_scratch_.end(), 0);
    const auto select_row = [&](std::size_t row) {
        selected_rows_scratch_[row >> 6] |= std::uint64_t{1} << (row & 63);
    };
    const int n = nqubits();
    for (int q = 0; q < n; ++q) {
        if (q == pivot) {
            continue;
        }
        if (coordinates.xbit(q)) {
            select_row(static_cast<std::size_t>(n + q));
        }
        if (coordinates.zbit(q)) {
            select_row(static_cast<std::size_t>(q));
        }
    }
    update_selected_generator_signs(pivot_generator, selected_rows_scratch_);
    tableau_core_.multiply_selected_generator_bodies(
        pivot_generator,
        selected_rows_scratch_);
}

void PlanningTableau::promote_dormant_rotation(
    const PauliString& coordinates,
    int active_count,
    int picked_dormant) {
    const int k = checked_nqubits(active_count);
    const int n = nqubits();
    if (k >= n || picked_dormant < 0 || picked_dormant >= n - k) {
        fail("dormant rotation promotion requires a dormant coordinate");
    }
    const int pivot = k + picked_dormant;
    if (!coordinates.xbit(pivot)) {
        fail("dormant rotation pivot must anticommute with the promoted Pauli");
    }

    const PauliString promoted_destabilizer = positive_physical_body(coordinates);
    const PauliString pivot_stabilizer = stabilizer(pivot);
    multiply_nonpivot_generators(coordinates, pivot, pivot_stabilizer);
    assign_generator(
        static_cast<std::size_t>(pivot),
        promoted_destabilizer);

    if (pivot != k) {
        swap_generator_pairs(pivot, k);
    }
}

PauliString PlanningTableau::replace_dormant_measurement(
    const PauliString& coordinates,
    int active_count,
    int picked_dormant) {
    const int k = checked_nqubits(active_count);
    const int n = nqubits();
    if (k > n || picked_dormant < 0 || picked_dormant >= n - k) {
        fail("dormant measurement requires a dormant coordinate");
    }
    const int pivot = k + picked_dormant;
    if (!coordinates.xbit(pivot)) {
        fail("dormant measurement pivot must anticommute with the measured Pauli");
    }

    const PauliString new_stabilizer = positive_physical_body(coordinates);
    const PauliString old_stabilizer = stabilizer(pivot);
    multiply_nonpivot_generators(coordinates, pivot, old_stabilizer);
    assign_generator(
        static_cast<std::size_t>(n + pivot),
        new_stabilizer);
    assign_generator(
        static_cast<std::size_t>(pivot),
        old_stabilizer);
    return destabilizer(pivot);
}

PauliString PlanningTableau::remove_active_measurement(
    const PauliString& coordinates,
    int active_count,
    int pivot,
    bool diagonal) {
    const int k = checked_nqubits(active_count);
    const int n = nqubits();
    if (k <= 0 || k > n || pivot < 0 || pivot >= k) {
        fail("active measurement pivot is outside the active block");
    }
    if ((diagonal && (!coordinates.zbit(pivot) || coordinates.xbit(pivot))) ||
        (!diagonal && !coordinates.xbit(pivot))) {
        fail("active measurement pivot does not match the Pauli coordinates");
    }

    const PauliString new_stabilizer = positive_physical_body(coordinates);
    const PauliString old_stabilizer = stabilizer(pivot);
    const PauliString old_destabilizer = destabilizer(pivot);
    const PauliString pivot_conjugate = diagonal
        ? old_destabilizer
        : old_stabilizer;
    multiply_nonpivot_generators(coordinates, pivot, pivot_conjugate);
    assign_generator(
        static_cast<std::size_t>(n + pivot),
        new_stabilizer);
    assign_generator(
        static_cast<std::size_t>(pivot),
        pivot_conjugate);

    for (int q = pivot; q + 1 < k; ++q) {
        swap_generator_pairs(q, q + 1);
    }
    return destabilizer(k - 1);
}

} // namespace symft
