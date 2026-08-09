#pragma once

#include "core/pauli.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace symft::detail {

enum class CoordinateIndexMode {
    Lazy,
    Incremental,
};

// Generator storage shared by tableau-like Pauli bases. Rows are ordered as
// X/destabilizer generators followed by Z/stabilizer generators. The complete
// PauliString phase is retained in each row, while coordinate indexing and
// incremental updates deliberately depend only on the symplectic X/Z bodies.
class TableauCore {
  public:
    TableauCore() = default;
    explicit TableauCore(
        int nqubits,
        CoordinateIndexMode index_mode = CoordinateIndexMode::Lazy);

    int nqubits() const;
    int xrow(int q) const;
    int zrow(int q) const;
    std::size_t row_count() const;
    std::size_t row_words() const;
    std::uint64_t body_generation() const;

    const PauliString& generator(std::size_t row) const;
    const std::vector<PauliString>& generators() const;
    void assign_generator(std::size_t row, PauliString generator);
    void replace_generators(std::vector<PauliString> generators);
    void phase_shift_generator(std::size_t row, int delta);
    void swap_generators(std::size_t row_a, std::size_t row_b);
    void multiply_selected_generator_bodies(
        const PauliString& factor,
        const std::vector<std::uint64_t>& selected_rows);

    // Returns only the coordinate X/Z body. The owning wrapper reconstructs
    // the operator to determine its phase according to that wrapper's policy.
    PauliString decompose_body(const PauliString& physical_pauli) const;
    PauliString decompose_body(
        const PauliString& physical_pauli,
        std::vector<std::uint64_t>& parity_scratch) const;

    std::span<const std::uint64_t> x_column(int q) const;
    std::span<const std::uint64_t> z_column(int q) const;

  private:
    int nqubits_ = 0;
    std::size_t row_words_ = 0;
    CoordinateIndexMode index_mode_ = CoordinateIndexMode::Lazy;
    std::uint64_t body_generation_ = 0;
    std::vector<PauliString> generators_;
    mutable std::vector<std::uint64_t> x_coordinate_columns_;
    mutable std::vector<std::uint64_t> z_coordinate_columns_;
    mutable bool coordinate_columns_valid_ = true;

    void invalidate_coordinate_columns() const;
    void ensure_coordinate_columns() const;
    void require_row(std::size_t row) const;
    void assign_row_body(
        std::size_t row,
        const PauliString& old_body,
        const PauliString& new_body);
    void xor_selected_rows(
        const PauliString& factor,
        const std::vector<std::uint64_t>& selected_rows);
    void swap_row_columns(std::size_t row_a, std::size_t row_b);
};

inline PauliString TableauCore::decompose_body(
    const PauliString& physical_pauli,
    std::vector<std::uint64_t>& parity_scratch) const {
    if (physical_pauli.nqubits != nqubits_) {
        throw Error("Pauli string and tableau have different numbers of qubits");
    }
    ensure_coordinate_columns();
    if (parity_scratch.size() != row_words_) {
        parity_scratch.resize(row_words_);
    }
    std::fill(parity_scratch.begin(), parity_scratch.end(), 0);
    for (std::size_t word = 0; word < physical_pauli.x.size(); ++word) {
        std::uint64_t x_bits = physical_pauli.x[word];
        std::uint64_t z_bits = physical_pauli.z[word];
        while (x_bits) {
            const std::size_t q =
                word * 64 + static_cast<std::size_t>(std::countr_zero(x_bits));
            const std::size_t base = q * row_words_;
            for (std::size_t row_word = 0; row_word < row_words_; ++row_word) {
                parity_scratch[row_word] ^= z_coordinate_columns_[base + row_word];
            }
            x_bits &= x_bits - 1;
        }
        while (z_bits) {
            const std::size_t q =
                word * 64 + static_cast<std::size_t>(std::countr_zero(z_bits));
            const std::size_t base = q * row_words_;
            for (std::size_t row_word = 0; row_word < row_words_; ++row_word) {
                parity_scratch[row_word] ^= x_coordinate_columns_[base + row_word];
            }
            z_bits &= z_bits - 1;
        }
    }

    PauliString coordinates(nqubits_);
    const std::size_t n = static_cast<std::size_t>(nqubits_);
    const std::size_t shift_words = n >> 6;
    const unsigned shift = static_cast<unsigned>(n & 63);
    for (std::size_t word = 0; word < coordinates.z.size(); ++word) {
        coordinates.z[word] = parity_scratch[word];
        if (word + 1 == coordinates.z.size() && shift != 0) {
            coordinates.z[word] &= (std::uint64_t{1} << shift) - 1;
        }
        std::uint64_t value = parity_scratch[shift_words + word] >> shift;
        if (shift != 0 && shift_words + word + 1 < parity_scratch.size()) {
            value |= parity_scratch[shift_words + word + 1] << (64 - shift);
        }
        coordinates.x[word] = value;
    }
    return coordinates;
}

} // namespace symft::detail

namespace symft {

struct CliffordFrame;

// Tableau operations on the Clifford frame. The frame itself, including its
// storage and caches, is defined in core/frames.hpp.
PauliString preimage(const CliffordFrame& frame, const PauliString& pauli);
PauliString coordinates_in_frame(const CliffordFrame& frame, const PauliString& pauli);

void left_H(CliffordFrame& frame, int q);
void left_H_NXY(CliffordFrame& frame, int q);
void left_H_NXZ(CliffordFrame& frame, int q);
void left_H_NYZ(CliffordFrame& frame, int q);
void left_H_XY(CliffordFrame& frame, int q);
void left_H_YZ(CliffordFrame& frame, int q);
void left_C_NXYZ(CliffordFrame& frame, int q);
void left_C_NZYX(CliffordFrame& frame, int q);
void left_C_XNYZ(CliffordFrame& frame, int q);
void left_C_XYNZ(CliffordFrame& frame, int q);
void left_C_XYZ(CliffordFrame& frame, int q);
void left_C_ZNYX(CliffordFrame& frame, int q);
void left_C_ZYNX(CliffordFrame& frame, int q);
void left_C_ZYX(CliffordFrame& frame, int q);
void left_S(CliffordFrame& frame, int q);
void left_SDG(CliffordFrame& frame, int q);
void left_SQRT_X(CliffordFrame& frame, int q);
void left_SQRT_X_DAG(CliffordFrame& frame, int q);
void left_SQRT_Y(CliffordFrame& frame, int q);
void left_SQRT_Y_DAG(CliffordFrame& frame, int q);
void left_X(CliffordFrame& frame, int q);
void left_Y(CliffordFrame& frame, int q);
void left_Z(CliffordFrame& frame, int q);
void left_CX(CliffordFrame& frame, int control, int target);
void left_CY(CliffordFrame& frame, int control, int target);
void left_CZ(CliffordFrame& frame, int a, int b);
void left_SWAP(CliffordFrame& frame, int a, int b);
void left_CXSWAP(CliffordFrame& frame, int a, int b);
void left_CZSWAP(CliffordFrame& frame, int a, int b);
void left_ISWAP(CliffordFrame& frame, int a, int b);
void left_ISWAP_DAG(CliffordFrame& frame, int a, int b);
void left_SQRT_XX(CliffordFrame& frame, int a, int b);
void left_SQRT_XX_DAG(CliffordFrame& frame, int a, int b);
void left_SQRT_YY(CliffordFrame& frame, int a, int b);
void left_SQRT_YY_DAG(CliffordFrame& frame, int a, int b);
void left_SQRT_ZZ(CliffordFrame& frame, int a, int b);
void left_SQRT_ZZ_DAG(CliffordFrame& frame, int a, int b);
void left_SWAPCX(CliffordFrame& frame, int a, int b);
void left_XCX(CliffordFrame& frame, int control, int target);
void left_XCY(CliffordFrame& frame, int control, int target);
void left_XCZ(CliffordFrame& frame, int control, int target);
void left_YCX(CliffordFrame& frame, int control, int target);
void left_YCY(CliffordFrame& frame, int control, int target);
void left_YCZ(CliffordFrame& frame, int control, int target);
void right_H(CliffordFrame& frame, int q);
void right_S(CliffordFrame& frame, int q);
void right_SDG(CliffordFrame& frame, int q);
void right_X(CliffordFrame& frame, int q);
void right_Z(CliffordFrame& frame, int q);
void right_CX(CliffordFrame& frame, int control, int target);
void right_CZ(CliffordFrame& frame, int a, int b);
void right_SWAP(CliffordFrame& frame, int a, int b);

// The physical stabilizer--destabilizer basis followed by the planning pass.
// Coordinate X bits multiply destabilizers and coordinate Z bits multiply
// stabilizers. Pending operations remain physical Pauli strings and are
// decomposed against this tableau only when the planner reaches them.
class PlanningTableau {
  public:
    PlanningTableau() = default;
    explicit PlanningTableau(int nqubits);

    int nqubits() const;
    PauliString stabilizer(int q) const;
    PauliString destabilizer(int q) const;

    PauliString decompose(const PauliString& physical_pauli) const;
    PauliString reconstruct(const PauliString& coordinates) const;

    void promote_dormant_rotation(
        const PauliString& coordinates,
        int active_count,
        int picked_dormant);

    PauliString replace_dormant_measurement(
        const PauliString& coordinates,
        int active_count,
        int picked_dormant);

    PauliString remove_active_measurement(
        const PauliString& coordinates,
        int active_count,
        int pivot,
        bool diagonal);

  private:
    // Generator rows are canonical Hermitian PauliStrings in the shared
    // tableau. Their real signs are stored separately in packed machine words.
    detail::TableauCore tableau_core_;
    std::vector<std::uint64_t> generator_signs_;
    mutable std::vector<std::uint64_t> coordinate_parity_scratch_;
    std::vector<std::uint64_t> selected_rows_scratch_;
    std::vector<std::uint64_t> product_phase_low_scratch_;
    std::vector<std::uint64_t> product_phase_high_scratch_;

    bool generator_sign(std::size_t row) const;
    void set_generator_sign(std::size_t row, bool sign);
    PauliString generator(std::size_t row) const;
    void assign_generator(std::size_t row, const PauliString& generator);
    void update_selected_generator_signs(
        const PauliString& factor,
        const std::vector<std::uint64_t>& selected_rows);
    void swap_generator_signs(std::size_t row_a, std::size_t row_b);
    void swap_generator_pairs(int q_a, int q_b);
    PauliString positive_physical_body(const PauliString& coordinates) const;
    void multiply_nonpivot_generators(
        const PauliString& coordinates,
        int pivot,
        const PauliString& pivot_generator);
};

} // namespace symft
