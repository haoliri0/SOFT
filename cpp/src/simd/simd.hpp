#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace symft::simd {

using Complex = std::complex<double>;

// A Clifford map on one or two transposed Pauli columns. Input and output
// planes are ordered x0, z0, x1, z1. Pauli bodies are linear over F_2; the
// two phase-delta bits are stored in algebraic normal form over those planes.
struct PackedCliffordTransform {
    std::uint8_t arity = 0;
    std::array<std::uint8_t, 4> output_masks{};
    std::uint16_t phase_low_anf = 0;
    std::uint16_t phase_high_anf = 0;
};

enum class PackedPauliAxis : std::uint8_t {
    X = 1,
    Z = 2,
    Y = 3,
};

struct KernelTable {
    const char* name;
    std::size_t packed_word_lanes;
    void (*mul_assign)(Complex* alpha, const Complex* coeff, double c, std::size_t n);
    double (*norm_sum)(const Complex* alpha, const std::size_t* indices, std::size_t n);
    void (*mul_assign_soa)(double* re, double* im, const Complex* coeff, double c, std::size_t n);
    double (*norm_sum_soa)(const double* re, const double* im, const std::size_t* indices, std::size_t n);
    double (*measure_nondiagonal_probability_soa)(
        const double* re,
        const double* im,
        std::size_t dim,
        std::uint64_t xmask,
        std::uint64_t zmask,
        unsigned pivot,
        Complex coefficient1_even,
        bool branch);
    void (*project_nondiagonal_soa)(
        const double* re,
        const double* im,
        double* out_re,
        double* out_im,
        std::size_t dim,
        std::uint64_t xmask,
        std::uint64_t zmask,
        unsigned pivot,
        Complex coefficient1_even,
        bool branch,
        double invnorm);
    void (*rotate_uniform_imag_pairs_soa)(
        double* re,
        double* im,
        std::size_t dim,
        std::uint64_t xmask,
        unsigned pair_bit,
        double c,
        double q);
    void (*rotate_real_pair_flip_soa)(
        double* re,
        double* im,
        std::size_t dim,
        std::uint64_t xmask,
        unsigned pair_bit,
        const double* phase_signs,
        double c,
        double base_coeff);
    void (*rotate_general_pairs_soa)(
        double* re,
        double* im,
        std::size_t dim,
        std::uint64_t xmask,
        unsigned pair_bit,
        const Complex* left_coeff,
        const Complex* right_coeff,
        double c);
    void (*xor_packed_words)(
        std::uint64_t* destination,
        const std::uint64_t* source,
        std::size_t n);
    void (*xor_packed_columns)(
        std::uint64_t* destination,
        const std::uint64_t* columns,
        std::size_t column_stride,
        const std::uint32_t* column_indices,
        std::size_t column_count,
        std::size_t word_offset,
        std::size_t n);
    void (*apply_packed_clifford)(
        std::uint64_t* x0,
        std::uint64_t* z0,
        std::uint64_t* x1,
        std::uint64_t* z1,
        std::uint64_t* phase_low,
        std::uint64_t* phase_high,
        std::size_t n,
        const PackedCliffordTransform& transform);
    void (*accumulate_tableau_phase)(
        std::uint64_t* phase_low,
        std::uint64_t* phase_high,
        const std::uint64_t* row_x,
        const std::uint64_t* row_z,
        const std::uint64_t* selected,
        std::size_t n,
        PackedPauliAxis factor_axis);
};

const KernelTable& scalar_table();
const KernelTable& dispatch_table();
const char* dispatch_name();

} // namespace symft::simd
