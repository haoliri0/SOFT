#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace symft::batch_interleaved {

using Complex = std::complex<double>;

// Auxiliary kernels for the lower-level basis-major layout
// active[basis * leading_shots + shot]. Prepared batch sampling uses
// contiguous per-shot vectors and the shared simd::dispatch_table() instead.
struct KernelTable {
    void (*rotate_uniform_imag_pairs)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        double c,
        const double* q_by_shot);

    void (*rotate_uniform_imag_xmask)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        double c,
        const double* q_by_shot);

    void (*rotate_uniform_imag_xmask_const)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        double c,
        double q);

    void (*rotate_real_pair_flip)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        const double* basis_phase_signs,
        std::size_t npairs,
        double c,
        const double* q_by_shot);

    void (*rotate_real_pair_flip_xmask)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        const double* basis_phase_signs,
        std::size_t npairs,
        double c,
        const double* q_by_shot);

    void (*rotate_real_pair_flip_xmask_const)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        const double* basis_phase_signs,
        std::size_t npairs,
        double c,
        double q);

    void (*rotate_diagonal_const)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::size_t dim,
        const Complex* coeff,
        double c);

    void (*rotate_diagonal_mixed)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::size_t dim,
        const Complex* minus_coeff,
        const Complex* plus_coeff,
        const std::uint64_t* sign_bits,
        double c);

    void (*rotate_general_xmask_const)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        const Complex* left_coeff,
        const Complex* right_coeff,
        double c);

    void (*rotate_general_xmask_mixed)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        const Complex* left_minus_coeff,
        const Complex* right_minus_coeff,
        const Complex* left_plus_coeff,
        const Complex* right_plus_coeff,
        const std::uint64_t* sign_bits,
        double c);

    void (*promote_first_dormant_rotation)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        std::size_t dim,
        double c,
        const double* q_by_shot);

    void (*diagonal_measure_true_prob)(
        const double* re,
        const double* im,
        std::size_t leading_shots,
        int active_shots,
        const std::size_t* source_true,
        std::size_t out_dim,
        double* prob_true);

    void (*diagonal_project)(
        double* re,
        double* im,
        std::size_t leading_shots,
        int active_shots,
        const std::size_t* source_false,
        const std::size_t* source_true,
        std::size_t out_dim,
        const std::uint64_t* branch_bits,
        const double* invnorms);

    void (*nondiagonal_measure_true_prob)(
        const double* re,
        const double* im,
        std::size_t leading_shots,
        int active_shots,
        const std::size_t* source0_false,
        const std::size_t* source1_false,
        const double* coeff1_false_real,
        const double* coeff1_false_imag,
        std::size_t out_dim,
        double* prob_true);

    void (*nondiagonal_xmask_measure_true_prob)(
        const double* re,
        const double* im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        const double* coeff1_false_real,
        const double* coeff1_false_imag,
        double* prob_true);

    void (*nondiagonal_project)(
        const double* re,
        const double* im,
        double* scratch_re,
        double* scratch_im,
        std::size_t leading_shots,
        int active_shots,
        const std::size_t* source0_false,
        const std::size_t* source1_false,
        const double* coeff1_false_real,
        const double* coeff1_false_imag,
        std::size_t out_dim,
        const std::uint64_t* branch_bits,
        const double* invnorms);

    void (*nondiagonal_xmask_project)(
        const double* re,
        const double* im,
        double* scratch_re,
        double* scratch_im,
        std::size_t leading_shots,
        int active_shots,
        std::uint64_t xmask,
        unsigned pair_bit,
        std::size_t npairs,
        const double* coeff1_false_real,
        const double* coeff1_false_imag,
        const std::uint64_t* branch_bits,
        const double* invnorms);
};

const KernelTable& table();

} // namespace symft::batch_interleaved
