#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>

namespace symft::simd {

using Complex = std::complex<double>;

struct KernelTable {
    const char* name;
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
};

const KernelTable& scalar_table();
const KernelTable& dispatch_table();
const char* dispatch_name();

} // namespace symft::simd
