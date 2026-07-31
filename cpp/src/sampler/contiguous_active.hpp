#pragma once

#include "sampler/active.hpp"

#include <cstddef>

namespace symft::detail {

// Shared dense-vector kernels. Single-shot states and prepared batch sampler
// shots both use the same contiguous split-complex representation.
void rotate_contiguous_active(
    double* re,
    double* im,
    std::size_t dim,
    const PrecomputedActivePauliRotationKernel& kernel,
    bool sign);

void promote_contiguous_active(
    double* re,
    double* im,
    std::size_t dim,
    double c,
    double q);

double diagonal_probability_contiguous(
    const double* re,
    const double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch);

double nondiagonal_probability_contiguous(
    const double* re,
    const double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch);

void project_diagonal_contiguous(
    double* re,
    double* im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double invnorm);

void project_nondiagonal_contiguous(
    double* re,
    double* im,
    double* scratch_re,
    double* scratch_im,
    const PrecomputedActivePauliMeasurementKernel& kernel,
    bool branch,
    double invnorm);

} // namespace symft::detail
