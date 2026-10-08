#pragma once

#include "sampler/active.hpp"

namespace symft::detail {

// Exact gauge: psi[b] = global_phase * i^parity(b & gauge) * r[b].
// No tolerance-based eligibility and no small-amplitude truncation.
bool real_rotation_compatible(const PrecomputedActivePauliRotationKernel& kernel, unsigned gauge);
bool real_measurement_compatible(const PrecomputedActivePauliMeasurementKernel& kernel, unsigned gauge);
unsigned real_measurement_next_gauge(const PrecomputedActivePauliMeasurementKernel& kernel, unsigned gauge);

struct RealRotationKernel {
    unsigned dim = 0;
    unsigned xmask = 0;
    unsigned pair_bit = 0;
    double c = 1;
    // One coefficient per SOURCE pair's left entry. The right coefficient
    // is its negative for the real, orthogonal, non-diagonal rotation.
    std::vector<double> coefficients;
    std::vector<double> small_pair_dest_coefficients;
    std::vector<unsigned> left;
    std::vector<unsigned> right;
};
struct RealMeasurementKernel {
    bool diagonal = false;
    unsigned pivot = 0, xmask = 0;
    unsigned next_gauge = 0;
    std::vector<unsigned> source0;
    std::vector<unsigned> source1;
    std::vector<double> coefficient;
    std::vector<double> diagonal_sign0;
    std::vector<double> diagonal_sign1;
};

RealRotationKernel make_real_rotation(const PrecomputedActivePauliRotationKernel& kernel, unsigned gauge);
RealMeasurementKernel make_real_measurement(const PrecomputedActivePauliMeasurementKernel& kernel, unsigned gauge);
void rotate_real_active(double* r, const RealRotationKernel& kernel, bool sign);
void promote_real_active(double* r, unsigned dim, unsigned gauge, double c, double q);
double real_measurement_probability(const double* r, const RealMeasurementKernel& kernel);
void project_real_active(const double* r, double* out, const RealMeasurementKernel& kernel,
                         bool branch, double invnorm);

} // namespace symft::detail
