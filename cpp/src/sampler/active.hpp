#pragma once

#include "core/pauli.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace symft {

struct ActiveState {
    int k = 0;
    std::vector<Complex> alpha;

    ActiveState();
    explicit ActiveState(int k);
    ActiveState(int k, std::vector<Complex> alpha);

    std::size_t dim() const;
    void reserve_for_k(int max_k);
};

struct ActivePauliAction {
    int nqubits = 0;
    std::uint64_t xmask = 0;
    std::uint64_t zmask = 0;
    Complex even_phase = Complex(1.0, 0.0);
    Complex odd_phase = Complex(-1.0, 0.0);
    bool xz_overlap_odd = false;

    ActivePauliAction() = default;
    explicit ActivePauliAction(const PauliString& pauli);
};

// Legacy name retained for API compatibility. This is a compact descriptor;
// basis-dependent coefficients are derived by the sampler instead of stored.
struct PrecomputedActivePauliRotationKernel {
    ActivePauliAction action;
    bool is_diagonal = true;
    bool uniform_imag_pairs = false;
    bool real_pair_flip = false;
    unsigned pair_bit = 0;
    std::size_t pair_count = 0;
    // Internal angle phi for exp(-i phi P). Frontend R_P(theta) passes phi = theta/2.
    double kernel_angle = 0.0;
    double cos_kernel_angle = 1.0;
    double sin_kernel_angle = 0.0;
    Complex minus_even_coefficient = Complex(0.0, 0.0);

    PrecomputedActivePauliRotationKernel() = default;
    PrecomputedActivePauliRotationKernel(const ActivePauliAction& action, double kernel_angle);
};

// Compact measurement mapping. Source indices and coefficients are derived
// from these masks during execution and therefore consume O(1) plan memory.
struct PrecomputedActivePauliMeasurementKernel {
    ActivePauliAction action;
    int pivot = 0;
    bool is_diagonal = true;
    int diagonal_phase_bit = 0;
    std::uint64_t z_without_pivot = 0;
    std::size_t out_dim = 0;
    Complex nondiagonal_coefficient1_even = Complex(0.0, 0.0);

    PrecomputedActivePauliMeasurementKernel() = default;
    explicit PrecomputedActivePauliMeasurementKernel(const PauliString& pauli);
    explicit PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action);
    PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action, int pivot);
};

static_assert(sizeof(PrecomputedActivePauliRotationKernel) <= 128);
static_assert(sizeof(PrecomputedActivePauliMeasurementKernel) <= 128);

void apply_pauli(std::vector<Complex>& out, const PauliString& pauli, const std::vector<Complex>& alpha);
void apply_pauli(ActiveState& state, const PauliString& pauli);
void rotate_pauli(ActiveState& state, const PauliString& pauli, double kernel_angle);
void rotate_pauli(ActiveState& state, const ActivePauliAction& action, double kernel_angle);
void rotate_pauli(ActiveState& state, const PrecomputedActivePauliRotationKernel& kernel, bool sign);
std::string active_simd_backend();

} // namespace symft
