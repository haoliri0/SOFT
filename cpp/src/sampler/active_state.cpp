#include "active_internal.hpp"

#include "simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace symft {
using namespace detail;

ActiveState::ActiveState() : k(0), alpha{Complex(1.0, 0.0)} {}

ActiveState::ActiveState(int k_) : k(checked_nqubits(k_)), alpha(active_length(k), Complex(0.0, 0.0)) {
    alpha[0] = Complex(1.0, 0.0);
}

ActiveState::ActiveState(int k_, std::vector<Complex> alpha_) : k(checked_nqubits(k_)), alpha(std::move(alpha_)) {
    if (alpha.size() != active_length(k)) {
        fail("active vector length does not match active qubit count");
    }
}

std::size_t ActiveState::dim() const {
    return active_length(k);
}

void ActiveState::reserve_for_k(int max_k) {
    const std::size_t max_dim = active_length(max_k);
    if (alpha.size() < max_dim) {
        alpha.resize(max_dim, Complex(0.0, 0.0));
    }
}

ActivePauliAction::ActivePauliAction(const PauliString& pauli) : nqubits(pauli.nqubits) {
    if (nqubits >= 63) {
        fail("Pauli string has too many qubits for active-state basis indexing");
    }
    if (!pauli_squares_to_identity(pauli)) {
        fail("Pauli rotation requires P^2 == I");
    }
    xmask = active_mask_x(pauli);
    zmask = active_mask_z(pauli);
    even_phase = phase_factor(pauli.phase_exponent());
    odd_phase = -even_phase;
    xz_overlap_odd = is_odd_popcount(xmask & zmask);
}

PrecomputedActivePauliRotationKernel::PrecomputedActivePauliRotationKernel(
    const ActivePauliAction& action_,
    double kernel_angle_)
    : action(action_),
      is_diagonal(action.xmask == 0),
      kernel_angle(kernel_angle_),
      cos_kernel_angle(std::cos(kernel_angle_)),
      sin_kernel_angle(std::sin(kernel_angle_)),
      minus_even_coefficient(Complex(0.0, -sin_kernel_angle) * action.even_phase) {
    const std::size_t dim = active_length(action.nqubits);
    if (is_diagonal) {
        return;
    }
    uniform_imag_pairs = action.zmask == 0;
    real_pair_flip = can_rotate_real_pair_flip(action);
    pair_bit = static_cast<unsigned>(highest_set_bit64(action.xmask));
    pair_count = dim >> 1;
}

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(const PauliString& pauli)
    : PrecomputedActivePauliMeasurementKernel(ActivePauliAction(pauli)) {}

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(const ActivePauliAction& action_) {
    const int selected_pivot =
        action_.xmask != 0
            ? highest_set_bit64(action_.xmask)
            : (action_.zmask != 0 ? highest_set_bit64(action_.zmask) : -1);
    if (selected_pivot < 0) {
        fail("cannot build an active measurement kernel for identity Pauli");
    }
    *this = PrecomputedActivePauliMeasurementKernel(action_, selected_pivot);
}

PrecomputedActivePauliMeasurementKernel::PrecomputedActivePauliMeasurementKernel(
    const ActivePauliAction& action_,
    int pivot_) {
    if (action_.nqubits <= 0) {
        fail("cannot build an active measurement kernel for k == 0");
    }
    if (pivot_ < 0 || pivot_ >= action_.nqubits) {
        fail("active measurement pivot is out of range");
    }
    action = action_;
    pivot = pivot_;
    out_dim = active_length(action.nqubits) >> 1;
    constexpr double inv_sqrt2 = 0.707106781186547524400844362104849039;
    nondiagonal_coefficient1_even = std::conj(action.even_phase) * inv_sqrt2;
    if (action_.xmask != 0) {
        if ((action_.xmask & (std::uint64_t{1} << pivot)) == 0) {
            fail("nondiagonal active measurement pivot must have an X component");
        }
        is_diagonal = false;
    } else if (action_.zmask != 0) {
        if ((action_.zmask & (std::uint64_t{1} << pivot)) == 0) {
            fail("diagonal active measurement pivot must have a Z component");
        }
        is_diagonal = true;
        const bool negative_phase = std::abs(action.even_phase.real() + 1.0) < 1e-12 &&
                                    std::abs(action.even_phase.imag()) < 1e-12;
        const bool positive_phase = std::abs(action.even_phase.real() - 1.0) < 1e-12 &&
                                    std::abs(action.even_phase.imag()) < 1e-12;
        if (!negative_phase && !positive_phase) {
            fail("diagonal active measurement Pauli must have real eigenvalues");
        }
        diagonal_phase_bit = negative_phase ? 1 : 0;
        z_without_pivot = action.zmask & ~(std::uint64_t{1} << pivot);
    } else {
        fail("cannot build an active measurement kernel for identity Pauli");
    }
}

void apply_pauli(std::vector<Complex>& out, const PauliString& pauli, const std::vector<Complex>& alpha) {
    if (pauli.nqubits >= 63) {
        fail("Pauli string has too many qubits for active-state basis indexing");
    }
    const std::size_t dim = active_length(pauli.nqubits);
    if (alpha.size() < dim || out.size() < dim || out.data() == alpha.data()) {
        fail("invalid active Pauli application buffers");
    }
    const std::uint64_t xmask = active_mask_x(pauli);
    const std::uint64_t zmask = active_mask_z(pauli);
    const Complex even = phase_factor(pauli.phase_exponent());
    const Complex odd = -even;
    for (std::size_t src = 0; src < dim; ++src) {
        const std::size_t dst = src ^ xmask;
        out[dst] = (is_odd_popcount(static_cast<std::uint64_t>(src) & zmask) ? odd : even) * alpha[src];
    }
}

void apply_pauli(ActiveState& state, const PauliString& pauli) {
    if (pauli.nqubits != state.k) {
        fail("Pauli string dimension does not match active state");
    }
    const ActivePauliAction action(pauli);
    const std::size_t dim = state.dim();
    if (action.xmask == 0) {
        for (std::size_t src = 0; src < dim; ++src) {
            state.alpha[src] *= active_action_phase(action, src);
        }
        return;
    }
    const std::size_t selector = action.xmask & (~action.xmask + 1);
    for (std::size_t base = 0; base < dim; ++base) {
        if ((base & selector) != 0) {
            continue;
        }
        const std::size_t paired = base ^ action.xmask;
        const Complex a0 = state.alpha[base];
        const Complex a1 = state.alpha[paired];
        const bool parity0 = active_action_phase_odd(action, base);
        const Complex phase0 = parity0 ? action.odd_phase : action.even_phase;
        const Complex phase1 = (parity0 != action.xz_overlap_odd) ? action.odd_phase : action.even_phase;
        state.alpha[base] = phase1 * a1;
        state.alpha[paired] = phase0 * a0;
    }
}

void rotate_pauli(ActiveState& state, const PauliString& pauli, double kernel_angle) {
    rotate_pauli(state, ActivePauliAction(pauli), kernel_angle);
}

void rotate_pauli(ActiveState& state, const ActivePauliAction& action, double kernel_angle) {
    if (action.nqubits != state.k) {
        fail("Pauli action dimension does not match active state");
    }
    const std::size_t dim = state.dim();
    const double c = std::cos(kernel_angle);
    const Complex minus_i_s(0.0, -std::sin(kernel_angle));
    if (action.xmask == 0) {
        for (std::size_t basis = 0; basis < dim; ++basis) {
            state.alpha[basis] *= Complex(c, 0.0) + minus_i_s * active_action_phase(action, basis);
        }
        return;
    }
    const std::size_t selector = action.xmask & (~action.xmask + 1);
    for (std::size_t base = 0; base < dim; ++base) {
        if ((base & selector) != 0) {
            continue;
        }
        const std::size_t paired = base ^ action.xmask;
        const Complex a0 = state.alpha[base];
        const Complex a1 = state.alpha[paired];
        const bool parity0 = active_action_phase_odd(action, base);
        const Complex phase0 = parity0 ? action.odd_phase : action.even_phase;
        const Complex phase1 = (parity0 != action.xz_overlap_odd) ? action.odd_phase : action.even_phase;
        state.alpha[base] = c * a0 + minus_i_s * phase1 * a1;
        state.alpha[paired] = c * a1 + minus_i_s * phase0 * a0;
    }
}

void rotate_pauli(ActiveState& state, const PrecomputedActivePauliRotationKernel& kernel, bool sign) {
    if (kernel.action.nqubits != state.k) {
        fail("rotation kernel dimension does not match active state");
    }
    rotate_pauli(state, kernel.action, sign ? -kernel.kernel_angle : kernel.kernel_angle);
}

std::string active_simd_backend() {
    return simd::dispatch_name();
}

} // namespace symft
