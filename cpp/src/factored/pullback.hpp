#pragma once

#include "core/frames.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace symft::detail {

inline constexpr std::size_t kMinimumBatchedPullbackCount = 128;

enum class LocalCliffordGate : std::uint8_t {
    H,
    H_NXY,
    H_NXZ,
    H_NYZ,
    H_XY,
    H_YZ,
    C_NXYZ,
    C_NZYX,
    C_XNYZ,
    C_XYNZ,
    C_XYZ,
    C_ZNYX,
    C_ZYNX,
    C_ZYX,
    S,
    SDG,
    SqrtX,
    SqrtXDag,
    SqrtY,
    SqrtYDag,
    X,
    Y,
    Z,
    CX,
    CY,
    CZ,
    SWAP,
    CXSWAP,
    CZSWAP,
    ISWAP,
    ISWAP_DAG,
    SQRT_XX,
    SQRT_XX_DAG,
    SQRT_YY,
    SQRT_YY_DAG,
    SQRT_ZZ,
    SQRT_ZZ_DAG,
    SWAPCX,
    XCX,
    XCY,
    XCZ,
    YCX,
    YCY,
    YCZ,
    Count,
};

// Alternative to materializing global Clifford and symbolic Pauli frames.
// Events stay in sparse physical coordinates. The per-qubit index lets each
// pending Pauli traverse only its backward Clifford light cone and corrections
// that can overlap its current support.
class DirectPullbackFrame {
  public:
    DirectPullbackFrame() = default;
    explicit DirectPullbackFrame(int nqubits);

    std::size_t event_count() const;
    void append_clifford(LocalCliffordGate gate, int q);
    void append_clifford(LocalCliffordGate gate, int q0, int q1);
    void append_pauli(const PauliString& pauli, int condition);
    void append_single_qubit_pauli(int q, bool x, bool z, int condition);
    SymbolicPauliString pull_back(const PauliString& pauli) const;
    SymbolicPauliString pull_back(
        const PauliString& pauli,
        std::size_t event_bound) const;
    std::vector<SymbolicPauliString> pull_back_batch(
        std::span<const PauliString> paulis,
        std::span<const std::size_t> event_bounds,
        std::span<const SymbolicBool> initial_signs = {}) const;

  private:
    struct PauliWord {
        std::size_t word = 0;
        std::uint64_t x = 0;
        std::uint64_t z = 0;
    };

    struct Event {
        enum class Kind : std::uint8_t {
            Clifford,
            ConditionalPauli,
        };

        Kind kind = Kind::Clifford;
        LocalCliffordGate gate = LocalCliffordGate::H;
        int q0 = 0;
        int q1 = 0;
        int condition = 0;
        std::size_t pauli_word_begin = 0;
        std::size_t pauli_word_count = 0;
    };

    int nqubits_ = 0;
    std::vector<Event> events_;
    std::vector<PauliWord> pauli_words_;
    std::vector<std::vector<std::size_t>> event_indices_by_qubit_;

    bool anticommutes_with(const PauliString& pauli, const Event& event) const;
};

} // namespace symft::detail
