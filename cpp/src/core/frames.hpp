#pragma once

#include "core/pauli.hpp"
#include "core/symbolic.hpp"

#include <memory>

namespace symft {

struct ConditionalPauliString {
    PauliString pauli;
    int condition = 0;

    ConditionalPauliString() = default;
    ConditionalPauliString(PauliString pauli, int condition);
};

bool operator==(const ConditionalPauliString& lhs, const ConditionalPauliString& rhs);

struct SymbolicPauliString {
    PauliString pauli;
    SymbolicBool sign;

    SymbolicPauliString() = default;
    explicit SymbolicPauliString(PauliString pauli);
    SymbolicPauliString(PauliString pauli, SymbolicBool sign);
};

bool operator==(const SymbolicPauliString& lhs, const SymbolicPauliString& rhs);

struct ActivePauliFrame {
    int k = 0;
    std::vector<ConditionalPauliString> terms;
    std::shared_ptr<SymbolicContext> context;

    ActivePauliFrame() = default;
    explicit ActivePauliFrame(int k);
    ActivePauliFrame(int k, std::shared_ptr<SymbolicContext> context);

    ConditionalPauliString add_pauli(const PauliString& pauli);
    ConditionalPauliString add_pauli(const PauliString& pauli, int condition);
};

SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const PauliString& pauli);
SymbolicPauliString conjugate_by(const ConditionalPauliString& cp, const SymbolicPauliString& pauli);
SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const PauliString& pauli);
SymbolicPauliString conjugate_by(const ActivePauliFrame& frame, const SymbolicPauliString& pauli);

struct DormantState {
    int d = 0;
    std::vector<SymbolicBool> bits;
    std::shared_ptr<SymbolicContext> context;

    DormantState() = default;
    explicit DormantState(int d);
    DormantState(int d, std::shared_ptr<SymbolicContext> context);
    DormantState(std::vector<SymbolicBool> bits, std::shared_ptr<SymbolicContext> context);

    SymbolicBool dormant_bit(int q) const;
    void set_dormant_bit(int q, const SymbolicBool& value);
    SymbolicBool assign_dormant_symbol(int q);
};

struct CliffordFrame {
    int nqubits = 0;
    int nwords = 0;
    std::vector<PauliString> rows;

    CliffordFrame() = default;
    explicit CliffordFrame(int nqubits);

    int xrow(int q) const;
    int zrow(int q) const;
    void copy_pauli_to_row(int row, const PauliString& pauli);
};

bool operator==(const CliffordFrame& lhs, const CliffordFrame& rhs);
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

} // namespace symft
