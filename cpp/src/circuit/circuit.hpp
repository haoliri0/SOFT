#pragma once

#include "factored/factored.hpp"

#include <vector>

namespace symft {

enum class CircuitInstructionKind {
    Tick,
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
    T,
    TDag,
    PauliRotation,
    MZ,
    MX,
    MY,
    MRZ,
    MRX,
    MRY,
    RZ,
    RX,
    RY,
    MPP,
    XError,
    YError,
    ZError,
    Depolarize1,
    Depolarize2,
    Depolarize3,
    PauliChannel1,
    PauliChannel2,
    PauliChannel3,
    PauliProductError,
    PauliProductChannel,
    HeraldedErase,
    HeraldedPauliChannel1,
    MPad,
    FeedbackX,
    FeedbackY,
    FeedbackZ,
};

struct CircuitMeasurementTarget {
    int qubit = 0;
    bool inverted = false;
};

struct CircuitPauliProduct {
    PauliString pauli;
    bool inverted = false;
};

struct CircuitFeedbackTarget {
    int record = 0;
    int qubit = 0;
};

struct CircuitInstruction {
    CircuitInstructionKind kind = CircuitInstructionKind::Tick;
    double probability = 0.0;
    double kernel_angle = 0.0;
    std::vector<int> qubits;
    std::vector<CircuitMeasurementTarget> measurement_targets;
    std::vector<CircuitPauliProduct> pauli_products;
    std::vector<CircuitFeedbackTarget> feedback_targets;
    std::vector<double> probabilities;
    int line = 0;
};

struct CircuitDetector {
    std::vector<int> records;
    std::vector<double> coords;
    int line = 0;
    int after_instruction = 0;
    int after_pending_operation = 0;
};

struct CircuitObservableInclude {
    int index = 0;
    std::vector<int> records;
    int line = 0;
};

struct QuantumCircuit {
    int nqubits = 0;
    int nrecords = 0;
    std::vector<CircuitInstruction> instructions;
    std::vector<CircuitDetector> detectors;
    std::vector<CircuitObservableInclude> observables;
};

struct CircuitLoweringResult {
    FrameFactoredState state;
    std::vector<SymbolicBool> measurement_records;
    std::vector<int> instruction_pending_operation_counts;
};

CircuitLoweringResult lower_circuit_to_factored(const QuantumCircuit& circuit);

} // namespace symft
