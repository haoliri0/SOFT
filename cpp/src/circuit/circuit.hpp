#pragma once

#include "factored/factored.hpp"

#include <cstdint>
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
    EXP_VAL,
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
    int exp_val = -1;
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
    int nexpvals = 0;
    std::vector<CircuitInstruction> instructions;
    std::vector<CircuitDetector> detectors;
    std::vector<CircuitObservableInclude> observables;
};

struct CircuitFactorizationEstimate {
    // Counts refer to elementary target applications after REPEAT expansion,
    // not to source-level instruction lines.
    std::uint64_t clifford_operations = 0;
    // Noise, reset, and feedback instructions can expand into several
    // independently conditioned Pauli corrections.
    std::uint64_t conditional_pauli_operations = 0;
    std::uint64_t pending_pauli_operations = 0;
    // A two-qubit event contributes two touches; a Pauli product contributes
    // its physical weight. This supports the conservative locality guard used
    // by automatic strategy selection.
    std::uint64_t pullback_event_qubit_touches = 0;
    // Raw operation-count versions of
    //   n*nc + n*ne + (n^2 + n*ne)*(nt + nm)
    // and
    //   (nt + nm)*(nc + ne),
    // respectively. Automatic selection also accounts for operation order
    // and packed storage, but keeps those implementation details private.
    long double frame_work = 0.0;
    long double direct_pullback_work = 0.0;
    FactorizationStrategy preferred_strategy = FactorizationStrategy::CliffordFrames;
};

struct CircuitLoweringResult {
    FrameFactoredState state;
    std::vector<SymbolicBool> measurement_records;
    std::vector<int> instruction_pending_operation_counts;
    CircuitFactorizationEstimate factorization_estimate;
    FactorizationStrategy factorization_strategy = FactorizationStrategy::CliffordFrames;
};

CircuitFactorizationEstimate estimate_circuit_factorization(const QuantumCircuit& circuit);
CircuitLoweringResult lower_circuit_to_factored(
    const QuantumCircuit& circuit,
    FactorizationStrategy strategy = FactorizationStrategy::Automatic);

} // namespace symft
