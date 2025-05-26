#ifndef StabilizerSimulator_HPP
#define StabilizerSimulator_HPP

class StabilizerSimulator {
    int _qubits_n;
public:
    explicit StabilizerSimulator(int qubits_n);

    [[nodiscard]]
    int qubits_n() const;

    // void apply_reset(int qubit);
    // void apply_y(int qubit);
    // void apply_z(int qubit);
    // void apply_h(int qubit);
    // void apply_s(int qubit);
    // void apply_sdg(int qubit);
    // void apply_t(int qubit);
    // void apply_tdg(int qubit);
    // void apply_cnot(int qubit);
};

#endif