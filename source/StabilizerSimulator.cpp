#include "./StabilizerSimulator.hpp"

StabilizerSimulator::StabilizerSimulator(const int qubits_n): _qubits_n(qubits_n) {}

int StabilizerSimulator::qubits_n() const {
    return this->_qubits_n;
}