#pragma once

#include <xtensor.hpp>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using complex_t = std::complex<double>;

struct QuantumGate {
    std::string gate;          
    std::vector<int> targets;
    std::vector<double> args;
    bool invert = false;
};

struct MeasureResults {
    int    reg;
    double out0;
    double out1;
    double ev;
};

struct DetectorResult {
    std::vector<int> targets;
    bool value = false;
};

struct ObservableResult {
    int index = 0;
    std::vector<int> targets;
    bool value = false;
};

struct SimulationResult {
    std::vector<MeasureResults> measurements;
    std::vector<DetectorResult> detectors;
    std::vector<ObservableResult> observables;
    std::vector<int> observable_bits;
    bool discarded = false;
    int failed_detector_index = -1;
    int executed_operations = 0;
};

struct CircuitInput {
    std::vector<QuantumGate> circuit;
    int num_qubits = 0;
    std::string source_path;
};

class Stabilizer {

public:

    xt::xtensor<bool, 2> tableau;
    std::unordered_map<long long, complex_t> xvec;
    int num_qubits;
    int omp_threshold;

    Stabilizer(int num_qubits):
        num_qubits(num_qubits),
        tableau(xt::zeros<bool>({num_qubits * 2, num_qubits * 2 + 1}))
        {
            for(int i = 0; i < 2 * num_qubits; i++){
                for (int j = 0; j < 2 * num_qubits + 1; j++){
                    if (i == j) this->tableau(i, i) = true;
                    else this->tableau(i, j) = false;
                }
            }
            this->omp_threshold = 100000;
            this->xvec[0] = complex_t(1, 0);
        }

    void _x(int qubit);
    void _y(int qubit);
    void _z(int qubit);
    void _h(int qubit);
    void _s(int qubit);
    void _sdg(int qubit);
    void _cx(int c, int t);
    void _t(int qubit);
    void _tdg(int qubit);
    void _reset(int qubit);
    MeasureResults _measure(int qubit, bool reset_flag);
    MeasureResults _measure_pauli_product(const std::vector<int>& targets, const std::vector<double>& paulis, bool invert = false);
    void init();
    std::vector<MeasureResults> sim(std::vector<QuantumGate>& circuit);
    SimulationResult run(std::vector<QuantumGate>& circuit, bool stop_on_detector = true, bool print_detector_errors = false);
    void print_xvec();
    xt::xtensor<bool, 1> phase();

private:
    int calc_g(bool x1, bool z1, bool x2, bool z2);
    void rowsum(int row1, int row2);
    void multiply_bool_pauli(complex_t& phase, xt::xtensor<bool, 1>& pauli1, const xt::xtensor<bool, 1> pauli2);
    int check_comm(xt::xtensor<bool, 1>& gate, xt::xtensor<bool, 1> entry, xt::xtensor<bool, 1> complement, complex_t& phase, xt::xtensor<bool, 1>& pauli, int qubit);
    void gate_decomposition(xt::xtensor<bool, 1>& gate, xt::xtensor<bool, 1>& destab, xt::xtensor<bool, 1>& stab, complex_t& phase, int qubit);
    void bin(long long value, xt::xtensor<bool, 1>& bin_v);
    long long convert(xt::xtensor<bool, 1>& bin_a, xt::xtensor<bool, 1>& bin_b);
    void meas_tableau(xt::xtensor<bool, 1>& obs, xt::xtensor<bool, 1>& destab, xt::xtensor<bool, 1>& stab, int sign);
    void renorm();
    void tgate_decomp(std::vector<complex_t>& coefs, std::vector<xt::xtensor<bool, 1>>& destab_list, std::vector<xt::xtensor<bool, 1>>& stab_list, int qubit, bool dag);
    void update_xvec(std::vector<complex_t>& coefs, std::vector<xt::xtensor<bool, 1>>& destab_list, std::vector<xt::xtensor<bool, 1>>& stab_list);
};

CircuitInput load_stim_file(const std::string& path);
CircuitInput load_stim_text(const std::string& text, const std::string& source_name = "<string>");
std::vector<SimulationResult> run_stim_text(const std::string& text, int shots = 1, int num_qubits = -1, bool stop_on_detector = true);
std::vector<SimulationResult> run_stim_file(const std::string& path, int shots = 1, int num_qubits = -1, bool stop_on_detector = true);
std::string shot_result_to_json(const SimulationResult& result);
void write_shot_results_jsonl(const std::string& path, const std::vector<SimulationResult>& results);
