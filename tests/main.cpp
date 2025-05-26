#include "cstdio"
#include "../source/StabilizerSimulator.hpp"

int main() {
    StabilizerSimulator simulator(2);
    printf("qubits_n: %d\n", simulator.qubits_n());
}