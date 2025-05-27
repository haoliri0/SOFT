#include "cstdio"
#include "../source/simulator.hpp"

using namespace StnCuda;

int main() {
    const Simulator simulator(1, 2, 4);
    printf("qubits_n: %ld\n", simulator.qubits_n);
}