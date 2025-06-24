#include <cstdio>
#include "./scan.hpp"
#include "../source/simulator.hpp"

void apply_operation(const Simulator &simulator, const OperationArgs args) {
    const OpType type = args.type;
    const Qid target0 = args.target0;
    const Qid target1 = args.target1;

    if (type == X)
        return simulator.apply_x(target0);
    if (type == Y)
        return simulator.apply_y(target0);
    if (type == Z)
        return simulator.apply_z(target0);
    if (type == H)
        return simulator.apply_h(target0);
    if (type == S)
        return simulator.apply_s(target0);
    if (type == SDG)
        return simulator.apply_sdg(target0);
    if (type == T)
        return simulator.apply_t(target0);
    if (type == TDG)
        return simulator.apply_tdg(target0);
    if (type == CX)
        return simulator.apply_cx(target0, target1);
    if (type == M)
        return simulator.measure(target0);
    if (type == D)
        return simulator.desire(target0, target1);
    if (type == R)
        return simulator.assign(target0, target1);
}

int main() {
    SimulatorArgs args{};
    Simulator simulator;
    ScanError scan_err = Success;
    cudaError cuda_err = cudaSuccess;

    do {
        scan_err = scan_simulator_args(stdin, &args);
        if (scan_err != Success) break;

        cuda_err = simulator.create(args.shots_n, args.qubits_n, args.amps_m, args.results_m);
        if (cuda_err != cudaSuccess) break;

        fprintf(stderr, "Simulator created\n");
        fprintf(stderr, "shots_n=%u, qubits_n=%u, amps_m=%u, results_m=%u\n",
            args.shots_n, args.qubits_n, args.amps_m, args.results_m);

        while (true) {
            OperationArgs op{};
            scan_err = scan_operation_args(stdin, op);
            if (scan_err != Success) break;
            apply_operation(simulator, op);
        }

        if (scan_err == ReadLineFailed)
            scan_err = Success;

    } while (false);

    if (scan_err != Success)
        fprintf(stderr, "%s", get_scan_error_name(scan_err));

    if (cuda_err != cudaSuccess)
        fprintf(stderr, "%s\n%s", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));

    simulator.destroy();
}
