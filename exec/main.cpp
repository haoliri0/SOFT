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

void print_results(const Simulator &simulator, const Rid results_n, const Rid result_i) {
    const ShotsStatePtr shots_state_ptr = simulator.shots_state_ptr;
    const Kid results_m = shots_state_ptr.results_m;
    const Sid shots_n = shots_state_ptr.shots_n;
    cudaStreamSynchronize(simulator.stream);
    for (Sid shot_i = 0; shot_i < shots_n; ++shot_i) {
        const ShotStatePtr shot_state_ptr = shots_state_ptr.get_shot_ptr(shot_i);

        Kid amps_n;
        cudaMemcpy(&amps_n, shot_state_ptr.get_amps_ptr().get_amps_n_ptr(),
            sizeof(Kid), cudaMemcpyDeviceToHost);
        const Bit failed = amps_n == 0;

        Bit results_bit[results_m];
        cudaMemcpy(results_bit, shot_state_ptr.get_results_ptr().get_bits_ptr(),
            results_m * sizeof(Bit), cudaMemcpyDeviceToHost);

        Flt results_prob[results_m];
        cudaMemcpy(results_prob, shot_state_ptr.get_results_ptr().get_probs_ptr(),
            results_m * sizeof(Flt), cudaMemcpyDeviceToHost);

        for (Rid result_j = result_i; result_j < results_n; ++result_j) {
            const Bit bit = results_bit[result_j % results_m];
            const Flt prob = results_prob[result_j % results_m];
            printf("%u,%u,%u,%f\n", shot_i, failed, bit, prob);
        }
    }
}

int main() {
    SimulatorArgs args{};
    Simulator simulator;
    ScanError scan_err = Success;
    cudaError cuda_err = cudaSuccess;

    do {
        scan_err = scan_simulator_args(stdin, &args);
        if (scan_err != Success) break;
        const Sid shots_n = args.shots_n;
        const Qid qubits_n = args.qubits_n;
        const Kid amps_m = args.amps_m;
        const Rid results_m = args.results_m;

        cuda_err = simulator.create(shots_n, qubits_n, amps_m, results_m);
        if (cuda_err != cudaSuccess) break;

        fprintf(stderr, "Simulator created\n");
        fprintf(stderr, "shots_n=%u, qubits_n=%u, amps_m=%u, results_m=%u\n",
            shots_n, qubits_n, amps_m, results_m);

        Rid results_n = 0;
        Rid results_n_printed = 0;

        while (true) {
            OperationArgs op{};
            scan_err = scan_operation_args(stdin, op);
            if (scan_err != Success) break;
            apply_operation(simulator, op);

            if (op.type == M || op.type == D || op.type == R) {
                results_n += 1;
                if (results_n - results_n_printed >= results_m) {
                    print_results(simulator, results_n, results_n_printed);
                    results_n_printed = results_n;
                }
            }
        }

        if (results_n - results_n_printed > 0)
            print_results(simulator, results_n, results_n_printed);

        if (scan_err == ReadLineFailed)
            scan_err = Success;

    } while (false);

    if (scan_err != Success)
        fprintf(stderr, "%s", get_scan_error_name(scan_err));

    if (cuda_err != cudaSuccess)
        fprintf(stderr, "%s\n%s", cudaGetErrorName(cuda_err), cudaGetErrorString(cuda_err));

    simulator.destroy();
}
