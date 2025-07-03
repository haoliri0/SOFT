import os
from typing import Iterable

import numpy as np
from tqdm import tqdm

from scripts.verify.utils_ops import Op, generate_random_ops, parse_ops, print_ops
from scripts.verify.utils_qiskit import make_qiskit_circuit, run_qiskit_circuit
from scripts.verify.utils_stn import make_stn_cmd, run_stn_mode2

np.set_printoptions(precision=6, sign='+', floatmode='fixed')


def sync_global_phase(state0, state1):
    index = np.argmax(np.abs(state0) * np.abs(state1))
    angle0 = np.angle(state0[index])
    angle1 = np.angle(state1[index])
    angle_diff = angle0 - angle1
    coef = np.exp(1j * angle_diff)
    return state1 * coef


def verify_ops(
    key: str,
    ops: Iterable[Op],
    qubits_n: int,
    amps_m: int,
):
    ops = tuple(ops)
    results_n = sum(typ == "M" for typ, _ in ops)
    qc = make_qiskit_circuit(ops, qubits_n, results_n)
    states, results, _ = run_qiskit_circuit(qc)

    project_dir_path = os.path.join(os.path.dirname(__file__), "../..")
    exec_file_path = os.path.join(project_dir_path, "cmake-build-release/stn_cuda_exec")
    states_stn = run_stn_mode2(
        exec_file_path, ops, results,
        qubits_n, amps_m, max(results_n, 1))

    states = np.asarray(states)
    states_stn = np.asarray(states_stn)

    states_stn = np.asarray([
        sync_global_phase(state, state_stn)
        for state, state_stn in zip(states, states_stn)])

    close = np.isclose(states, states_stn, atol=1e-5)
    err_step_indices = np.nonzero(~close)[0]
    if len(err_step_indices) > 0:
        err_step_index = err_step_indices[0]
        print("\n\n\n")
        print("Found error!!!!!")
        print(f"key={key}")
        cmd = make_stn_cmd(
            exec_file_path=exec_file_path,
            qubits_n=qubits_n,
            amps_m=amps_m,
            results_n=results_n,
            mode=2)
        print("command=" + " ".join(cmd))
        print(f"circuit=")
        print_ops(ops, results)
        print(f"results={results}")
        print(f"err_step={err_step_index}")
        print(f"state_expected=\n\t{states[err_step_index]!r}")
        print(f"state_actual=\n\t{states_stn[err_step_index]!r}")
        return False
    return True


def verify_custom(
    ops_str: str,
    qubits_n: int,
    amps_m: int,
):
    ops = parse_ops(ops_str)
    verify_ops("custom", ops, qubits_n, amps_m)


def verify_random(
    ops_n: int = 16,
    qubits_n: int = 2,
    amps_m: int = 8,
    seed_head: int = 0,
    seed_tail: int = 1024,
):
    for seed in tqdm(range(seed_head, seed_tail)):
        rng = np.random.default_rng(seed)
        ops = generate_random_ops(ops_n, qubits_n, rng)
        verified = verify_ops(
            key=f"seed_{seed}",
            ops=ops,
            qubits_n=qubits_n,
            amps_m=amps_m)
        if not verified:
            break


if __name__ == "__main__":
    verify_random(
        qubits_n=4,
        amps_m=64,
        ops_n=256)
