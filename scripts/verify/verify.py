import os
import sys
from itertools import count
from typing import Iterable

import numpy as np
from tqdm import tqdm

project_dir_path = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(project_dir_path)

from scripts.verify.utils_ops import Op, generate_random_ops, parse_ops, print_ops
from scripts.verify.utils_qiskit import make_qiskit_circuit, run_qiskit_circuit
from scripts.verify.utils_soft import run_soft_and_collect_states

np.set_printoptions(precision=6, sign='+', floatmode='fixed')


def sync_global_phase(state0: np.ndarray, state1: np.ndarray) -> np.ndarray:
    index = np.argmax(np.abs(state0) * np.abs(state1))
    angle0 = np.angle(state0[index])
    angle1 = np.angle(state1[index])
    angle_diff = angle0 - angle1
    coef = np.exp(1j * angle_diff)
    return state1 * coef


def check_steps_state(states: Iterable[np.ndarray], states_soft: Iterable[np.ndarray]) -> int | None:
    for index, (state, state_soft) in enumerate(zip(states, states_soft)):
        state = np.asarray(state)
        state_soft = np.asarray(state_soft)
        state_soft = sync_global_phase(state, state_soft)
        if not np.isclose(state, state_soft, atol=1e-5).all():
            return index
    return None


def verify_ops(*,
    label: str | None = None,
    exec_file_path: str,
    qubits_n: int,
    entries_m: int,
    ops: Iterable[Op],
):
    ops = tuple(ops)
    results_n = sum(typ == "MEASURE" for typ, _ in ops)

    circuit = make_qiskit_circuit(ops, qubits_n, results_n)
    states, results, _ = run_qiskit_circuit(circuit)
    states_soft = run_soft_and_collect_states(
        exec_file_path, ops, results,
        qubits_n, entries_m, results_n)

    err_step_index = check_steps_state(states, states_soft)
    if err_step_index is not None:
        print("\n\n\n")
        print("Found error!!!!!")
        if label is not None:
            print(f"label={label}")
        print(f"circuit=")
        print_ops(ops, results)
        print(f"results={results}")
        print(f"err_step={err_step_index}")
        print(f"state_expected=\n\t{states[err_step_index]!r}")
        print(f"state_actual=\n\t{states_soft[err_step_index]!r}")

    return err_step_index is None


def verify_custom(*,
    exec_file_path: str,
    qubits_n: int,
    entries_m: int,
    ops_str: str,
):
    ops = parse_ops(ops_str)
    verify_ops(
        exec_file_path=exec_file_path,
        qubits_n=qubits_n,
        entries_m=entries_m,
        ops=ops)


def verify_random(*,
    exec_file_path: str,
    qubits_n: int,
    entries_m: int,
    ops_n: int,
    seed_head: int = 0,
    seed_tail: int | None = None,
):
    seeds = range(seed_head, seed_tail) \
        if seed_tail is not None else count(seed_head)
    for seed in tqdm(seeds):
        rng = np.random.default_rng(seed)
        ops = generate_random_ops(ops_n, qubits_n, rng)
        verified = verify_ops(
            label=f"seed_{seed}",
            exec_file_path=exec_file_path,
            qubits_n=qubits_n,
            entries_m=entries_m,
            ops=ops)
        if not verified:
            break


if __name__ == "__main__":
    exec_file_path = os.path.join(project_dir_path, "cmake-build-release/soft_cuda_exec")
    verify_random(
        exec_file_path=exec_file_path,
        qubits_n=8,
        entries_m=1024,
        ops_n=1024)
