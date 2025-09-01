import subprocess
import sys
from io import StringIO
from typing import Iterable, Iterator

import numpy as np

from scripts.utils import stn
from scripts.verify.utils_clifford import compute_statevector_from_stn
from scripts.verify.utils_ops import Op


def make_stn_op(op: Op, results: Iterator[int] | None = None):
    match op:
        case ('CX', (control, target)):
            return f'CX {control} {target}'
        case ('X', target):
            return f'X {target}'
        case ('Y', target):
            return f'Y {target}'
        case ('Z', target):
            return f'Z {target}'
        case ('H', target):
            return f'H {target}'
        case ('S', target):
            return f'S {target}'
        case ('SDG', target):
            return f'SDG {target}'
        case ('T', target):
            return f'T {target}'
        case ('TDG', target):
            return f'TDG {target}'
        case ('MEASURE', target):
            if results is not None:
                result = next(results)
                return f'DESIRE {target} {result}'
            else:
                return f'MEASURE {target}'
        case _:
            raise ValueError(f"Unsupported operation: {op}")


def make_stn_stdin(ops: Iterable[Op], results: Iterable[int] | None = None):
    results = iter(results) if results is not None else None
    stdin_io = StringIO()
    for op in ops:
        stdin_io.write(make_stn_op(op, results))
        stdin_io.write("\n")
        stdin_io.write("PRINT STATE")
        stdin_io.write("\n")
    return stdin_io.getvalue()


def parse_stn_state(lines: Iterator[str], args: stn.Args) -> np.ndarray:
    error, table, entries = stn.read_printed_shots_state(lines, args)[0]
    if error:
        raise RuntimeError(f"Simulator error with code {error}")
    statevector = compute_statevector_from_stn(table, entries)
    return statevector


def parse_stn_stdout(stdout: str, args: stn.Args) -> tuple[np.ndarray, ...]:
    stdout_io = StringIO(stdout)
    stdout_lines = iter(stdout_io)
    steps_statevector = []
    while True:
        try:
            statevector = parse_stn_state(stdout_lines, args)
            steps_statevector.append(statevector)
        except StopIteration:
            break
    return tuple(steps_statevector)


def run_stn_and_collect_states(
    exec_file_path: str,
    ops: Iterable[Op],
    results: Iterable[int],
    qubits_n: int,
    entries_m: int,
    results_n: int,
):
    args = stn.Args(
        qubits_n=qubits_n,
        entries_m=entries_m,
        mem_ints_m=results_n)
    cmd = stn.make_cmd(
        exec_file_path=exec_file_path,
        args=args)
    process = subprocess.Popen(cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)

    stdin = make_stn_stdin(ops, results)
    stdout, stderr = process.communicate(stdin)
    if process.returncode != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError

    return parse_stn_stdout(stdout, args)
