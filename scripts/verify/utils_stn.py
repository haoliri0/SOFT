import subprocess
import sys
from io import StringIO
from typing import Iterable, Iterator

from qiskit.quantum_info import Clifford

from scripts.compare.compare import StnArgs, read_error, read_shot_state_content, read_specified_label
from scripts.verify.utils_clifford import compute_clifford_state
from scripts.verify.utils_ops import Op


def make_stn_cmd(
    exec_file_path: str,
    qubits_n: int,
    entries_m: int,
    results_n: int,
):
    return [
        exec_file_path,
        "--shots_n", "1",
        "--qubits_n", str(qubits_n),
        "--entries_m", str(entries_m),
        "--results_m", str(results_n)]


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


def read_state(lines: Iterator[str], args: StnArgs):
    read_specified_label(lines, "state")
    read_specified_label(lines, "shot 0")
    if error := read_error(lines):
        raise RuntimeError(f"Simulator error with code {error}")
    table, entries = read_shot_state_content(lines, args)
    table = tuple(
        row[0:1] + row[1:][::-1]
        for row in table)
    clifford = Clifford(table)
    entries = tuple(
        (tuple(bool(int(b)) for b in bst), amp)
        for bst, amp in entries.items())
    return compute_clifford_state(clifford, entries)


def parse_stn_mode2_stdout2(stdout: str, args: StnArgs):
    stdout_io = StringIO(stdout)
    stdout_lines = iter(stdout_io)
    states = []
    while True:
        try:
            state = read_state(stdout_lines, args)
            states.append(state)
        except StopIteration:
            break
    return tuple(states)


def run_stn_states(
    exec_file_path: str,
    ops: Iterable[Op],
    results: Iterable[int],
    qubits_n: int,
    entries_m: int,
    results_n: int,
):
    cmd = make_stn_cmd(
        exec_file_path=exec_file_path,
        qubits_n=qubits_n,
        entries_m=entries_m,
        results_n=results_n)
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

    args = StnArgs(
        qubits_n=qubits_n,
        entries_m=entries_m,
        results_m=results_n)
    return parse_stn_mode2_stdout2(stdout, args)
