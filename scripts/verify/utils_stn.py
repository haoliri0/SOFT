import re
import subprocess
import sys
from typing import Iterable, Iterator

from qiskit.quantum_info import Clifford

from scripts.verify.utils_ops import Op
from scripts.verify.utils_clifford import compute_clifford_state
from scripts.verify.utils_str import split_and_clean_lines


def make_stn_cmd(
    exec_file_path: str,
    qubits_n: int,
    entries_m: int,
    results_n: int,
    mode: int,
):
    return [
        exec_file_path,
        "--shots_n", "1",
        "--qubits_n", str(qubits_n),
        "--entries_m", str(entries_m),
        "--results_m", str(results_n),
        "--mode", str(mode)]


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
        case ('M', target):
            if results is not None:
                result = next(results)
                return f'D {target} {result}'
            else:
                return f'M {target}'
        case _:
            raise ValueError(f"Unsupported operation: {op}")


def make_stn_ops(ops: Iterable[Op], results: Iterable[int] | None = None):
    results = iter(results) if results is not None else None
    return "\n".join([make_stn_op(op, results) for op in ops])


def parse_stn_mode1_stdout_line(line: str):
    if not line.startswith("0,"):
        raise ValueError
    if line.startswith("0,1"):
        raise ValueError
    return float(line[6:])


def parse_stn_mode1_stdout(stdout: str) -> tuple[float, ...]:
    lines = split_and_clean_lines(stdout)
    probs = map(parse_stn_mode1_stdout_line, lines)
    return tuple(probs)


def parse_stn_stdout_table(table_str: str) -> Clifford:
    lines = split_and_clean_lines(table_str)
    lines = map(lambda line: line[0:1] + line[1:][::-1], lines)
    lines = tuple(lines)
    return Clifford(lines)


def parse_stn_stdout_entry(s: str):
    bst, amp = s.split(':')
    bst = bst.strip()
    bst = map(int, bst)
    bst = map(bool, bst)
    bst = tuple(bst)
    amp = amp.strip()
    real, imag, _ = amp.split(' ')
    amp = float(real) + 1j * float(imag)
    return bst, amp


def parse_stn_stdout_entries(entries_str: str):
    lines = split_and_clean_lines(entries_str)
    matching = re.match(r'entries_n=([0-9]+)', lines[0])
    entries_n = int(matching.group(1))
    entries = map(parse_stn_stdout_entry, lines[1:1 + entries_n])
    return tuple(entries)


def parse_stn_stdout_state(s: str):
    table_title = 'table:'
    decomp_title = 'decomposed:'
    entries_title = 'entries:'
    results_title = 'results:'

    table_head = s.find(table_title, 0, len(s))
    table_tail = s.find(decomp_title, table_head, len(s))
    entries_head = s.find(entries_title, table_tail, len(s))
    entries_tail = s.find(results_title, entries_head, len(s))
    entries_tail = s.find("\n\n", entries_head, len(s)) \
        if entries_tail == -1 else entries_tail
    if -1 in (table_head, table_tail, entries_head, entries_tail):
        raise ValueError

    table_str = s[table_head + len(table_title):table_tail]
    entries_str = s[entries_head + len(entries_title):entries_tail]
    clifford = parse_stn_stdout_table(table_str)
    entries = parse_stn_stdout_entries(entries_str)
    return compute_clifford_state(clifford, entries)


def parse_stn_mode2_stdout(stdout: str):
    states_str = stdout.split('Simulator:')
    states = map(parse_stn_stdout_state, states_str[1:])
    return tuple(states)


def run_stn_mode1(
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
        results_n=results_n,
        mode=1)
    process = subprocess.Popen(cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)

    stdin = make_stn_ops(ops, results)
    stdout, stderr = process.communicate(stdin)
    if process.returncode != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError

    return parse_stn_mode1_stdout(stdout)


def run_stn_mode2(
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
        results_n=results_n,
        mode=2)
    process = subprocess.Popen(cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)

    stdin = make_stn_ops(ops, results)
    stdout, stderr = process.communicate(stdin)

    if process.returncode != 0:
        print(stderr, file=sys.stderr)
        raise RuntimeError

    return parse_stn_mode2_stdout(stdout)
