import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread

import numpy as np

project_dir_path = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(project_dir_path)


@dataclass(kw_only=True)
class StnArgs:
    qubits_n: int
    entries_m: int
    results_m: int


def read_nonempty_line(lines: Iterator[str]) -> str:
    while True:
        line = next(lines)
        line = line.strip()
        if line == "":
            continue
        break
    return line


def read_label(lines: Iterator[str]) -> tuple[str, str]:
    line = read_nonempty_line(lines)
    label_pattern = re.compile(f"(.*):(.*)")
    match = label_pattern.match(line)
    if not match:
        raise ValueError
    label = match.group(1).strip()
    content = match.group(2).strip()
    return label, content


def read_specified_label(lines: Iterator[str], label: str):
    label_, content = read_label(lines)
    if label_ != label:
        raise ValueError
    return content


def read_args(lines: Iterator[str]) -> StnArgs:
    read_specified_label(lines, "args")

    qubits_n_pattern = re.compile(f"qubits_n=(.*)")
    entries_m_pattern = re.compile(f"entries_m=(.*)")
    results_m_pattern = re.compile(f"results_m=(.*)")

    line = next(lines)
    line = line.strip()
    qubits_n_match = qubits_n_pattern.match(line)
    if qubits_n_match is None:
        raise ValueError
    qubits_n = int(qubits_n_match.group(1))

    line = next(lines)
    line = line.strip()
    entries_m_match = entries_m_pattern.match(line)
    if entries_m_match is None:
        raise ValueError
    entries_m = int(entries_m_match.group(1))

    line = next(lines)
    line = line.strip()
    results_m_match = results_m_pattern.match(line)
    if results_m_match is None:
        raise ValueError
    results_m = int(results_m_match.group(1))

    return StnArgs(
        qubits_n=qubits_n,
        entries_m=entries_m,
        results_m=results_m)


def read_error(lines: Iterator[str]):
    error_pattern = re.compile(f"error=(.*)")
    line = read_nonempty_line(lines)
    match = error_pattern.match(line)
    if not match:
        raise ValueError
    error = match.group(1)
    error = error.strip()
    error = int(error)
    return error


def read_table_content(lines: Iterator[str], args: StnArgs) -> tuple[str, ...]:
    return tuple(read_nonempty_line(lines) for _ in range(args.qubits_n * 2))


def read_amps_content(lines: Iterator[str]) -> dict[str, complex]:
    line = read_nonempty_line(lines)
    entries_n_pattern = re.compile(f"entries_n=(.*)")
    entries_n_str = entries_n_pattern.match(line).group(1)
    entries_n = int(entries_n_str)

    amps = {}
    amp_entry_pattern = re.compile(f"(.*):(.*)")
    for _ in range(entries_n):
        line = read_nonempty_line(lines)
        match = amp_entry_pattern.match(line)
        key = match.group(1)
        key = key.strip()
        value = match.group(2)
        value = value.replace("i", "j")
        value = value.replace(" ", "")
        value = complex(value)
        amps[key] = value
    return amps


def read_shot_state_content(lines: Iterator[str], args: StnArgs):
    read_specified_label(lines, "table")
    table = read_table_content(lines, args)
    read_specified_label(lines, "amplitudes")
    amps = read_amps_content(lines)
    return table, amps


def main(
    exec_file_path: str,
    logs_file_path: str,
):
    with open(logs_file_path) as fp:
        args = read_args(fp)
        cmd = [
            exec_file_path,
            "--shots_n", "1",
            "--qubits_n", str(args.qubits_n),
            "--entries_m", str(args.entries_m),
            "--results_m", str(args.results_m)]
        print(f"{cmd=}")
        process = subprocess.Popen(cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True)

        errors = []
        queue = Queue(64)

        def thread_func():
            exhausted = False
            while True:
                message = queue.get()
                if message is None:
                    exhausted = True
                    break
                gate, table, amps = message
                if gate is not None:
                    print(f"gate: {gate}")

                print(f"table:")
                for line in table:
                    print(f"  {line}")
                print(f"amps:")
                for key, value in amps.items():
                    value=complex(value)
                    print(f"  {key}: {value.real:+f} {value.imag:+f} i")

                read_specified_label(process.stdout, "state")
                read_specified_label(process.stdout, "shot 0")
                error = read_error(process.stdout)
                table2, amps2 = read_shot_state_content(process.stdout, args)

                print(f"table2:")
                for line in table2:
                    print(f"  {line}")
                print(f"amps2:")
                for key, value in amps2.items():
                    value = complex(value)
                    print(f"  {key}: {value.real:+f} {value.imag:+f} i")

                if error:
                    error = ValueError(f"Found error: {error}")
                    errors.append(error)
                    break

                if table != table2:
                    error = ValueError(f"Found differences in table: \n{table} != {table2}")
                    errors.append(error)
                    break

                for key in set(amps.keys()) | set(amps2.keys()):
                    value1 = amps.get(key, 0)
                    value2 = amps2.get(key, 0)
                    if not np.allclose(value1, value2, rtol=1e-05, atol=1e-05):
                        error = ValueError(f"Found differences in amps[{key}]: \n{value1} != {value2}")
                        errors.append(error)
                        break
                if errors:
                    break

                print("verified")

            if not exhausted:
                while queue.get():
                    pass

            process.terminate()
            process.wait()

        thread = Thread(target=thread_func, daemon=True)
        thread.start()

        last_gate = None
        while True:
            try:
                match read_label(fp):
                    case "gate", gate:
                        process.stdin.write(gate)
                        process.stdin.write("\n")
                        last_gate = gate
                    case "state", _:
                        table, amps = read_shot_state_content(fp, args)
                        process.stdin.write("STATE")
                        process.stdin.write("\n")
                        queue.put((last_gate, table, amps))
                        last_gate = None
                process.stdin.flush()
            except StopIteration:
                break

        queue.put(None)
        thread.join()

        if errors:
            raise errors[0]


if __name__ == '__main__':
    exec_file_path = os.path.join(project_dir_path, "cmake-build-release/stn_cuda_exec")
    logs_file_path = os.path.join(project_dir_path, "scripts/compare/test_0.logs.txt")
    main(
        exec_file_path=exec_file_path,
        logs_file_path=logs_file_path)
