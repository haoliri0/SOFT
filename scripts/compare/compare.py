import os
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread

import numpy as np
from tqdm import tqdm

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


def read_entries_content(lines: Iterator[str]) -> dict[str, complex]:
    line = read_nonempty_line(lines)
    entries_n_pattern = re.compile(f"entries_n=(.*)")
    entries_n_str = entries_n_pattern.match(line).group(1)
    entries_n = int(entries_n_str)

    entries = {}
    entry_pattern = re.compile(f"(.*):(.*)")
    for _ in range(entries_n):
        line = read_nonempty_line(lines)
        match = entry_pattern.match(line)
        bst = match.group(1)
        bst = bst.strip()
        amp = match.group(2)
        amp = amp.replace("i", "j")
        amp = amp.replace(" ", "")
        amp = complex(amp)
        entries[bst] = amp
    return entries


def read_shot_state_content(lines: Iterator[str], args: StnArgs):
    read_specified_label(lines, "table")
    table = read_table_content(lines, args)
    read_specified_label(lines, "entries")
    entries = read_entries_content(lines)
    return table, entries


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
                match queue.get():
                    case None:
                        exhausted = True
                        break
                    case 'gate', gate:
                        print(f"gate: {gate}")
                    case 'prob', prob:
                        prob = prob.strip()
                        prob = float(prob)

                        read_specified_label(process.stdout, "result")
                        read_specified_label(process.stdout, "shot_0")
                        prob2 = read_specified_label(process.stdout, "prob")
                        prob2 = prob2.strip()
                        prob2 = float(prob2)
                        read_specified_label(process.stdout, "value")

                        if not np.allclose(prob, prob2, rtol=1e-05, atol=1e-05):
                            error = ValueError(f"Found differences in prob.")
                            errors.append(error)
                            print(f"prob (expected): {prob}")
                            print(f"prob (actual): {prob2}")
                            break

                        print(f"prob: {prob}")

                    case 'state', table, entries:
                        read_specified_label(process.stdout, "state")
                        read_specified_label(process.stdout, "shot 0")
                        error = read_error(process.stdout)
                        table2, entries2 = read_shot_state_content(process.stdout, args)

                        if error:
                            error = ValueError(f"Found error: {error}")
                            errors.append(error)
                            break

                        if table != table2:
                            error = ValueError(f"Found differences in table.")
                            errors.append(error)
                            print(f"table (expected):")
                            for line in table:
                                print(f"  {line}")
                            print(f"table (actual):")
                            for line in table2:
                                print(f"  {line}")
                            break

                        print(f"table:")
                        for line in table:
                            print(f"  {line}")

                        for key in set(entries.keys()) | set(entries2.keys()):
                            value1 = entries.get(key, 0)
                            value2 = entries2.get(key, 0)
                            if not np.allclose(value1, value2, rtol=1e-05, atol=1e-05):
                                error = ValueError(f"Found differences in entries ({key=}).")
                                errors.append(error)
                                break
                        if errors:
                            print(f"entries (expected):")
                            for key in sorted(entries.keys()):
                                value = entries[key]
                                value = complex(value)
                                if abs(value) > 1e-6:
                                    print(f"  {key}: {value.real:+f} {value.imag:+f} i")
                            print(f"entries (actual):")
                            for key in sorted(entries2.keys()):
                                value = entries2[key]
                                value = complex(value)
                                if abs(value) > 1e-6:
                                    print(f"  {key}: {value.real:+f} {value.imag:+f} i")
                            break

                        print(f"entries:")
                        for key in sorted(entries.keys()):
                            value = entries[key]
                            value = complex(value)
                            if abs(value) > 1e-6:
                                print(f"  {key}: {value.real:+f} {value.imag:+f} i")

            if not exhausted:
                while queue.get():
                    pass

            process.terminate()
            process.wait()

        thread = Thread(target=thread_func, daemon=True)
        thread.start()

        while True:
            try:
                match read_label(fp):
                    case "gate", gate:
                        process.stdin.write(gate)
                        process.stdin.write("\n")
                        queue.put(('gate', gate))
                    case "prob", prob:
                        process.stdin.write("RESULT")
                        process.stdin.write("\n")
                        queue.put(('prob', prob))
                    case "state", _:
                        table, entries = read_shot_state_content(fp, args)
                        process.stdin.write("STATE")
                        process.stdin.write("\n")
                        queue.put(('state', table, entries))
                process.stdin.flush()
            except StopIteration:
                break

        queue.put(None)
        thread.join()

        if errors:
            raise errors[0]


if __name__ == '__main__':
    for i in tqdm(range(10)):
        exec_file_path = os.path.join(project_dir_path, "cmake-build-release/stn_cuda_exec")
        logs_file_path = os.path.join(project_dir_path, f"scripts/compare/test_{i}.logs.txt")
        main(
            exec_file_path=exec_file_path,
            logs_file_path=logs_file_path)
