import os
import subprocess
import sys
from collections.abc import Iterator
from functools import partial

import fire
import numpy as np
import pyzstd

project_dir_path = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(project_dir_path)

from scripts.utils.jobs import JobsQueueExecutor
from scripts.utils.stn import Args, make_cmd, read_args, read_dict_key_value, read_entries, read_printed_shots_flt, \
    read_printed_shots_state, read_table


def print_operation(gate: str, steps_i: int):
    print(f"[{steps_i=}] applied {gate}")


def read_and_compare_prob(lines: Iterator[str], args: Args, steps_i: int, prob_expect: float):
    prob_actual = read_printed_shots_flt(lines, args)[0]
    if not np.allclose(prob_expect, prob_actual, rtol=1e-06, atol=1e-06):
        print(f"prob (expect): {prob_expect}")
        print(f"prob (actual): {prob_actual}")
        raise ValueError(f"Found differences in prob! ({steps_i=})")

    print(f"[{steps_i=}] verified prob")
    # print(f"prob: {prob}")


def read_and_compare_state(lines: Iterator[str], args: Args, steps_i: int, state_expect: tuple):
    state_actual = read_printed_shots_state(lines, args)[0]

    table_expect, entries_expect = state_expect
    error_actual, table_actual, entries_actual = state_actual
    if error_actual:
        raise ValueError(f"Found error {error_actual}!")
    entries_expect = {"".join(str(int(bit)) for bit in bst): amp for bst, amp in entries_expect}
    entries_actual = {"".join(str(int(bit)) for bit in bst): amp for bst, amp in entries_actual}

    if table_expect != table_actual:
        print(f"table (expect):")
        for line in table_expect:
            print(f"  {line}")
        print(f"table (actual):")
        for line in table_actual:
            print(f"  {line}")
        raise ValueError(f"Found differences in table! ({steps_i=})")

    print(f"[{steps_i=}] verified table")
    # print(f"table:")
    # for line in table:
    #     print(f"  {line}")

    for key in set(entries_expect.keys()) | set(entries_actual.keys()):
        value1 = entries_expect.get(key, 0)
        value2 = entries_actual.get(key, 0)
        if not np.allclose(value1, value2, rtol=1e-06, atol=1e-06):
            print(f"entries (expect):")
            for key in sorted(entries_expect.keys()):
                value = entries_expect[key]
                value = complex(value)
                if abs(value) > 1e-6:
                    print(f"  {key}: {value.real:+f}{value.imag:+f}i")
            print(f"entries (actual):")
            for key in sorted(entries_actual.keys()):
                value = entries_actual[key]
                value = complex(value)
                if abs(value) > 1e-6:
                    print(f"  {key}: {value.real:+f}{value.imag:+f}i")
            raise ValueError(f"Found differences in entries! ({steps_i=}, {key=})")

    print(f"[{steps_i=}] verified entries")
    # print(f"entries:")
    # for key in sorted(entries.keys()):
    #     value = entries[key]
    #     value = complex(value)
    #     if abs(value) > 1e-6:
    #         print(f"  {key}: {value.real:+f} {value.imag:+f} i")


def main(
    exec_file_path: str,
    logs_file_path: str,
):
    open_func = open
    if logs_file_path.endswith(".zst"):
        open_func = pyzstd.open

    with open_func(logs_file_path, 'rt') as lines, JobsQueueExecutor() as executor:
        args = read_args(lines)
        cmd = make_cmd(exec_file_path, args)

        print(f"{cmd=}")
        process = subprocess.Popen(cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True)

        steps_n = 0
        while True:
            try:
                match read_dict_key_value(lines):
                    case "gate", gate:
                        process.stdin.write(gate)
                        process.stdin.write("\n")
                        steps_n += 1
                        executor.append(partial(print_operation, gate, steps_n))
                    case "prob", prob:
                        prob = float(prob)
                        process.stdin.write("PRINT FLT")
                        process.stdin.write("\n")
                        executor.append(partial(read_and_compare_prob,
                            process.stdout, args, steps_n, prob))
                    case "state", _:
                        table = read_table(lines, args)
                        entries = read_entries(lines)
                        process.stdin.write("PRINT STATE")
                        process.stdin.write("\n")
                        executor.append(partial(read_and_compare_state,
                            process.stdout, args, steps_n, (table, entries)))
                process.stdin.flush()
            except StopIteration:
                break

    process.terminate()
    process.wait()
    print("Finished")


if __name__ == '__main__':
    fire.Fire(main)
