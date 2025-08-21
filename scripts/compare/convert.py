import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator
from typing import IO, TypeVar

import numpy as np
from tqdm import tqdm

project_dir_path = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(project_dir_path)

AnyItem = TypeVar('AnyItem')


def make_iterator_iterator(
    items: Iterable[AnyItem],
    separate: Callable[[AnyItem], bool],
) -> Iterator[Iterator[AnyItem]]:
    iterator = iter(items)
    separated = True
    exhausted = False

    def iterate_part():
        nonlocal separated, exhausted
        for item in iterator:
            if separate(item):
                separated = True
                break
            yield item
        else:
            exhausted = True

    while not exhausted:
        if not separated:
            raise ValueError("You must empty the last iterator!")
        separated = False
        yield iterate_part()


def iter_shots_lines(src_file: IO[str]):
    sep_pattern = re.compile(r"==========shot (\d+)==========\n")
    sep_func = lambda line: bool(sep_pattern.match(line))
    shots_lines = make_iterator_iterator(src_file, sep_func)
    for _ in next(shots_lines):
        pass  # skip first part
    yield from shots_lines


def iter_convert_lines(lines: Iterable[str]) -> Iterator[str]:
    qubits_n = 42

    yield "args:"
    yield f"  qubits_n={qubits_n}"
    yield "  entries_m=4096"
    yield "  results_m=1024"
    yield ""

    lines = iter(lines)
    while True:
        try:
            gate = next(lines)
        except StopIteration:
            break
        gate = gate.strip()

        if gate.startswith("DETECTOR") or gate.startswith("OBSERVABLE"):
            discarded = next(lines)
            discarded = discarded.strip()
            discarded = discarded == "discard"
            if discarded:
                for _ in lines:
                    pass
                break
            else:
                next(lines)
                next(lines)
                next(lines)
            continue
        elif (gate.startswith("OBSERVABLE") or
              gate.startswith("DEPOLARIZE") or
              gate.startswith("X_ERROR") or
              gate.startswith("Z_ERROR")):
            next(lines)
            next(lines)
            next(lines)
            next(lines)
            continue
        elif gate.startswith("M"):
            target = gate[gate.index(" ") + 1:]
            result = next(lines)
            result = result.strip()
            if not result.startswith("result: "):
                raise ValueError(f"Unexpected line {result}")
            result = result[len("result: "):]
            result_pattern = re.compile(r"'reg': ([01])")
            match = result_pattern.search(result)
            if not match:
                raise ValueError(f"Unexpected result {result}")
            result = int(match.group(1))
            yield f"gate: D {target} {result}"
        elif gate.startswith("R "):
            target = gate[gate.index(" ") + 1:]
            result = next(lines)
            result = result.strip()
            if not result.startswith("result: "):
                raise ValueError(f"Unexpected line {result}")
            result = result[len("result: "):]
            result = int(result)
            yield f"gate: D {target} {result}"
            if result:
                yield f"gate: X {target}"
        else:
            yield "gate: " + gate

        yield "state:"

        yield "  table:"
        next(lines)  # table header
        table = next(lines)
        table = table.strip()
        table = eval(table)
        table = np.array(table, dtype=np.uint8)
        table = np.unpackbits(table, count=84 * 85)
        table = table.reshape(84, 85).astype(bool)
        for row in table:
            sign = "-" if row[-1] else "+"
            letters = []
            for x, z in zip(row[:42], row[42:84]):
                if (x, z) == (True, True):
                    letters.append("Y")
                if (x, z) == (True, False):
                    letters.append("X")
                if (x, z) == (False, True):
                    letters.append("Z")
                if (x, z) == (False, False):
                    letters.append("I")
            yield "    " + sign + "".join(letters)

        yield "  entries:"
        next(lines)  # map header
        entries = next(lines)
        entries = entries.strip()
        entries = entries.replace("np.complex128", "")
        entries = eval(entries)
        yield f"    entries_n={len(entries)}"
        for bst, amp in entries.items():
            bst = int(bst)
            amp = complex(amp)
            yield f"    {bst:042b} : {amp.real:+f} {amp.imag:+f} i"


def main(
    src_file_path: str,
    dst_file_path_func: Callable[[int], str],
):
    with open(src_file_path, "rt") as src_file:
        shots_lines = iter_shots_lines(src_file)
        for shot_i, shot_lines in enumerate(shots_lines):
            dst_file_path = dst_file_path_func(shot_i)
            with open(dst_file_path, "wt") as dst_file:
                shot_lines = tqdm(shot_lines)
                for line in iter_convert_lines(shot_lines):
                    dst_file.write(line)
                    dst_file.write("\n")
            print(f"extracted {dst_file_path}")


if __name__ == '__main__':
    src_file_path = os.path.join(project_dir_path, "scripts/compare/test.txt")
    dst_file_path_func = lambda shot_i: os.path.join(project_dir_path,f"scripts/compare/test_{shot_i}.logs.txt")
    main(
        src_file_path=src_file_path,
        dst_file_path_func=dst_file_path_func)
