from typing import Iterable

import numpy as np

from scripts.verify.utils_str import split_and_clean_lines

Op = tuple[str, int | tuple[int, int]]


def generate_random_op(qubits_n: int, rng: np.random.Generator) -> Op:
    types = ['X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG', 'CX', 'M']
    if qubits_n < 2:
        types.remove('CX')

    typ = types[rng.choice(len(types))]
    if typ in ['X', 'Y', 'Z', 'H', 'S', 'SDG', 'T', 'TDG', 'M']:
        target = rng.choice(qubits_n)
        return typ, target
    if typ in ['CX']:
        control, target = rng.choice(qubits_n, 2, replace=False)
        control, target = int(control), int(target)
        return typ, (control, target)
    raise TypeError


def iter_generate_random_ops(ops_n: int, qubits_n: int, rng: np.random.Generator) -> Iterable[Op]:
    for _ in range(ops_n):
        yield generate_random_op(qubits_n, rng)


def generate_random_ops(ops_n: int, qubits_n: int, rng: np.random.Generator) -> tuple[Op, ...]:
    return tuple(iter_generate_random_ops(ops_n, qubits_n, rng))


def iter_parse_ops(s: str) -> Iterable[Op]:
    lines = split_and_clean_lines(s)
    for line in lines:
        words = line.split(' ')
        match words:
            case (typ, target):
                yield typ, int(target)
            case (typ, control, target):
                yield typ, (int(control), int(target))
            case _:
                raise ValueError(f"Invalid line: {line}")


def parse_ops(s: str) -> tuple[Op, ...]:
    return tuple(iter_parse_ops(s))


def print_ops(ops: Iterable[Op], results: Iterable[int] | None = None):
    results = iter(results) if results is not None else None
    for i, (typ, args) in enumerate(ops):
        if typ == 'CX':
            print(f"{typ} {args[0]} {args[1]}")
            continue
        if typ == "M" and results is not None:
            result = next(results)
            print(f"D {args} {result}")
        else:
            print(f"{typ} {args}")
