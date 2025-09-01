from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from operator import eq


@dataclass(kw_only=True)
class Args:
    shots_n: int = 1
    qubits_n: int
    entries_m: int
    mem_ints_m: int = 0
    mem_flts_m: int = 0
    seed: int | None = None


def make_cmd(exec_file_path: str, args: Args) -> tuple[str, ...]:
    return (
        exec_file_path,
        "--shots_n", str(args.shots_n),
        "--qubits_n", str(args.qubits_n),
        "--entries_m", str(args.entries_m),
        "--mem_ints_m", str(args.mem_ints_m),
        "--mem_flts_m", str(args.mem_flts_m),
        "--seed", str(0 if args.seed is None else args.seed))


def read_nonempty_line(lines: Iterator[str]) -> str:
    while True:
        line = next(lines)
        line = line.strip()
        if line == "":
            continue
        break
    return line


def read_list_item(lines: Iterator[str]) -> str:
    line = read_nonempty_line(lines)
    line = line.strip()
    if not line.startswith("- "):
        raise ValueError
    item = line[len("- "):]
    return item


def read_dict_key(lines: Iterator[str]) -> str:
    line = read_nonempty_line(lines)
    line = line.strip()
    if not line.endswith(":"):
        raise ValueError
    key = line[:-len(":")]
    return key


def read_dict_key_value(lines: Iterator[str]) -> tuple[str, str]:
    line = read_nonempty_line(lines)
    line = line.strip()
    colon = line.index(":")
    key = line[:colon]
    key = key.strip()
    value = line[colon + len(":"):]
    value = value.strip()
    return key, value


def read_dict_key_and_check(lines: Iterator[str], expected_key: str | Callable[[str], bool]):
    key = read_dict_key(lines)
    if not isinstance(expected_key, Callable):
        expected_key = partial(eq, expected_key)
    if not expected_key(key):
        raise ValueError


def read_dict_key_value_and_check(lines: Iterator[str], expected_key: str | Callable[[str], bool]):
    key, value = read_dict_key_value(lines)
    if not isinstance(expected_key, Callable):
        expected_key = partial(eq, expected_key)
    if not expected_key(key):
        raise ValueError
    return value


def read_args(lines: Iterator[str]) -> Args:
    read_dict_key_and_check(lines, "args")
    shots_n = read_dict_key_value_and_check(lines, "shots_n")
    qubits_n = read_dict_key_value_and_check(lines, "qubits_n")
    entries_m = read_dict_key_value_and_check(lines, "entries_m")
    mem_ints_m = read_dict_key_value_and_check(lines, "mem_ints_m")
    mem_flts_m = read_dict_key_value_and_check(lines, "mem_flts_m")
    seed = read_dict_key_value_and_check(lines, "seed")
    return Args(
        shots_n=int(shots_n),
        qubits_n=int(qubits_n),
        entries_m=int(entries_m),
        mem_ints_m=int(mem_ints_m),
        mem_flts_m=int(mem_flts_m),
        seed=int(seed))


def read_table(lines: Iterator[str], args: Args) -> tuple[str, ...]:
    read_dict_key_value_and_check(lines, "table")
    return tuple(next(lines).strip() for _ in range(args.qubits_n * 2))


def read_entries(lines: Iterator[str]) -> tuple[tuple[tuple[bool, ...], complex], ...]:
    read_dict_key_and_check(lines, "entries")

    entries_n = read_dict_key_value_and_check(lines, "entries_n")
    entries_n = int(entries_n)

    entries = []
    for _ in range(entries_n):
        bst, amp = read_dict_key_value(lines)
        bst = bst.strip()
        bst = tuple(bool(int(b)) for b in bst)
        amp = amp.strip()
        amp = amp.replace("i", "j")
        amp = complex(amp)
        entries.append((bst, amp))
    return tuple(entries)


def read_shot_state(lines: Iterator[str], args: Args):
    read_dict_key_and_check(lines, lambda key: key.startswith("shot_"))
    error = read_dict_key_value_and_check(lines, "error")
    error = int(error)
    if error != 0:
        return error, None, None
    table = read_table(lines, args)
    entries = read_entries(lines)
    return 0, table, entries


def read_printed_shots_state(lines: Iterator[str], args: Args):
    read_dict_key_and_check(lines, lambda key: key.startswith("print_"))
    return tuple(read_shot_state(lines, args) for _ in range(args.shots_n))


def read_shot_int(lines: Iterator[str]):
    value = read_dict_key_value_and_check(lines, lambda key: key.startswith("shot_"))
    value = int(value)
    return value


def read_shot_flt(lines: Iterator[str]):
    value = read_dict_key_value_and_check(lines, lambda key: key.startswith("shot_"))
    value = float(value)
    return value


def read_shot_err(lines: Iterator[str]):
    error = read_dict_key_value_and_check(lines, lambda key: key.startswith("shot_"))
    error = int(error)
    return error


def read_printed_shots_int(lines: Iterator[str], args: Args):
    read_dict_key_and_check(lines, lambda key: key.startswith("print_"))
    return tuple(read_shot_int(lines) for _ in range(args.shots_n))


def read_printed_shots_flt(lines: Iterator[str], args: Args):
    read_dict_key_and_check(lines, lambda key: key.startswith("print_"))
    return tuple(read_shot_flt(lines) for _ in range(args.shots_n))


def read_printed_shots_err(lines: Iterator[str], args: Args):
    read_dict_key_and_check(lines, lambda key: key.startswith("print_"))
    return tuple(read_shot_err(lines) for _ in range(args.shots_n))
