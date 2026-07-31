from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
import os
import time
from typing import Iterable, Literal, Union

import numpy as np

from ._native import CpuCircuit, Simulator
from ._stim import compile_stim_file, compile_stim_text


Pathish = Union[str, bytes, PathLike[str], PathLike[bytes]]
PrintKind = Literal["INT", "FLT", "ERR", "ENTRIES_N", "STATE"]
CircuitFormat = Literal["auto", "soft", "stim"]


@dataclass(frozen=True)
class SimulatorArgs:
    shot_i: int = 0
    shots_n: int = 1
    qubits_n: int = 4
    entries_m: int = 16
    mem_ints_n: int = 16
    mem_flts_n: int = 16
    epsilon: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class PrintRecord:
    kind: PrintKind
    value: object


@dataclass(frozen=True)
class RunResult:
    simulator: Simulator
    prints: tuple[PrintRecord, ...]

    @property
    def work_int(self):
        return self.simulator.work_int()

    @property
    def work_float(self):
        return self.simulator.work_float()

    @property
    def errors(self):
        return self.simulator.errors()

    @property
    def entries_n(self):
        return self.simulator.entries_n()


@dataclass(frozen=True)
class _CircuitStaticStats:
    qubits_n: int
    mem_ints_n: int


class Circuit:
    """A SOFT operation program using the same line format as the CLI."""

    def __init__(
        self,
        text: Union[str, bytes, None] = None,
        *,
        path: Pathish | None = None,
        format: CircuitFormat = "auto",
        append_outputs: bool = True,
    ) -> None:
        if text is not None and path is not None:
            raise TypeError("pass either text or path, not both")
        if format not in {"auto", "soft", "stim"}:
            raise ValueError("format must be auto, soft, or stim")
        cpu_text: str | None = None
        cpu_path: Pathish | None = None
        observable_indices: tuple[int, ...] | None = None
        inferred_qubits_n: int | None = None
        inferred_mem_ints_n: int | None = None
        if path is not None:
            if format == "stim" or (format == "auto" and _is_stim_path(path)):
                cpu_path = path
                compiled = compile_stim_file(path, append_outputs=append_outputs)
                text = compiled.text
                observable_indices = compiled.observable_indices
                inferred_qubits_n = compiled.stats.qubits_n
                inferred_mem_ints_n = compiled.stats.mem_ints_n
            else:
                with open(path, "rb") as file:
                    text = file.read()
        if text is None:
            text = ""
        if isinstance(text, bytes):
            text = text.decode()
        if not isinstance(text, str):
            raise TypeError("text must be str or bytes")
        if path is None and format == "stim":
            cpu_text = text
            compiled = compile_stim_text(text, append_outputs=append_outputs)
            text = compiled.text
            observable_indices = compiled.observable_indices
            inferred_qubits_n = compiled.stats.qubits_n
            inferred_mem_ints_n = compiled.stats.mem_ints_n
        if inferred_qubits_n is None or inferred_mem_ints_n is None:
            stats = _infer_static_stats(text)
            if inferred_qubits_n is None:
                inferred_qubits_n = stats.qubits_n
            if inferred_mem_ints_n is None:
                inferred_mem_ints_n = stats.mem_ints_n
        self._text = text
        self._observable_indices = observable_indices
        self._inferred_qubits_n = inferred_qubits_n
        self._inferred_mem_ints_n = inferred_mem_ints_n
        self._cpu_text = cpu_text
        self._cpu_path = cpu_path

    @property
    def text(self) -> str:
        return self._text

    @property
    def operations(self) -> tuple[str, ...]:
        return tuple(line.strip() for line in self._text.splitlines() if line.strip())

    @property
    def observable_indices(self) -> tuple[int, ...] | None:
        return self._observable_indices

    @property
    def inferred_qubits_n(self) -> int | None:
        return self._inferred_qubits_n

    @property
    def inferred_mem_ints_n(self) -> int | None:
        return self._inferred_mem_ints_n

    def run(
        self,
        args: SimulatorArgs | None = None,
        *,
        simulator: Simulator | None = None,
        **kwargs,
    ) -> RunResult:
        return run(self, args=args, simulator=simulator, **kwargs)

    def sample_counts(
        self,
        args: SimulatorArgs | None = None,
        *,
        observable: int = 0,
        cuda: bool = False,
        **kwargs,
    ) -> dict[str, object]:
        return sample_counts(self, args=args, observable=observable, cuda=cuda, **kwargs)


def _is_stim_path(path: Pathish) -> bool:
    return os.fsdecode(os.fspath(path)).lower().endswith(".stim")


def read_file(
    path: Pathish,
    *,
    format: CircuitFormat = "auto",
    append_outputs: bool = True,
) -> Circuit:
    return Circuit(path=path, format=format, append_outputs=append_outputs)


def read_stim_file(path: Pathish, *, append_outputs: bool = True) -> Circuit:
    return read_file(path, format="stim", append_outputs=append_outputs)


def run_file(
    path: Pathish,
    args: SimulatorArgs | None = None,
    *,
    format: CircuitFormat = "auto",
    append_outputs: bool = True,
    **kwargs,
) -> RunResult:
    circuit = read_file(path, format=format, append_outputs=append_outputs)
    return run(circuit, args=args, **kwargs)


def sample_counts_file(
    path: Pathish,
    args: SimulatorArgs | None = None,
    *,
    format: CircuitFormat = "auto",
    observable: int = 0,
    cuda: bool = False,
    **kwargs,
) -> dict[str, object]:
    if not cuda and (format == "stim" or (format == "auto" and _is_stim_path(path))):
        return _sample_counts_cpu_source(path=path, args=args, observable=observable, **kwargs)

    sample_start = time.perf_counter()
    parse_start = time.perf_counter()
    circuit = read_file(path, format=format, append_outputs=True)
    parse_s = time.perf_counter() - parse_start
    return _sample_counts_circuit(
        circuit,
        args=args,
        observable=observable,
        parse_s=parse_s,
        sample_start=sample_start,
        **kwargs,
    )


def sample(circuit: Circuit | str | bytes, *args, **kwargs) -> RunResult:
    """SymFT-style convenience alias for :func:`run`."""
    return run(circuit, *args, **kwargs)


def sample_counts(
    circuit: Circuit | str | bytes,
    args: SimulatorArgs | None = None,
    *,
    observable: int = 0,
    cuda: bool = False,
    format: CircuitFormat = "auto",
    append_outputs: bool = True,
    **kwargs,
) -> dict[str, object]:
    if not cuda and not isinstance(circuit, Circuit) and format == "stim":
        return _sample_counts_cpu_source(text=circuit, args=args, observable=observable, **kwargs)
    if not isinstance(circuit, Circuit):
        circuit = Circuit(circuit, format=format, append_outputs=append_outputs)
    if not cuda:
        cpu_counts = _sample_counts_cpu_circuit(circuit, args=args, observable=observable, **kwargs)
        if cpu_counts is not None:
            return cpu_counts
    return _sample_counts_circuit(circuit, args=args, observable=observable, **kwargs)


def run(
    circuit: Circuit | str | bytes,
    args: SimulatorArgs | None = None,
    *,
    simulator: Simulator | None = None,
    **kwargs,
) -> RunResult:
    if not isinstance(circuit, Circuit):
        circuit = Circuit(circuit)
    if simulator is not None and (args is not None or kwargs):
        raise TypeError("simulator cannot be combined with simulator arguments")
    args = _prepare_simulator_args(circuit, args, kwargs)

    sim = simulator if simulator is not None else Simulator(**asdict(args))
    prints: list[PrintRecord] = []
    for line_no, line in enumerate(circuit.text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = _execute_line(sim, stripped)
        except Exception as exc:
            raise type(exc)(f"line {line_no}: {exc}") from exc
        if record is not None:
            prints.append(record)
    sim.synchronize()
    return RunResult(simulator=sim, prints=tuple(prints))


def _prepare_simulator_args(
    circuit: Circuit,
    args: SimulatorArgs | None,
    kwargs: dict[str, object],
) -> SimulatorArgs:
    if args is None:
        data = dict(kwargs)
    else:
        data = asdict(args)
        data.update(kwargs)
    _apply_static_capacity(circuit, data)
    return SimulatorArgs(**data)


def _apply_static_capacity(circuit: Circuit, data: dict[str, object]) -> None:
    if circuit.inferred_qubits_n is not None:
        inferred_qubits_n = max(1, circuit.inferred_qubits_n)
        data["qubits_n"] = max(inferred_qubits_n, _optional_int(data.get("qubits_n"), 0))
    if circuit.inferred_mem_ints_n is not None:
        data["mem_ints_n"] = max(circuit.inferred_mem_ints_n, _optional_int(data.get("mem_ints_n"), 0))


def _optional_int(value: object, default: int) -> int:
    if value is None:
        return default
    return int(value)

def _cpu_sample_counts_kwargs(
    args: SimulatorArgs | None,
    kwargs: dict[str, object],
) -> tuple[int, int]:
    data = dict(kwargs) if args is None else asdict(args) | dict(kwargs)
    shots = int(data.pop("shots", data.pop("shots_n", 1)))
    num_qubits = int(data.pop("num_qubits", data.pop("qubits_n", -1)))
    for ignored in ("shot_i", "entries_m", "mem_ints_n", "mem_flts_n", "epsilon", "seed"):
        data.pop(ignored, None)
    if data:
        key = next(iter(data))
        raise TypeError(f"unexpected CPU backend argument: {key}")
    return shots, num_qubits

def _sample_counts_cpu_source(
    *,
    text: str | bytes | None = None,
    path: Pathish | None = None,
    args: SimulatorArgs | None = None,
    observable: int = 0,
    **kwargs,
) -> dict[str, object]:
    shots, num_qubits = _cpu_sample_counts_kwargs(args, kwargs)
    sample_start = time.perf_counter()
    parse_start = time.perf_counter()
    if path is not None:
        circuit = CpuCircuit(path=path, num_qubits=num_qubits)
    else:
        circuit = CpuCircuit(text or b"", num_qubits=num_qubits)
    parse_s = time.perf_counter() - parse_start
    execute_start = time.perf_counter()
    counts = circuit.sample_counts(shots=shots, observable=observable)
    execute_s = time.perf_counter() - execute_start
    timing = dict(counts.get("timing", {}))
    timing["parse_s"] = parse_s
    timing["execute_s"] = float(timing.get("execute_s", execute_s)) or execute_s
    timing["sample_s"] = time.perf_counter() - sample_start
    counts["timing"] = timing
    return counts

def _sample_counts_cpu_circuit(
    circuit: Circuit,
    args: SimulatorArgs | None = None,
    *,
    observable: int = 0,
    **kwargs,
) -> dict[str, object] | None:
    cpu_path = getattr(circuit, "_cpu_path", None)
    cpu_text = getattr(circuit, "_cpu_text", None)
    if cpu_path is None and cpu_text is None:
        return None
    return _sample_counts_cpu_source(
        text=cpu_text,
        path=cpu_path,
        args=args,
        observable=observable,
        **kwargs,
    )

def _sample_counts_circuit(
    circuit: Circuit,
    args: SimulatorArgs | None = None,
    *,
    observable: int = 0,
    parse_s: float = 0.0,
    sample_start: float | None = None,
    **kwargs,
) -> dict[str, object]:
    if not isinstance(observable, int) or observable < 0:
        raise ValueError("observable must be a non-negative integer")
    if sample_start is None:
        sample_start = time.perf_counter()

    execute_start = time.perf_counter()
    result = run(circuit, args=args, **kwargs)
    execute_s = time.perf_counter() - execute_start

    accumulate_start = time.perf_counter()
    counts = _counts_from_result(result, observable, circuit.observable_indices)
    accumulate_s = time.perf_counter() - accumulate_start

    counts["backend"] = "cuda"
    counts["active_threads"] = 1
    counts["timing"] = {
        "parse_s": parse_s,
        "plan_s": 0.0,
        "presample_s": 0.0,
        "execute_s": execute_s,
        "accumulate_s": accumulate_s,
        "sample_s": time.perf_counter() - sample_start,
    }
    return counts


def _counts_from_result(
    result: RunResult,
    observable: int,
    observable_indices: tuple[int, ...] | None,
) -> dict[str, object]:
    errors = _first_print_value(result, "ERR")
    if errors is None:
        errors = result.errors
    errors_array = np.asarray(errors)
    if errors_array.ndim != 1:
        raise ValueError("ERR output must be a one-dimensional per-shot array")

    shots = int(errors_array.shape[0])
    accepted_mask = errors_array == 0
    accepted = int(np.count_nonzero(accepted_mask))
    discarded = shots - accepted

    parity = np.zeros(shots, dtype=np.int8)
    int_print_i = 0
    for record in result.prints:
        if record.kind != "INT":
            continue
        record_observable = 0
        if observable_indices is not None:
            if int_print_i >= len(observable_indices):
                raise ValueError("observable metadata does not match PRINT INT outputs")
            record_observable = observable_indices[int_print_i]
        int_print_i += 1
        if record_observable != observable:
            continue
        values = np.asarray(record.value)
        if values.shape != errors_array.shape:
            raise ValueError("observable output shape does not match ERR output shape")
        parity ^= (values.astype(np.int64) & 1).astype(np.int8)

    logical_errors = int(np.count_nonzero(accepted_mask & (parity != 0)))
    return {
        "shots": shots,
        "discarded": discarded,
        "accepted": accepted,
        "logical_errors": logical_errors,
        "discard_rate": _safe_ratio(discarded, shots),
        "logical_error_rate": _safe_ratio(logical_errors, accepted),
    }


def _first_print_value(result: RunResult, kind: PrintKind) -> object | None:
    for record in result.prints:
        if record.kind == kind:
            return record.value
    return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def _expect(tokens: list[str], count: int, name: str) -> None:
    if len(tokens) != count:
        got = len(tokens) - 1
        expected = count - 1
        raise ValueError(f"{name} expects {expected} arguments, got {got}")


def _int(token: str) -> int:
    return int(token, 0)


def _float(token: str) -> float:
    return float(token)


def _split_match(items: Iterable[str]) -> tuple[list[int], list[int]]:
    values = [_int(item) for item in items]
    if len(values) % 2 != 0:
        raise ValueError(f"MATCH expects 2*n arguments, got {len(values)}")
    mid = len(values) // 2
    return values[:mid], values[mid:]


def _infer_static_stats(text: str) -> _CircuitStaticStats:
    qubits_n = 0
    mem_ints_n = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            line_stats = _infer_line_static_stats(stripped)
        except (TypeError, ValueError, OverflowError):
            continue
        qubits_n = max(qubits_n, line_stats.qubits_n)
        mem_ints_n = max(mem_ints_n, line_stats.mem_ints_n)
    return _CircuitStaticStats(qubits_n=qubits_n, mem_ints_n=mem_ints_n)


def _infer_line_static_stats(line: str) -> _CircuitStaticStats:
    tokens = line.split()
    if not tokens:
        return _CircuitStaticStats(qubits_n=0, mem_ints_n=0)

    name = tokens[0].upper()
    qubits_n = 0
    mem_ints_n = 0

    def include_qubit(token: str) -> None:
        nonlocal qubits_n
        qubit = _int(token)
        if qubit >= 0:
            qubits_n = max(qubits_n, qubit + 1)

    def include_mem_int_value(pointer: int) -> None:
        nonlocal mem_ints_n
        if pointer >= 0:
            mem_ints_n = max(mem_ints_n, pointer + 1)

    def include_mem_int(token: str) -> None:
        include_mem_int_value(_int(token))

    if name in {"X", "Y", "Z", "H", "S", "SDG", "T", "TDG", "MEASURE", "CCX", "CCY", "CCZ"}:
        _expect(tokens, 2, name)
        include_qubit(tokens[1])
    elif name == "CX":
        _expect(tokens, 3, name)
        include_qubit(tokens[1])
        include_qubit(tokens[2])
    elif name in {"DESIRE", "RESET"}:
        _expect(tokens, 3, name)
        include_qubit(tokens[1])
    elif name in {"XERR", "ZERR", "DEP1"}:
        _expect(tokens, 3, name)
        include_qubit(tokens[2])
    elif name == "DEP2":
        _expect(tokens, 4, name)
        include_qubit(tokens[2])
        include_qubit(tokens[3])
    elif name in {"OR", "XOR", "AND"}:
        for token in tokens[1:]:
            include_mem_int(token)
    elif name == "MATCH":
        pointers, _ = _split_match(tokens[1:])
        for pointer in pointers:
            include_mem_int_value(pointer)
    elif name in {"LOAD", "SAVE"}:
        _expect(tokens, 3, name)
        if tokens[1].upper() == "INT":
            include_mem_int(tokens[2])

    return _CircuitStaticStats(qubits_n=qubits_n, mem_ints_n=mem_ints_n)


def _execute_line(sim: Simulator, line: str) -> PrintRecord | None:
    tokens = line.split()
    if not tokens:
        return None
    name = tokens[0].upper()

    if name in {"X", "Y", "Z", "H", "S", "SDG", "T", "TDG"}:
        _expect(tokens, 2, name)
        getattr(sim, f"apply_{name.lower()}")(_int(tokens[1]))
        return None
    if name == "CX":
        _expect(tokens, 3, name)
        sim.apply_cx(_int(tokens[1]), _int(tokens[2]))
        return None

    if name == "MEASURE":
        _expect(tokens, 2, name)
        sim.apply_measure(_int(tokens[1]))
        return None
    if name == "DESIRE":
        _expect(tokens, 3, name)
        sim.apply_desire(_int(tokens[1]), _int(tokens[2]))
        return None
    if name == "RESET":
        _expect(tokens, 3, name)
        sim.apply_reset(_int(tokens[1]), _int(tokens[2]))
        return None

    if name == "XERR":
        _expect(tokens, 3, name)
        sim.apply_noise_x(_float(tokens[1]), _int(tokens[2]))
        return None
    if name == "ZERR":
        _expect(tokens, 3, name)
        sim.apply_noise_z(_float(tokens[1]), _int(tokens[2]))
        return None
    if name == "DEP1":
        _expect(tokens, 3, name)
        sim.apply_noise_depo1(_float(tokens[1]), _int(tokens[2]))
        return None
    if name == "DEP2":
        _expect(tokens, 4, name)
        sim.apply_noise_depo2(_float(tokens[1]), _int(tokens[2]), _int(tokens[3]))
        return None

    if name == "FLIP":
        _expect(tokens, 1, name)
        sim.apply_classical_flip()
        return None
    if name == "RANDFLIP":
        _expect(tokens, 2, name)
        sim.apply_classical_random_flip(_float(tokens[1]))
        return None
    if name == "CHECK":
        _expect(tokens, 2, name)
        sim.apply_classical_check(_int(tokens[1]))
        return None
    if name == "OR":
        sim.apply_classical_or([_int(token) for token in tokens[1:]])
        return None
    if name == "XOR":
        sim.apply_classical_xor([_int(token) for token in tokens[1:]])
        return None
    if name == "AND":
        sim.apply_classical_and([_int(token) for token in tokens[1:]])
        return None
    if name == "MATCH":
        pointers, values = _split_match(tokens[1:])
        sim.apply_classical_match(pointers, values)
        return None

    if name in {"CCX", "CCY", "CCZ"}:
        _expect(tokens, 2, name)
        getattr(sim, f"apply_classical_controlled_{name[-1].lower()}")(_int(tokens[1]))
        return None

    if name == "LOAD":
        _expect(tokens, 3, name)
        object_name = tokens[1].upper()
        if object_name == "INT":
            sim.apply_classical_load_int(_int(tokens[2]))
            return None
        if object_name == "FLT":
            sim.apply_classical_load_flt(_int(tokens[2]))
            return None
        raise ValueError(f"unknown load object: {tokens[1]}")

    if name == "SAVE":
        _expect(tokens, 3, name)
        object_name = tokens[1].upper()
        if object_name == "INT":
            sim.apply_classical_save_int(_int(tokens[2]))
            return None
        if object_name == "FLT":
            sim.apply_classical_save_flt(_int(tokens[2]))
            return None
        raise ValueError(f"unknown save object: {tokens[1]}")

    if name == "PRINT":
        _expect(tokens, 2, name)
        object_name = tokens[1].upper()
        if object_name == "INT":
            return PrintRecord("INT", sim.work_int())
        if object_name == "FLT":
            return PrintRecord("FLT", sim.work_float())
        if object_name == "ERR":
            return PrintRecord("ERR", sim.errors())
        if object_name == "ENTRIES_N":
            return PrintRecord("ENTRIES_N", sim.entries_n())
        if object_name == "STATE":
            return PrintRecord("STATE", sim.state_text())
        raise ValueError(f"unknown print object: {tokens[1]}")

    raise ValueError(f"unknown op: {tokens[0]}")

