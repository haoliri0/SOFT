from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
import re
from typing import Iterable, Literal, Sequence, Union


Pathish = Union[str, bytes, PathLike[str], PathLike[bytes]]
Pauli = Literal["X", "Y", "Z"]


class StimParseError(ValueError):
    """Raised when a Stim operation cannot be compiled to SOFT operations."""


@dataclass(frozen=True)
class StimCompileStats:
    qubits_n: int
    measurements_n: int
    mem_ints_n: int
    detectors_n: int
    observables_n: int


@dataclass(frozen=True)
class StimCompileResult:
    text: str
    observable_indices: tuple[int, ...]
    stats: StimCompileStats


@dataclass(frozen=True)
class _Line:
    text: str
    line_no: int


@dataclass(frozen=True)
class _Instruction:
    name: str
    args: tuple[str, ...]
    targets: tuple[str, ...]
    line_no: int


@dataclass(frozen=True)
class _QubitTarget:
    qubit: int
    inverted: bool = False


@dataclass(frozen=True)
class _RecordTarget:
    index: int
    inverted: bool = False


@dataclass(frozen=True)
class _PauliTarget:
    pauli: Pauli
    qubit: int
    inverted: bool = False


_INSTRUCTION_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\((.*)\))?$")
_RECORD_RE = re.compile(r"^(!)?rec\[(-?\d+)\]$")


def compile_stim_file(path: Pathish, *, append_outputs: bool = True) -> StimCompileResult:
    with open(path, "rb") as file:
        return compile_stim_text(file.read(), append_outputs=append_outputs)


def compile_stim_text(
    text: str | bytes,
    *,
    append_outputs: bool = True,
) -> StimCompileResult:
    if isinstance(text, bytes):
        text = text.decode()
    if not isinstance(text, str):
        raise TypeError("text must be str or bytes")
    compiler = _StimCompiler(append_outputs=append_outputs)
    compiler.compile(text.splitlines())
    return compiler.result()


def stim_to_soft(text: str | bytes, *, append_outputs: bool = True) -> str:
    return compile_stim_text(text, append_outputs=append_outputs).text


class _StimCompiler:
    def __init__(self, *, append_outputs: bool) -> None:
        self.append_outputs = append_outputs
        self.operations: list[str] = []
        self.qubits_n = 0
        self.measurements_n = 0
        self.detectors_n = 0
        self.observables: list[tuple[int, tuple[_RecordTarget, ...]]] = []

    def compile(self, raw_lines: Iterable[str]) -> None:
        lines = _logical_lines(raw_lines)
        index = self._compile_block(lines, 0, expect_closing=False)
        if index != len(lines):
            line = lines[index]
            raise StimParseError(f"line {line.line_no}: unexpected trailing input")
        if self.append_outputs:
            self._append_output_operations()

    def result(self) -> StimCompileResult:
        text = "\n".join(self.operations)
        if text:
            text += "\n"
        return StimCompileResult(
            text=text,
            observable_indices=tuple(index for index, _ in self.observables),
            stats=StimCompileStats(
                qubits_n=self.qubits_n,
                measurements_n=self.measurements_n,
                mem_ints_n=self.measurements_n,
                detectors_n=self.detectors_n,
                observables_n=len(self.observables),
            ),
        )

    def _compile_block(
        self,
        lines: Sequence[_Line],
        index: int,
        *,
        expect_closing: bool,
    ) -> int:
        while index < len(lines):
            line = lines[index]
            if line.text == "}":
                if expect_closing:
                    return index + 1
                raise StimParseError(f"line {line.line_no}: unexpected '}}'")

            repeat = _parse_repeat(line)
            if repeat is not None:
                count, has_open_brace = repeat
                if not has_open_brace:
                    index += 1
                    if index >= len(lines) or lines[index].text != "{":
                        raise StimParseError(f"line {line.line_no}: REPEAT expects '{{'")
                    index += 1
                else:
                    index += 1

                before = len(self.operations)
                index = self._compile_block(lines, index, expect_closing=True)
                block = self.operations[before:]
                del self.operations[before:]
                for _ in range(count):
                    self.operations.extend(block)
                continue

            if line.text == "{":
                raise StimParseError(f"line {line.line_no}: unexpected '{{'")

            self._compile_instruction(_parse_instruction(line))
            index += 1

        if expect_closing:
            raise StimParseError("missing closing '}'")
        return index

    def _compile_instruction(self, instruction: _Instruction) -> None:
        name = instruction.name

        if name == "QUBIT_COORDS":
            for target in self._qubit_targets(instruction.targets, instruction):
                if target.inverted:
                    raise self._error(instruction, "QUBIT_COORDS does not allow inverted targets")
            return
        if name in {"TICK", "SHIFT_COORDS"}:
            return

        if name in {"X", "Y", "Z", "H", "S", "T"}:
            self._compile_single_qubit_gate(name, instruction)
            return
        if name in {"S_DAG", "SQRT_Z_DAG", "SDG"}:
            self._compile_single_qubit_gate("SDG", instruction)
            return
        if name in {"T_DAG", "TDG"}:
            self._compile_single_qubit_gate("TDG", instruction)
            return
        if name == "SQRT_Z":
            self._compile_single_qubit_gate("S", instruction)
            return

        if name in {"CX", "CNOT"}:
            self._compile_controlled_gate("X", instruction)
            return
        if name == "CY":
            self._compile_controlled_gate("Y", instruction)
            return
        if name == "CZ":
            self._compile_controlled_gate("Z", instruction)
            return
        if name == "SWAP":
            self._compile_swap(instruction)
            return

        if name in {"R", "RZ"}:
            self._compile_reset("Z", instruction)
            return
        if name == "RX":
            self._compile_reset("X", instruction)
            return
        if name == "RY":
            self._compile_reset("Y", instruction)
            return

        if name in {"M", "MZ"}:
            self._compile_measure("Z", instruction, reset=False)
            return
        if name == "MX":
            self._compile_measure("X", instruction, reset=False)
            return
        if name == "MY":
            self._compile_measure("Y", instruction, reset=False)
            return
        if name in {"MR", "MRZ"}:
            self._compile_measure("Z", instruction, reset=True)
            return
        if name == "MRX":
            self._compile_measure("X", instruction, reset=True)
            return
        if name == "MRY":
            self._compile_measure("Y", instruction, reset=True)
            return

        if name == "X_ERROR":
            self._compile_single_qubit_noise("XERR", instruction)
            return
        if name == "Y_ERROR":
            self._compile_y_error(instruction)
            return
        if name == "Z_ERROR":
            self._compile_single_qubit_noise("ZERR", instruction)
            return
        if name == "DEPOLARIZE1":
            self._compile_single_qubit_noise("DEP1", instruction)
            return
        if name == "DEPOLARIZE2":
            self._compile_two_qubit_noise(instruction)
            return

        if name == "DETECTOR":
            self._compile_detector(instruction)
            return
        if name == "OBSERVABLE_INCLUDE":
            self._compile_observable(instruction)
            return
        if name == "MPP":
            self._compile_mpp(instruction)
            return

        raise self._error(instruction, f"unsupported Stim operation: {name}")

    def _compile_single_qubit_gate(self, soft_name: str, instruction: _Instruction) -> None:
        self._expect_no_args(instruction)
        for target in self._qubit_targets(instruction.targets, instruction):
            if target.inverted:
                raise self._error(instruction, f"{instruction.name} does not allow inverted targets")
            self._emit(soft_name, target.qubit)

    def _compile_controlled_gate(self, pauli: Pauli, instruction: _Instruction) -> None:
        self._expect_no_args(instruction)
        if len(instruction.targets) % 2 != 0:
            raise self._error(instruction, f"{instruction.name} expects an even number of targets")

        for control_token, target_token in _pairs(instruction.targets):
            control = self._control_target(control_token, instruction)
            target = self._qubit_target(target_token, instruction)
            if isinstance(control, _RecordTarget):
                self._emit("LOAD INT", control.index)
                if control.inverted:
                    self._emit("FLIP")
                self._emit(f"CC{pauli}", target.qubit)
                continue

            if control.inverted or target.inverted:
                raise self._error(instruction, f"{instruction.name} does not allow inverted qubit targets")
            if pauli == "X":
                self._emit("CX", control.qubit, target.qubit)
            elif pauli == "Y":
                self._emit("SDG", target.qubit)
                self._emit("CX", control.qubit, target.qubit)
                self._emit("S", target.qubit)
            elif pauli == "Z":
                self._emit("H", target.qubit)
                self._emit("CX", control.qubit, target.qubit)
                self._emit("H", target.qubit)

    def _compile_swap(self, instruction: _Instruction) -> None:
        self._expect_no_args(instruction)
        if len(instruction.targets) % 2 != 0:
            raise self._error(instruction, "SWAP expects an even number of targets")
        for token0, token1 in _pairs(instruction.targets):
            target0 = self._qubit_target(token0, instruction)
            target1 = self._qubit_target(token1, instruction)
            if target0.inverted or target1.inverted:
                raise self._error(instruction, "SWAP does not allow inverted targets")
            self._emit("CX", target0.qubit, target1.qubit)
            self._emit("CX", target1.qubit, target0.qubit)
            self._emit("CX", target0.qubit, target1.qubit)

    def _compile_reset(self, basis: Pauli, instruction: _Instruction) -> None:
        self._expect_no_args(instruction)
        for target in self._qubit_targets(instruction.targets, instruction):
            if target.inverted:
                raise self._error(instruction, f"{instruction.name} does not allow inverted targets")
            self._emit("RESET", target.qubit, 0)
            self._emit_basis_after(basis, target.qubit)

    def _compile_measure(self, basis: Pauli, instruction: _Instruction, *, reset: bool) -> None:
        probability = self._optional_probability(instruction)
        for target in self._qubit_targets(instruction.targets, instruction):
            self._emit_basis_before(basis, target.qubit)
            if reset:
                self._emit("RESET", target.qubit, 0)
            else:
                self._emit("MEASURE", target.qubit)
            if target.inverted:
                self._emit("FLIP")
            if probability is not None:
                self._emit("RANDFLIP", probability)
            self._save_measurement()
            self._emit_basis_after(basis, target.qubit)

    def _compile_single_qubit_noise(self, soft_name: str, instruction: _Instruction) -> None:
        probability = self._required_probability(instruction)
        for target in self._qubit_targets(instruction.targets, instruction):
            if target.inverted:
                raise self._error(instruction, f"{instruction.name} does not allow inverted targets")
            self._emit(soft_name, probability, target.qubit)

    def _compile_y_error(self, instruction: _Instruction) -> None:
        probability = self._required_probability(instruction)
        for target in self._qubit_targets(instruction.targets, instruction):
            if target.inverted:
                raise self._error(instruction, "Y_ERROR does not allow inverted targets")
            self._emit("XOR")
            self._emit("RANDFLIP", probability)
            self._emit("CCY", target.qubit)

    def _compile_two_qubit_noise(self, instruction: _Instruction) -> None:
        probability = self._required_probability(instruction)
        if len(instruction.targets) % 2 != 0:
            raise self._error(instruction, "DEPOLARIZE2 expects an even number of targets")
        for token0, token1 in _pairs(instruction.targets):
            target0 = self._qubit_target(token0, instruction)
            target1 = self._qubit_target(token1, instruction)
            if target0.inverted or target1.inverted:
                raise self._error(instruction, "DEPOLARIZE2 does not allow inverted targets")
            self._emit("DEP2", probability, target0.qubit, target1.qubit)

    def _compile_detector(self, instruction: _Instruction) -> None:
        records = self._record_targets(instruction.targets, instruction)
        self._emit_xor_records(records)
        self.detectors_n += 1
        self._emit("CHECK", self.detectors_n)

    def _compile_observable(self, instruction: _Instruction) -> None:
        if len(instruction.args) != 1:
            raise self._error(instruction, "OBSERVABLE_INCLUDE expects one observable index")
        try:
            index = int(instruction.args[0], 0)
        except ValueError as exc:
            raise self._error(instruction, "OBSERVABLE_INCLUDE index must be an integer") from exc
        if index < 0:
            raise self._error(instruction, "OBSERVABLE_INCLUDE index must be non-negative")
        records = self._record_targets(instruction.targets, instruction)
        self.observables.append((index, records))

    def _compile_mpp(self, instruction: _Instruction) -> None:
        probability = self._optional_probability(instruction)
        for token in instruction.targets:
            product = self._pauli_product(token, instruction)
            if not product:
                raise self._error(instruction, "MPP product cannot be empty")
            inverted = sum(target.inverted for target in product) % 2 == 1
            target = product[0].qubit

            for item in product:
                self._emit_basis_before(item.pauli, item.qubit)
            for item in product[1:]:
                self._emit("CX", item.qubit, target)
            self._emit("MEASURE", target)
            if inverted:
                self._emit("FLIP")
            if probability is not None:
                self._emit("RANDFLIP", probability)
            self._save_measurement()
            for item in product[1:]:
                self._emit("CX", item.qubit, target)
            for item in product:
                self._emit_basis_after(item.pauli, item.qubit)

    def _append_output_operations(self) -> None:
        self._emit("PRINT ERR")
        for _, records in self.observables:
            self._emit_xor_records(records)
            self._emit("PRINT INT")

    def _emit_basis_before(self, basis: Pauli, qubit: int) -> None:
        if basis == "X":
            self._emit("H", qubit)
        elif basis == "Y":
            self._emit("SDG", qubit)
            self._emit("H", qubit)
            self._emit("S", qubit)

    def _emit_basis_after(self, basis: Pauli, qubit: int) -> None:
        self._emit_basis_before(basis, qubit)

    def _emit_xor_records(self, records: Sequence[_RecordTarget]) -> None:
        self._emit("XOR", *(record.index for record in records))
        if sum(record.inverted for record in records) % 2 == 1:
            self._emit("FLIP")

    def _save_measurement(self) -> None:
        self._emit("SAVE INT", self.measurements_n)
        self.measurements_n += 1

    def _control_target(
        self,
        token: str,
        instruction: _Instruction,
    ) -> _QubitTarget | _RecordTarget:
        record = self._maybe_record_target(token, instruction)
        if record is not None:
            return record
        return self._qubit_target(token, instruction)

    def _qubit_targets(
        self,
        tokens: Sequence[str],
        instruction: _Instruction,
    ) -> tuple[_QubitTarget, ...]:
        return tuple(self._qubit_target(token, instruction) for token in tokens)

    def _qubit_target(self, token: str, instruction: _Instruction) -> _QubitTarget:
        inverted = token.startswith("!")
        value = token[1:] if inverted else token
        if not value or not value.isdecimal():
            raise self._error(instruction, f"expected qubit target, got {token!r}")
        qubit = int(value)
        self._record_qubit(qubit)
        return _QubitTarget(qubit, inverted=inverted)

    def _record_targets(
        self,
        tokens: Sequence[str],
        instruction: _Instruction,
    ) -> tuple[_RecordTarget, ...]:
        return tuple(self._record_target(token, instruction) for token in tokens)

    def _record_target(self, token: str, instruction: _Instruction) -> _RecordTarget:
        record = self._maybe_record_target(token, instruction)
        if record is None:
            raise self._error(instruction, f"expected rec[-k] target, got {token!r}")
        return record

    def _maybe_record_target(
        self,
        token: str,
        instruction: _Instruction,
    ) -> _RecordTarget | None:
        match = _RECORD_RE.match(token)
        if match is None:
            return None
        offset = int(match.group(2))
        if offset >= 0:
            raise self._error(instruction, f"record target must be relative and negative, got {token!r}")
        index = self.measurements_n + offset
        if index < 0:
            raise self._error(instruction, f"record target {token!r} refers before the first measurement")
        return _RecordTarget(index=index, inverted=bool(match.group(1)))

    def _pauli_product(
        self,
        token: str,
        instruction: _Instruction,
    ) -> tuple[_PauliTarget, ...]:
        return tuple(self._pauli_target(item, instruction) for item in token.split("*"))

    def _pauli_target(self, token: str, instruction: _Instruction) -> _PauliTarget:
        inverted = token.startswith("!")
        value = token[1:] if inverted else token
        if len(value) < 2:
            raise self._error(instruction, f"expected Pauli target, got {token!r}")
        pauli = value[0].upper()
        if pauli not in {"X", "Y", "Z"}:
            raise self._error(instruction, f"expected Pauli target, got {token!r}")
        qubit_text = value[1:]
        if not qubit_text.isdecimal():
            raise self._error(instruction, f"expected Pauli qubit index, got {token!r}")
        qubit = int(qubit_text)
        self._record_qubit(qubit)
        return _PauliTarget(pauli=pauli, qubit=qubit, inverted=inverted)  # type: ignore[arg-type]

    def _record_qubit(self, qubit: int) -> None:
        self.qubits_n = max(self.qubits_n, qubit + 1)

    def _required_probability(self, instruction: _Instruction) -> str:
        if len(instruction.args) != 1:
            raise self._error(instruction, f"{instruction.name} expects one probability argument")
        return instruction.args[0]

    def _optional_probability(self, instruction: _Instruction) -> str | None:
        if not instruction.args:
            return None
        if len(instruction.args) != 1:
            raise self._error(instruction, f"{instruction.name} expects zero or one probability argument")
        return instruction.args[0]

    def _expect_no_args(self, instruction: _Instruction) -> None:
        if instruction.args:
            raise self._error(instruction, f"{instruction.name} does not accept parenthesized arguments")

    def _emit(self, name: str, *args: object) -> None:
        if args:
            self.operations.append(" ".join((name, *(str(arg) for arg in args))))
        else:
            self.operations.append(name)

    def _error(self, instruction: _Instruction, message: str) -> StimParseError:
        return StimParseError(f"line {instruction.line_no}: {message}")


def _logical_lines(raw_lines: Iterable[str]) -> list[_Line]:
    lines: list[_Line] = []
    for line_no, raw_line in enumerate(raw_lines, start=1):
        text = raw_line.split("#", 1)[0].strip()
        if not text:
            continue
        if text == "{":
            lines.append(_Line(text, line_no))
            continue
        if text == "}":
            lines.append(_Line(text, line_no))
            continue
        if text.endswith("{") and not text.endswith(" {"):
            text = f"{text[:-1].rstrip()} {{"
        lines.append(_Line(text, line_no))
    return lines


def _parse_repeat(line: _Line) -> tuple[int, bool] | None:
    tokens = line.text.split()
    if not tokens or tokens[0].upper() != "REPEAT":
        return None
    if len(tokens) not in {2, 3}:
        raise StimParseError(f"line {line.line_no}: REPEAT expects a count and '{{'")
    if not tokens[1].isdecimal():
        raise StimParseError(f"line {line.line_no}: REPEAT count must be a non-negative integer")
    has_open_brace = len(tokens) == 3
    if has_open_brace and tokens[2] != "{":
        raise StimParseError(f"line {line.line_no}: REPEAT expects '{{'")
    return int(tokens[1]), has_open_brace


def _parse_instruction(line: _Line) -> _Instruction:
    first, rest = _split_instruction_head(line)
    match = _INSTRUCTION_RE.match(first)
    if match is None:
        raise StimParseError(f"line {line.line_no}: invalid operation syntax")
    args_text = match.group(2)
    args = () if args_text is None else tuple(arg.strip() for arg in args_text.split(","))
    if any(not arg for arg in args):
        raise StimParseError(f"line {line.line_no}: empty parenthesized argument")
    return _Instruction(
        name=match.group(1).upper(),
        args=args,
        targets=tuple(rest.split()),
        line_no=line.line_no,
    )


def _split_instruction_head(line: _Line) -> tuple[str, str]:
    text = line.text
    paren = text.find("(")
    space = text.find(" ")
    if paren < 0 or (space >= 0 and space < paren):
        first, _, rest = text.partition(" ")
        return first, rest

    close = text.find(")", paren + 1)
    if close < 0:
        raise StimParseError(f"line {line.line_no}: missing closing parenthesis")
    return text[: close + 1], text[close + 1 :].strip()


def _pairs(items: Sequence[str]) -> Iterable[tuple[str, str]]:
    for index in range(0, len(items), 2):
        yield items[index], items[index + 1]
