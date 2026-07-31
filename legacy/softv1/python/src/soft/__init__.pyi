from os import PathLike
from typing import Any, Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from ._native import CpuCircuit as CpuCircuit
from ._native import CudaSimulator as CudaSimulator, Simulator as Simulator
from ._native import SoftError as SoftError, SoftCudaError as SoftCudaError
from ._native import cuda_device_count as cuda_device_count
from ._native import cuda_driver_version as cuda_driver_version
from ._native import cuda_enabled as cuda_enabled
from ._native import cuda_runtime_version as cuda_runtime_version

IntSamples = npt.NDArray[np.intc]
FloatSamples = npt.NDArray[np.float64]
EntryCounts = npt.NDArray[np.uintc]
PrintKind = Literal["INT", "FLT", "ERR", "ENTRIES_N", "STATE"]
CircuitFormat = Literal["auto", "soft", "stim"]
Pathish = Union[str, bytes, PathLike[str], PathLike[bytes]]

__version__: str

class StimParseError(ValueError): ...

class StimCompileStats:
    qubits_n: int
    measurements_n: int
    mem_ints_n: int
    detectors_n: int
    observables_n: int
    def __init__(
        self,
        qubits_n: int,
        measurements_n: int,
        mem_ints_n: int,
        detectors_n: int,
        observables_n: int,
    ) -> None: ...

class StimCompileResult:
    text: str
    observable_indices: tuple[int, ...]
    stats: StimCompileStats
    def __init__(self, text: str, observable_indices: tuple[int, ...], stats: StimCompileStats) -> None: ...

class SimulatorArgs:
    shot_i: int
    shots_n: int
    qubits_n: int
    entries_m: int
    mem_ints_n: int
    mem_flts_n: int
    epsilon: float
    seed: int
    def __init__(
        self,
        shot_i: int = ...,
        shots_n: int = ...,
        qubits_n: int = ...,
        entries_m: int = ...,
        mem_ints_n: int = ...,
        mem_flts_n: int = ...,
        epsilon: float = ...,
        seed: int = ...,
    ) -> None: ...

class PrintRecord:
    kind: PrintKind
    value: object

class RunResult:
    simulator: Simulator
    prints: tuple[PrintRecord, ...]
    @property
    def work_int(self) -> IntSamples: ...
    @property
    def work_float(self) -> FloatSamples: ...
    @property
    def errors(self) -> IntSamples: ...
    @property
    def entries_n(self) -> EntryCounts: ...

class Circuit:
    def __init__(
        self,
        text: Optional[Union[str, bytes]] = ...,
        *,
        path: Optional[Pathish] = ...,
        format: CircuitFormat = ...,
        append_outputs: bool = ...,
    ) -> None: ...
    @property
    def text(self) -> str: ...
    @property
    def operations(self) -> tuple[str, ...]: ...
    @property
    def observable_indices(self) -> Optional[tuple[int, ...]]: ...
    @property
    def inferred_qubits_n(self) -> Optional[int]: ...
    @property
    def inferred_mem_ints_n(self) -> Optional[int]: ...
    def run(
        self,
        args: Optional[SimulatorArgs] = ...,
        *,
        simulator: Optional[Simulator] = ...,
        **kwargs: Any,
    ) -> RunResult: ...
    def sample_counts(
        self,
        args: Optional[SimulatorArgs] = ...,
        *,
        observable: int = ...,
        cuda: bool = ...,
        **kwargs: Any,
    ) -> dict[str, object]: ...

def compile_stim_text(text: Union[str, bytes], *, append_outputs: bool = ...) -> StimCompileResult: ...
def stim_to_soft(text: Union[str, bytes], *, append_outputs: bool = ...) -> str: ...
def read_file(path: Pathish, *, format: CircuitFormat = ..., append_outputs: bool = ...) -> Circuit: ...
def read_soft_file(path: Pathish, *, format: CircuitFormat = ..., append_outputs: bool = ...) -> Circuit: ...
def read_stim_file(path: Pathish, *, append_outputs: bool = ...) -> Circuit: ...
def read_cpu_stim_file(path: Pathish, *, num_qubits: int = ...) -> CpuCircuit: ...
def sample_cpu(circuit: Union[CpuCircuit, str, bytes], shots: int = ..., **kwargs: Any) -> object: ...
def sample_counts_cpu(circuit: Union[CpuCircuit, str, bytes], shots: int = ..., **kwargs: Any) -> dict[str, object]: ...
def run_file(
    path: Pathish,
    args: Optional[SimulatorArgs] = ...,
    *,
    format: CircuitFormat = ...,
    append_outputs: bool = ...,
    **kwargs: Any,
) -> RunResult: ...
def sample_counts_file(
    path: Pathish,
    args: Optional[SimulatorArgs] = ...,
    *,
    format: CircuitFormat = ...,
    observable: int = ...,
    cuda: bool = ...,
    **kwargs: Any,
) -> dict[str, object]: ...
def run(
    circuit: Union[Circuit, str, bytes],
    args: Optional[SimulatorArgs] = ...,
    *,
    simulator: Optional[Simulator] = ...,
    **kwargs: Any,
) -> RunResult: ...
def sample(
    circuit: Union[Circuit, str, bytes],
    args: Optional[SimulatorArgs] = ...,
    *,
    simulator: Optional[Simulator] = ...,
    **kwargs: Any,
) -> RunResult: ...
def sample_counts(
    circuit: Union[Circuit, str, bytes],
    args: Optional[SimulatorArgs] = ...,
    *,
    observable: int = ...,
    cuda: bool = ...,
    format: CircuitFormat = ...,
    append_outputs: bool = ...,
    **kwargs: Any,
) -> dict[str, object]: ...
