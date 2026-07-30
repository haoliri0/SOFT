from os import PathLike
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np
import numpy.typing as npt

BoolSamples = npt.NDArray[np.bool_]
PackedSamples = npt.NDArray[np.uint8]


class DetectorMetadata(TypedDict):
    records: Tuple[int, ...]
    coords: Tuple[float, ...]
    line: int


class ObservableMetadata(TypedDict):
    index: int
    records: Tuple[int, ...]
    line: int


class Timing(TypedDict):
    parse_s: float
    plan_s: float
    presample_s: float
    execute_s: float
    accumulate_s: float
    sample_s: float


class CountsResult(TypedDict):
    shots: int
    discarded: int
    accepted: int
    logical_errors: int
    discard_rate: float
    logical_error_rate: float
    active_threads: int
    timing: Timing


class CountsSamplerInfo(TypedDict):
    num_qubits: int
    num_measurements: int
    num_detectors: int
    num_observable_includes: int
    observable: int
    max_active_qubits: int
    batch_size: int
    sample_chunk_shots: int
    threads: int
    active_components: bool
    detector_postselection: bool
    batch_mask_threshold_denominator: int
    backend: str


class SymFTError(RuntimeError): ...


class Circuit:
    def __init__(
        self,
        text: Optional[Union[str, bytes]] = ...,
        path: Optional[Union[str, bytes, PathLike]] = ...,
    ) -> None: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_measurements(self) -> int: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def num_observables(self) -> int: ...
    @property
    def num_observable_includes(self) -> int: ...
    @property
    def detectors(self) -> List[DetectorMetadata]: ...
    @property
    def observables(self) -> List[ObservableMetadata]: ...
    def compile_sampler(
        self,
        batch: bool = ...,
        batch_size: int = ...,
        sample_chunk_shots: int = ...,
    ) -> CompiledMeasurementSampler: ...
    def compile_counts_sampler(
        self,
        batch: bool = ...,
        observable: int = ...,
        postselect_detectors: bool = ...,
        batch_size: int = ...,
        sample_chunk_shots: int = ...,
        threads: int = ...,
        batch_mask_threshold_denominator: int = ...,
        cuda: bool = ...,
        cuda_mode: str = ...,
        shots_per_launch: int = ...,
        threads_per_block: int = ...,
    ) -> CompiledCountsSampler: ...
    def sample(
        self,
        shots: int = ...,
        seed: int = ...,
        batch: bool = ...,
        bit_packed: bool = ...,
        batch_size: int = ...,
        sample_chunk_shots: int = ...,
    ) -> Union[BoolSamples, PackedSamples]: ...
    def sample_counts(
        self,
        shots: int = ...,
        seed: int = ...,
        batch: bool = ...,
        observable: int = ...,
        postselect_detectors: bool = ...,
        batch_size: int = ...,
        sample_chunk_shots: int = ...,
        threads: int = ...,
        batch_mask_threshold_denominator: int = ...,
        cuda: bool = ...,
        cuda_mode: str = ...,
        shots_per_launch: int = ...,
        threads_per_block: int = ...,
    ) -> CountsResult: ...
    def sample_detectors(
        self,
        shots: int = ...,
        seed: int = ...,
        bit_packed: bool = ...,
    ) -> Union[BoolSamples, PackedSamples]: ...


class CompiledMeasurementSampler:
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_measurements(self) -> int: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def max_active_qubits(self) -> int: ...
    def sample(
        self,
        shots: int = ...,
        seed: int = ...,
        bit_packed: bool = ...,
    ) -> Union[BoolSamples, PackedSamples]: ...
    def sample_detectors(
        self,
        shots: int = ...,
        seed: int = ...,
        bit_packed: bool = ...,
    ) -> Union[BoolSamples, PackedSamples]: ...


class CompiledCountsSampler:
    @property
    def info(self) -> CountsSamplerInfo: ...
    @property
    def preprocessing_timing(self) -> Timing: ...
    def sample(self, shots: int = ..., stream_id: Optional[int] = ...) -> CountsResult: ...


def simd_backend() -> str: ...
def cuda_enabled() -> bool: ...
def active_cuda_backend() -> str: ...
