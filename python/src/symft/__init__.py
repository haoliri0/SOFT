"""Python bindings for the SymFT Clifford+T simulator."""

from os import PathLike
from typing import Union

from ._native import (
    Circuit,
    CompiledCountsSampler,
    CompiledMeasurementSampler,
    SymFTError,
    active_cuda_backend,
    cuda_enabled,
    simd_backend,
)

__version__ = "0.1.0"


def read_stim_file(path: Union[str, bytes, PathLike]) -> Circuit:
    """Parse a Stim circuit file into a :class:`Circuit`.

    Args:
        path: A filesystem path represented by ``str``, ``bytes``, or
            :class:`os.PathLike`.

    Raises:
        SymFTError: If the file cannot be read or is not a supported Stim
            circuit.
    """
    return Circuit(path=path)


def sample(circuit, shots=1, **kwargs):
    """Sample measurements from Stim text or an existing :class:`Circuit`.

    Args:
        circuit: A :class:`Circuit`, or Stim circuit text as ``str`` or
            ``bytes``.
        shots: Number of samples. Must be nonnegative.
        **kwargs: Arguments forwarded to :meth:`Circuit.sample`.

    Returns:
        A two-dimensional NumPy array of measurement records.
    """
    if not isinstance(circuit, Circuit):
        circuit = Circuit(circuit)
    return circuit.sample(shots=shots, **kwargs)


__all__ = [
    "Circuit",
    "CompiledCountsSampler",
    "CompiledMeasurementSampler",
    "SymFTError",
    "__version__",
    "active_cuda_backend",
    "cuda_enabled",
    "read_stim_file",
    "sample",
    "simd_backend",
]
