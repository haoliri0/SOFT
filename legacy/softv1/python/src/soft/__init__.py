"""Python bindings for the SOFT CPU and CUDA simulators."""

from ._circuit import (
    Circuit,
    CircuitFormat,
    PrintRecord,
    RunResult,
    SimulatorArgs,
    read_file,
    read_stim_file,
    run,
    run_file,
    sample,
    sample_counts,
    sample_counts_file,
    _sample_counts_cpu_source,
)
from ._native import (
    CpuCircuit,
    CudaSimulator,
    Simulator,
    SoftError,
    SoftCudaError,
    cuda_device_count,
    cuda_driver_version,
    cuda_enabled,
    cuda_runtime_version,
)
from ._stim import StimCompileResult, StimCompileStats, StimParseError, compile_stim_text, stim_to_soft

__version__ = "0.1.0"

read_soft_file = read_file
def read_cpu_stim_file(path, *, num_qubits=-1):
    return CpuCircuit(path=path, num_qubits=num_qubits)

def sample_cpu(circuit, shots=1, **kwargs):
    if not isinstance(circuit, CpuCircuit):
        circuit = CpuCircuit(circuit)
    return circuit.sample(shots=shots, **kwargs)

def sample_counts_cpu(circuit, shots=1, **kwargs):
    if isinstance(circuit, CpuCircuit):
        return circuit.sample_counts(shots=shots, **kwargs)
    return _sample_counts_cpu_source(text=circuit, shots=shots, **kwargs)


__all__ = [
    "Circuit",
    "CircuitFormat",
    "CpuCircuit",
    "CudaSimulator",
    "PrintRecord",
    "RunResult",
    "Simulator",
    "SimulatorArgs",
    "SoftError",
    "SoftCudaError",
    "StimCompileResult",
    "StimCompileStats",
    "StimParseError",
    "__version__",
    "compile_stim_text",
    "cuda_device_count",
    "cuda_driver_version",
    "cuda_enabled",
    "cuda_runtime_version",
    "read_cpu_stim_file",
    "read_file",
    "read_soft_file",
    "read_stim_file",
    "run",
    "run_file",
    "sample_cpu",
    "sample_counts_cpu",
    "sample",
    "sample_counts",
    "sample_counts_file",
    "stim_to_soft",
]
