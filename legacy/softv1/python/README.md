# SOFT

SOFT is a high-performance parallel simulator for universal fault-tolerant quantum circuits. The Python package is imported as `soft` and wraps both the CPU Stim sampler and the CUDA SOFT simulator.

## Build

Build the Python extension from the `python` directory:

```bash
cd python
python setup.py build_ext --inplace
```

The current Python extension builds the CUDA and CPU wrappers together, so a CUDA toolkit with `nvcc` is required. Set `CUDA_HOME` or `CUDA_PATH` if `nvcc` is not on `PATH`. Optional CUDA build knobs:

```bash
SOFT_PY_CUDA_ARCH=sm_120 python setup.py build_ext --inplace
SOFT_PY_NVCC_FLAGS="--use_fast_math" python setup.py build_ext --inplace
```

Run examples from the same directory with the source tree on `PYTHONPATH`:

```bash
PYTHONPATH=src python examples/test.py
```

## Backend Inputs

CPU and GPU do not consume the same circuit representation.

| Backend | Input | Main use |
| --- | --- | --- |
| CPU | Stim text or `.stim` files | Stabilizer sampling and detector/logical statistics |
| GPU | SOFT CUDA program text; Stim input is first compiled by the Python wrapper | CUDA execution and CUDA statistics |

For statistics APIs, `cuda=False` selects the CPU backend for Stim input. Pass `cuda=True` to force the CUDA path. If the input is already a SOFT CUDA program, the CPU backend is not applicable and execution uses the CUDA path.

## CPU Usage

Use the CPU backend when the input is Stim text or a `.stim` file.

Pure sampling returns per-shot records:

```python
import soft

circuit = soft.CpuCircuit("""
H 0
M 0
""")

measurements = circuit.sample(shots=8)
detectors = circuit.sample_detectors(shots=8)
print(measurements)
print(detectors)
```

For files, construct the CPU circuit from the `.stim` path:

```python
import soft

circuit = soft.read_cpu_stim_file("circuit.stim")
measurements = soft.sample_cpu(circuit, shots=1000)
```

Use the statistics API when you only need detector discard counts and logical error counts:

```python
import soft

summary = soft.sample_counts_file(
    "circuit.stim",
    shots_n=10000,
    observable=0,
    cuda=False,
)

print(summary["backend"])
print(summary["discard_rate"])
print(summary["logical_error_rate"])
print(summary["timing"])
```

You can also pass Stim text directly:

```python
summary = soft.sample_counts(
    """
H 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
""",
    format="stim",
    shots=1000,
    observable=0,
    cuda=False,
)
```

## GPU Usage

Use the GPU backend for CUDA execution. Pure sampling/program execution goes through `run`:

```python
import soft

program = """
H 0
MEASURE 0
PRINT INT
"""

result = soft.run(program, shots_n=8, entries_m=16, seed=1)
print(result.prints[0].value)
```

For repeated calls, keep a `Circuit` object:

```python
circuit = soft.Circuit(program, format="soft")
result = circuit.run(shots_n=8, entries_m=16, seed=1)
```

For statistics on the CUDA path, pass `cuda=True`:

```python
summary = soft.sample_counts_file(
    "circuit.stim",
    shots_n=100000,
    entries_m=2048,
    seed=42,
    observable=0,
    cuda=True,
)

print(summary["backend"])
print(summary["logical_error_rate"])
```

## Statistics Result

`sample_counts`, `sample_counts_file`, and `sample_counts_cpu` return a dictionary with:

| Key | Meaning |
| --- | --- |
| `shots` | Requested shots |
| `discarded` | Shots rejected by detector events |
| `accepted` | Shots without detector events |
| `logical_errors` | Accepted shots where the selected observable has parity 1 |
| `discard_rate` | `discarded / shots` |
| `logical_error_rate` | `logical_errors / accepted` |
| `backend` | `"cpu"` or `"cuda"` |
| `active_threads` | Worker count reported by the backend |
| `timing` | `parse_s`, `plan_s`, `presample_s`, `execute_s`, `accumulate_s`, `sample_s` |

For CPU calls, `parse_s` is only nonzero when the high-level wrapper constructs the circuit from text or file. If you call `CpuCircuit(...).sample_counts()` directly, parsing has already happened, so `parse_s` is `0`.

## Project Docs

- Build details: [docs/Building.md](docs/Building.md)
- Command-line interface: [docs/UsingCli.md](docs/UsingCli.md)

## Paper

For the first version of the SOFT paper, see:

> Li, Riling, et al. "SOFT: A High-Performance Simulator for Universal Fault-Tolerant Quantum Circuits." _arXiv_, 2025, arXiv:2512.23037.
