# SymFT Python Interface

The SymFT Python interface exposes the C++ implementation of the exact
simulator described in the [project overview](../README.md).
It compiles noisy adaptive Clifford-plus-Pauli-rotation circuits into the
paper's planned sampling instruction stream and reuses that plan across shots.
It supports:

- parsing circuits from text or `.stim` files;
- returning per-shot measurement records or detector records;
- directly aggregating detector discards and logical error counts without
  materializing large per-shot result arrays;
- compiling and reusing samplers, with batching and multithreaded counts
  sampling;
- optionally using the C++ CUDA counts sampler when built with CUDA support;
- returning either regular boolean NumPy arrays or bit-packed arrays.

Package version `0.1.0` requires Python 3.9+, NumPy 1.20+, and a C++20-capable
compiler.

## Contents

- [Installation and Build](#installation-and-build)
- [Quick Start](#quick-start)
- [Creating Circuits](#creating-circuits)
- [Reading Circuit Metadata](#reading-circuit-metadata)
- [Measurement Record Sampling](#measurement-record-sampling)
- [Detector Sampling](#detector-sampling)
- [Logical Error Statistics](#logical-error-statistics)
- [Compiling and Reusing Samplers](#compiling-and-reusing-samplers)
- [Randomness and Reproducibility](#randomness-and-reproducibility)
- [Batching and Performance Parameters](#batching-and-performance-parameters)
- [Supported Stim Operations](#supported-stim-operations)
- [Full API Reference](#full-api-reference)
- [Exceptions, Bounds, and Threading](#exceptions-bounds-and-threading)
- [Development and Testing](#development-and-testing)

## Installation and Build

The Python extension compiles the C++ sources in this repository directly.
macOS and Linux need a C++20 compiler; Windows needs a suitable version of MSVC
Build Tools.

Using a virtual environment and editable install from this directory is
recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Regular local install:

```bash
python -m pip install .
```

Build and test in the source directory without installing:

```bash
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests -v
```

CPU source builds match the CMake default by enabling host-native optimization
and separately compiled AVX2/AVX-512 kernels with runtime dispatch. Set
`SYMFT_PY_NATIVE=0` when building a wheel that must run on CPUs other than the
build host; the separately dispatched SIMD kernels remain available when the
compiler supports them.

CUDA support is opt-in so CPU-only environments keep building normally. To build
the CUDA counts backend, install a CUDA toolkit with `nvcc` available and run:

```bash
SYMFT_PY_ENABLE_CUDA=1 python setup.py build_ext --inplace
```

`CUDA_HOME` or `CUDA_PATH` may be set explicitly when `nvcc` is not on `PATH`.
Set `SYMFT_PY_CUDA_REAL_DOUBLE=1` as well to compile CUDA active-state arithmetic
in double precision. The CUDA build tries to detect the local GPU architecture
with `nvidia-smi` and emits device-specific code, avoiding PTX JIT on machines
whose driver is older than the CUDA toolkit. If auto-detection is unavailable,
set `SYMFT_PY_CUDA_ARCH` yourself, for example `SYMFT_PY_CUDA_ARCH=sm_89` for
compute capability 8.9. Extra NVCC flags can be passed with
`SYMFT_PY_CUDA_NVCC_FLAGS`.

Check the installation and backends:

```python
import symft

print(symft.__version__)
print(symft.simd_backend())
print(symft.cuda_enabled())
print(symft.active_cuda_backend())
```

Prepared CPU batches use the same contiguous dense-vector kernels as
single-shot sampling. The selected CPU backend is reported by
`simd_backend()`. Symbolic signs use 64-shot packed-word operations and longer
word loops may also be compiler-vectorized; there is no separately selected
symbolic-sign backend. Backend names depend on the compilation target and
runtime machine. Application logic should not depend on a particular backend
name. `cuda_enabled()` reports whether the extension was compiled with CUDA
support; it does not guarantee that a runtime CUDA device is present.

## Quick Start

```python
import symft

circuit = symft.Circuit("""
H 0
M 0
""")

samples = circuit.sample(shots=10, seed=123)
print(samples.shape)  # (10, 1)
print(samples.dtype)  # bool
```

Compile a sampler when sampling the same circuit repeatedly:

```python
sampler = circuit.compile_sampler(batch=True)

first = sampler.sample(shots=1000, seed=1)
second = sampler.sample(shots=1000, seed=2)
```

Use the counts API when you only need the detector discard rate and the error
rate of a logical observable. It does not return all per-shot measurement
results to Python:

```python
result = circuit.sample_counts(shots=100_000, observable=0, threads=4)
print(result["discard_rate"])
print(result["logical_error_rate"])
```

## Creating Circuits

### From Text

`Circuit` accepts `str` or `bytes`. Calling it without arguments creates an
empty circuit.

```python
circuit = symft.Circuit("X 0\nM 0\n")
empty = symft.Circuit()
```

### From a File

`path` accepts `str`, `bytes`, and `os.PathLike`, so `pathlib.Path` can be used
directly:

```python
from pathlib import Path
import symft

path = Path("benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim")
circuit = symft.Circuit(path=path)

# Equivalent convenience function
circuit = symft.read_stim_file(path)
```

`text` and `path` are mutually exclusive. Parse failures, missing files, and
unreadable files raise `symft.SymFTError`. A `Circuit` cannot be rewritten by
calling `__init__` again after creation; create a new object when loading a new
circuit.

### Module-Level Sampling Convenience

`symft.sample` accepts circuit text or an existing `Circuit`. The remaining
arguments are forwarded to `Circuit.sample`:

```python
samples = symft.sample("X 0\nM 0\n", shots=8, seed=7)
```

File paths are not auto-detected as paths. Use `read_stim_file` or
`Circuit(path=...)` instead.

## Reading Circuit Metadata

```python
circuit = symft.Circuit("""
M 0
DETECTOR(1.5, 2.5) rec[-1]
OBSERVABLE_INCLUDE(2) rec[-1]
""")

print(circuit.num_qubits)                 # 1
print(circuit.num_measurements)           # 1
print(circuit.num_detectors)              # 1
print(circuit.num_observables)            # 3, max observable index + 1
print(circuit.num_observable_includes)    # 1, number of include instructions
print(circuit.detectors)
print(circuit.observables)
```

Each dictionary in `detectors` contains:

| Key | Type | Meaning |
| --- | --- | --- |
| `records` | `tuple[int, ...]` | Measurement record numbers referenced by the detector; internal numbering starts at 1 |
| `coords` | `tuple[float, ...]` | Coordinates after applying `SHIFT_COORDS` |
| `line` | `int` | Source circuit line number, starting at 1 |

Each dictionary in `observables` contains `index`, `records`, and `line`.
`records` also uses the internal 1-based record numbering, not Stim's negative
offset syntax.

## Measurement Record Sampling

Call signature:

```python
circuit.sample(
    shots=1,
    seed=1,
    batch=False,
    bit_packed=False,
    batch_size=0,
    sample_chunk_shots=0,
)
```

The regular output is a two-dimensional `numpy.ndarray[numpy.bool_]` with shape
`(shots, circuit.num_measurements)`. Columns follow the circuit's measurement
record order.

```python
circuit = symft.Circuit("X 0\nM 0\nM 1\n")
samples = circuit.sample(shots=4, seed=1)

assert samples.shape == (4, 2)
assert samples[:, 0].all()
```

`shots=0` still preserves the correct column width, for example
`(0, num_measurements)`. `shots` must be in the range `0..2**31-1`.

### Bit-Packed Output

`bit_packed=True` returns `numpy.uint8` with shape
`(shots, ceil(num_measurements / 8))`. Each byte is little-endian within the
byte: measurement record 0 is bit 0 of byte 0, record 7 is bit 7, and record 8
is bit 0 of byte 1.

```python
packed = circuit.sample(shots=1000, bit_packed=True)

# Recover measurement record 0
record_0 = (packed[:, 0] & 0x01) != 0
```

For large per-shot outputs, prefer bit packing to reduce the Python result array
size by about 8x. The native layer still holds intermediate packed words while
generating results, so very large `shots * records` tasks should be sampled in
chunks.

## Detector Sampling

```python
detectors = circuit.sample_detectors(
    shots=1000,
    seed=1,
    bit_packed=False,
)
```

The regular output shape is `(shots, circuit.num_detectors)` with dtype
`numpy.bool_`. `True` means the corresponding detector fired in that shot. The
bit-packing rules are the same as measurement records, with column width
`ceil(num_detectors / 8)`.

`sample_detectors` uses the single-shot detector execution path and does not
accept `batch`, `batch_size`, or `threads`. Use `sample_counts` when high
throughput detector or logical-error aggregation is needed.

## Logical Error Statistics

Call signature:

```python
circuit.sample_counts(
    shots=1,
    seed=1,
    batch=True,
    observable=0,
    postselect_detectors=False,
    batch_size=0,
    sample_chunk_shots=0,
    threads=1,
    batch_mask_threshold_denominator=2,
    cuda=False,
    cuda_mode="gpu",
    shots_per_launch=0,
    threads_per_block=0,
)
```

For each shot, any detector firing contributes to `discarded`; otherwise the
shot contributes to `accepted`. All `OBSERVABLE_INCLUDE` records for the
selected `observable` are combined by XOR parity, and accepted shots with parity
1 contribute to `logical_errors`.

Returned dictionary:

| Key | Type | Meaning |
| --- | --- | --- |
| `shots` | `int` | Number of requested and completed shots |
| `discarded` | `int` | Number of shots where at least one detector fired |
| `accepted` | `int` | Number of shots where no detector fired |
| `logical_errors` | `int` | Number of accepted shots whose logical observable parity is 1 |
| `discard_rate` | `float` | `discarded / shots` |
| `logical_error_rate` | `float` | `logical_errors / accepted` |
| `active_threads` | `int` | Number of worker threads actually used |
| `timing` | `dict` | `parse_s`, `plan_s`, `presample_s`, `execute_s`, `accumulate_s`, `sample_s` |

When a denominator is 0, the corresponding rate is `nan`.
`accepted + discarded == shots` always holds.

### CUDA Counts Backend

When the extension is built with `SYMFT_PY_ENABLE_CUDA=1`, counts sampling can use
the C++ CUDA sampler:

```python
result = circuit.sample_counts(shots=1_000_000, cuda=True)
sampler = circuit.compile_counts_sampler(cuda=True, cuda_mode="gpu")
```

`cuda_mode` selects how exogenous noise and expression bits are prepared:
`"gpu"` (default), `"cpu_presampled"`, `"gpu_presample_expressions"`,
`"gpu_on_demand_expressions"`, or `"gpu_lazy"`. `shots_per_launch=0` and
`threads_per_block=0` use the CUDA backend defaults. If CUDA support was not
compiled in, `cuda=True` raises `SymFTError`.

```python
import math

result = circuit.sample_counts(shots=10_000, observable=0)
if not math.isnan(result["logical_error_rate"]):
    print(result["logical_error_rate"])
```

`observable` is a non-negative index. If the circuit has no
`OBSERVABLE_INCLUDE` for that index, the logical error count is 0. Callers should
check `circuit.observables` to confirm that the intended index exists.

### Detector Postselection

`postselect_detectors=True` allows batch execution to eliminate a shot early
when a detector fires. The result statistics have the same semantics. This is
useful for circuits where detectors appear early and the discard rate is high.

`batch_mask_threshold_denominator` controls the dead-shot compaction threshold
during postselection. The default value, 2, allows compaction when dead shots
reach roughly `1/2`. It only applies when
`batch=True, postselect_detectors=True`; the default is usually appropriate.

## Compiling and Reusing Samplers

### `CompiledMeasurementSampler`

```python
sampler = circuit.compile_sampler(
    batch=True,
    batch_size=0,
    sample_chunk_shots=0,
)

samples = sampler.sample(shots=1000, seed=1, bit_packed=False)
detectors = sampler.sample_detectors(shots=1000, seed=1)

print(sampler.num_qubits)
print(sampler.num_measurements)
print(sampler.num_detectors)
print(sampler.max_active_qubits)
```

It reuses the lowered and planned instruction program, which is useful when
returning per-shot arrays from the same circuit multiple times. The `batch`
value set at compile time determines the measurement sampling execution path;
`sample_detectors` still uses the single-shot detector path.

### `CompiledCountsSampler`

```python
sampler = circuit.compile_counts_sampler(
    batch=True,
    observable=0,
    postselect_detectors=True,
    threads=4,
)

print(sampler.info)
print(sampler.preprocessing_timing)

result_a = sampler.sample(shots=100_000, stream_id=10)
result_b = sampler.sample(shots=100_000, stream_id=11)
```

This sampler reuses the instruction program, presampling plan, and worker
memory, making it suitable for repeated statistics jobs. `info` reports the
normalized configuration:

- `num_qubits`, `num_measurements`, `num_detectors`;
- `num_observable_includes`, `observable`;
- `max_active_qubits`;
- final `batch_size`, `sample_chunk_shots`, and `threads`;
- `active_components`, indicating whether the cost-gated product-component
  active-state backend was selected;
- `detector_postselection` and `batch_mask_threshold_denominator`;
- `backend`, one of `"single"`, `"batch"`, or `"cuda"`.

The same `CompiledCountsSampler` can be called from multiple Python threads.
The wrapper serializes these calls to protect mutable internal runtime state.
For truly parallel independent jobs, create a separate sampler per calling
thread or use a single sampler's `threads` parameter.

## Randomness and Reproducibility

Per-shot array APIs use `seed`: with the same version, call parameters, and
execution path, the same seed reproduces the same results.

```python
a = circuit.sample(shots=100, seed=123)
b = circuit.sample(shots=100, seed=123)
```

`Circuit.sample_counts(..., seed=N)` uses `N` as the explicit random stream ID
for the prepared counts sampler. `CompiledCountsSampler.sample` uses
`stream_id` instead:

- passing the same `stream_id` explicitly reproduces the same statistics;
- omitting `stream_id` makes the sampler auto-increment an internal stream
  counter starting at 0;
- do not depend on bit-for-bit equality between single-shot and batch paths for
  the same identifier;
- changing batch or chunk parameters can change random-number partitioning, so
  compare only statistical distributions across configurations.

## Batching and Performance Parameters

### `batch`

- Measurement array APIs default to `False`; this is appropriate for short jobs
  and debugging.
- Counts APIs default to `True`; this is suitable for large-scale aggregation.
- With `batch=False`, counts `threads` and `batch_size` do not increase
  parallelism.

### `batch_size`

`0` means automatic selection. The automatic value is:

```text
min(2048, max(1, 32768 / 2^max_active_qubits))
```

Increasing the batch can improve throughput, but it also increases memory usage
according to the dense active-state vector size. Circuits with high
`max_active_qubits` should prefer the automatic value. Explicit values must be
non-negative integers.

### `sample_chunk_shots`

`0` means automatic selection. The base default is 2048; batch counts ensure the
chunk is not smaller than the batch. A chunk is the unit of presampling and
thread work partitioning. When using multiple `threads`, the shot count should
contain enough chunks; otherwise `active_threads` may be smaller than requested.

### `threads`

Only the batch counts path uses this parameter. `threads=1` is the default, and
`threads=0` is normalized to 1. The C++ execution releases the Python GIL. Use
`sampler.info` and result `active_threads` to confirm the final configuration
and actual parallelism.

### Choosing an API

| Need | Recommended API |
| --- | --- |
| Fetch measurement bits once | `Circuit.sample` |
| Fetch measurement bits repeatedly | `compile_sampler().sample` |
| Fetch each detector bit | `sample_detectors` |
| Only count discard rate / logical error rate | `sample_counts` |
| Repeat large statistics jobs | `compile_counts_sampler().sample` |
| Reduce returned array memory | `bit_packed=True` |

## Supported Stim Operations

The interface parses Stim-style text, but it is not a transparent replacement
for the full Stim grammar. Supported operations include:

- metadata and structure: `REPEAT`, `TICK`, `QUBIT_COORDS`, `SHIFT_COORDS`,
  `DETECTOR`, `OBSERVABLE_INCLUDE`;
- single-qubit Clifford operations: `I`, `H` and H/C axis variants, `S`/`S_DAG`,
  `SQRT_X/Y/Z` and dagger variants, `X`, `Y`, `Z`;
- two-qubit Clifford operations: `CX`/`CNOT`, `CY`, `CZ`, `SWAP`, `ISWAP`,
  `SQRT_XX/YY/ZZ` and dagger variants, plus axis-controlled parser variants;
- non-Clifford and rotations: `T`, `T_DAG`, `R_X/Y/Z`, `R_XX/YY/ZZ`,
  `R_PAULI`, `U`/`U3`, `SPP`/`SPP_DAG`;
- measurements and resets: `M`/`MZ`, `MX`, `MY`, `MR*`, `R`/`RX/RY/RZ`,
  `MPP`, `MXX/MYY/MZZ`, `MPAD`;
- noise: `X/Y/Z_ERROR`, `I/II_ERROR`, `DEPOLARIZE1/2/3`,
  `PAULI_CHANNEL_1/2/3`, `CORRELATED_ERROR`/`E`,
  `ELSE_CORRELATED_ERROR`, `HERALDED_ERASE`,
  `HERALDED_PAULI_CHANNEL_1`;
- Pauli feedback controlled by measurement records. Sweep-controlled operations
  are not supported and are rejected by the parser.

Operation parameters, target counts, and available control directions are still
validated by the parser. Unsupported operations raise `SymFTError`; the error
message includes the specific operation or violated constraint.
Rotation parameters follow Clifft's half-turn convention. Multiply each angle
parameter by pi to obtain radians; this applies to every parameter of `U` and
`U3` as well as the Pauli rotation gates.

## Full API Reference

### Module-Level Members

```python
symft.__version__ -> str
symft.read_stim_file(path) -> Circuit
symft.sample(circuit, shots=1, **kwargs) -> numpy.ndarray
symft.simd_backend() -> str
symft.cuda_enabled() -> bool
symft.active_cuda_backend() -> str
```

### `Circuit`

```python
symft.Circuit(text=None, path=None)

Circuit.compile_sampler(
    batch=False, batch_size=0, sample_chunk_shots=0
) -> CompiledMeasurementSampler

Circuit.compile_counts_sampler(
    batch=True,
    observable=0,
    postselect_detectors=False,
    batch_size=0,
    sample_chunk_shots=0,
    threads=1,
    batch_mask_threshold_denominator=2,
    cuda=False,
    cuda_mode="gpu",
    shots_per_launch=0,
    threads_per_block=0,
) -> CompiledCountsSampler

Circuit.sample(
    shots=1,
    seed=1,
    batch=False,
    bit_packed=False,
    batch_size=0,
    sample_chunk_shots=0,
) -> numpy.ndarray

Circuit.sample_counts(
    shots=1,
    seed=1,
    batch=True,
    observable=0,
    postselect_detectors=False,
    batch_size=0,
    sample_chunk_shots=0,
    threads=1,
    batch_mask_threshold_denominator=2,
    cuda=False,
    cuda_mode="gpu",
    shots_per_launch=0,
    threads_per_block=0,
) -> dict

Circuit.sample_detectors(
    shots=1, seed=1, bit_packed=False
) -> numpy.ndarray
```

Read-only properties: `num_qubits`, `num_measurements`, `num_detectors`,
`num_observables`, `num_observable_includes`, `detectors`, and `observables`.

### `CompiledMeasurementSampler`

```python
sampler.sample(shots=1, seed=1, bit_packed=False) -> numpy.ndarray
sampler.sample_detectors(shots=1, seed=1, bit_packed=False) -> numpy.ndarray
```

Read-only properties: `num_qubits`, `num_measurements`, `num_detectors`, and
`max_active_qubits`.

### `CompiledCountsSampler`

```python
sampler.sample(shots=1, stream_id=None) -> dict
```

Read-only properties: `info` and `preprocessing_timing`.

IDEs and static type checkers can read the package's `.pyi` and `py.typed`
files. At runtime, `help(...)` and `inspect.signature(...)` can inspect native
method signatures.

## Exceptions, Bounds, and Threading

- Stim parsing errors and most C++ runtime errors are mapped to
  `symft.SymFTError`, which inherits from `RuntimeError`.
- Python argument type errors raise `TypeError`; negative numeric arguments
  raise `ValueError`; values outside the C++ `int` range raise `OverflowError`.
- `seed` and `stream_id` must be non-negative integers convertible to `uint64`.
- `shots=0` is valid, and rates return `nan` when their denominator is 0.
- Native computation usually releases the GIL, but concurrent calls into one
  `CompiledCountsSampler` are serialized by the wrapper.
- Treat `Circuit` and compiled samplers as immutable configuration objects; do
  not manually call `__init__` again.
- Returned metadata lists and dictionaries are new Python objects. Mutating them
  does not modify the circuit.

Recommended service-layer error handling:

```python
try:
    circuit = symft.read_stim_file("circuit.stim")
except (TypeError, symft.SymFTError) as exc:
    print(f"cannot load circuit: {exc}")
```

## Development and Testing

Run the full test suite:

```bash
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run examples:

```bash
PYTHONPATH=src python examples/basic.py
PYTHONPATH=src python examples/test_stim.py path/to/circuit.stim --shots 10000 --threads 4
```

Build a wheel and source distribution:

```bash
python -m pip wheel . --no-deps -w dist
python setup.py sdist
```

The Python directory depends on `../cpp/src`. The generated sdist copies that
C++ source tree into the archive so it can be built independently from the
source package. Compiler warnings such as "loop not vectorized" do not indicate
a failed build; query the backend actually in use with `simd_backend()`.
