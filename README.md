# SOFT

SOFT is an exact Python/C++ simulator for noisy, adaptive quantum circuits dominated
by Clifford operations. It supports non-Clifford Pauli rotations, stochastic
noise, mid-circuit measurements, feedback, detectors, and postselection.

The current implementation, SymFT, is the second generation of SOFT. It builds
a shared symbolic Clifford–Pauli frame and a sampling plan once, then reuses
them across shots. CPU sampling supports single-threaded and multithreaded
execution, with an optional CUDA backend. The Python package is called `symft`;
the original implementation is kept in [legacy/softv1](legacy/softv1/).

**The `symft_26_10_08` update makes d=5 magic-state cultivation sampling about
3× faster on one CPU core and 10× faster on an RTX 4090. The single-core
magic-state distillation benchmark is 3.3× faster.** The new compiled CPU and CUDA JIT paths
are opt-in. See [performance](#performance) for the comparison and
[quick start](#quick-start) for how to use them.

- [Python interface guide](python/README.md)
- [Benchmark circuits and methodology](benchmark/README.md)
- [Documentation](docs/README.md) and [changelog](CHANGELOG.md)

## From SOFT to SymFT

The original SOFT evolves a separate generalized-stabilizer state for each
shot. SymFT shares the Clifford part of that work across all shots:

- A symbolic Clifford–Pauli frame carries the Clifford evolution, Pauli noise,
  and measurement feedback.
- A dense coefficient vector stores the active non-stabilizer degrees of
  freedom. Basis changes are planned once rather than repeated for every shot.
- Sampling evaluates the symbolic signs, updates the active coefficients,
  and produces measurement, detector, and observable results.

The new CPU and GPU paths also move detector checks earlier, so rejected shots
can stop before doing unnecessary state evolution. When the circuit permits
it, they use real amplitudes instead of complex ones. These changes do not
truncate the state or approximate the noise model.

## Installation

Install the published Python package with:

```bash
python -m pip install symft
```

To use the CPU and CUDA updates described here, build from this repository.
A Python source build requires Python 3.9+, NumPy 1.20+, and a C++20 compiler;
CMake is not required.

```bash
git clone https://github.com/haoliri0/SOFT.git
cd SOFT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./python
```

Source builds use host-native CPU optimizations by default. Set
`SYMFT_PY_NATIVE=0` if you need a portable build.

### CUDA

With a CUDA toolkit, NVRTC, and a compatible NVIDIA driver installed:

```bash
SYMFT_PY_ENABLE_CUDA=1 SYMFT_PY_CUDA_REAL_DOUBLE=1 \
  python -m pip install ./python
```

This selects FP64, as used in the benchmarks below. CUDA builds otherwise
default to FP32. Architecture and build options are described in the
[Python interface guide](python/README.md#cuda-counts-backend).

## Quick start

```python
import symft

circuit = symft.Circuit("""
H 0
T 0
M 0
OBSERVABLE_INCLUDE(0) rec[-1]
""")

sampler = circuit.compile_counts_sampler(batch=True, observable=0)
result = sampler.sample(shots=100_000, stream_id=42)

print(result["logical_error_rate"])
print(result["timing"])
```

Use `Circuit.sample` for full measurement records, `Circuit.sample_detectors`
for detector records, and `Circuit.sample_counts` or a prepared counts sampler
for aggregate statistics. Reference-normalized counts and non-destructive
`EXP_VAL` probes are also available; see the [API guide](python/README.md).

`discard_rate` is the fraction of attempted shots rejected by the detectors.
`logical_error_rate` is the fraction of accepted shots whose selected
observable differs from its reference value (zero by default). A rate with a
zero denominator is reported as `nan`.

### Faster postselected CPU sampling

From the repository root:

```python
circuit = symft.read_stim_file(
    "benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim"
)
sampler = circuit.compile_counts_sampler(
    cpu_backend="compiled",
    postselect_detectors=True,
    threads=1,
)
result = sampler.sample(shots=1_000_000, stream_id=42)
print(result["discard_rate"], result["logical_error_rate"])
```

The compiled CPU path currently handles single-worker, dense-state counts
sampling with detector postselection and at most 10 active qubits. Other
configurations, including `EXP_VAL` circuits, use the existing CPU executor.
Check `sampler.info["cpu_compiled"]` and `sampler.info["cpu_fallback_reason"]`
to see which path was selected. Different backends need not produce identical
shots from the same seed.

CUDA JIT has its own activation flags. See the
[CPU guide](docs/optimization/CPU.md) and [CUDA guide](docs/optimization/CUDA.md)
for configuration and implementation details.

## Performance

### symft_26_10_08

The measurements below use an Intel Xeon Gold 5218R for single-core CPU
sampling and an RTX 4090 for GPU sampling. All use FP64 and postselection on
all detectors. MSC uses physical noise `p=0.001` and includes injection and
cultivation, but not the later escape stage. Distillation uses the unchanged
85-qubit [Z-basis benchmark circuit](benchmark/circuit/distillation.stim),
with its own noise settings.

| Workload | Previous path | Optimized path | Speedup |
| --- | ---: | ---: | ---: |
| MSC d=3, one CPU core | 1.800 M shots/s | 2.019 M shots/s | 1.12× |
| MSC d=5, one CPU core | 121.3 k shots/s | 360.4 k shots/s | 2.97× |
| Distillation, 85 qubits, one CPU core | 1.149 M shots/s | 3.791 M shots/s | 3.30× |
| MSC d=5, RTX 4090 | 2.337 M shots/s | 24.672 M shots/s | 10.56× |

CPU rows compare pre-update main (`c89b985`) with the compiled backend, using
matching Release/native/LTO build settings. Cultivation rates are medians of
three runs; distillation rates are medians of seven runs, each with 8 million
attempted shots.

The GPU row compares the existing CUDA path with JIT in the same FP64 build,
each over three 10-million-shot runs, using the average sampling time.
Rates count attempted shots and exclude preparation and JIT compilation.

These results are from the October 9 measurements. Speedups depend on the
circuit and hardware. See the [cultivation measurements](docs/MAIN_INTEGRATION.md)
for run sizes, timing ranges, and correctness checks, and the
[distillation measurements](benchmark/results/distillation_cpu_20261009.json)
for per-run timings and build settings. Earlier comparisons and
the separate 50-billion-shot CPU/GPU runs are in
[performance details](docs/PERFORMANCE.md).

### Earlier cross-simulator benchmarks

For context, the original SymFT CPU benchmarks compared against Stim for
pure-Clifford circuits and Clifft for magic-state cultivation:

| Circuit | Baseline | SymFT before this update | Speedup |
| --- | ---: | ---: | ---: |
| Pure-Clifford surface code, d=7, r=7 | Stim: 816.93 k/s | 2.06 M/s | 2.52× |
| Pure-Clifford surface code, d=9, r=9 | Stim: 350.34 k/s | 899.20 k/s | 2.56× |
| MSC d=3 | Clifft: 502.3 k/s | 1.762 M/s | 3.51× |
| MSC d=5 | Clifft: 42.67 k/s | 107.35 k/s | 2.52× |

These are the results from the previous README, not new measurements with the
compiled CPU backend. They use one pinned Xeon Gold 5218R core, complex FP64,
and the mean of two approximately 60-second sampling runs. Only the MSC cases
use postselection. The [original comparison](docs/HISTORICAL_BENCHMARKS.md)
also includes GPU results against SOFT v1 and Tsim, with their precision and
output-format differences.

## Supported circuit model

The frontend accepts a Stim-style format with non-Clifford extensions:

- Clifford gates and Pauli-product operations;
- `T`, `T_DAG`, arbitrary-axis Pauli rotations, and `U`/`U3`;
- stochastic Pauli channels and correlated errors;
- Pauli measurements, resets, and measurement-record-controlled feedback;
- repeat blocks, detectors, observables, and postselection.

Not every Stim instruction is supported. Rotation angles are specified in
half-turns: `R_Z(0.02)` means `0.02 * pi` radians. See the
[operation reference](python/README.md#supported-stim-operations) for the full
list.

## C++ build

CMake 3.20+ and a C++20 compiler are required:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYMFT_CPP_BUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

Run the single-core compiled counts benchmark:

```bash
./build/cpp/symft_rate_bench \
  --circuit benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim \
  --shots 1000000 --sampler batch --threads 1 \
  --postselect-detectors --cpu-backend compiled
```

The library target is `symft_cpp`, with headers under `cpp/src`. A prepared
sampler can also be used directly:

```cpp
#include "frontend/stim_prepared_sampler.hpp"

#include <cstdint>
#include <iostream>

int main() {
    symft::CircuitSamplingOptions options;
    options.threads = 1;
    options.postselect_detectors = true;
    options.cpu_compiled = true;

    auto sampler = symft::prepare_batch_sampler_from_stim_file(
        "benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim", options);
    auto run = sampler.sample(1'000'000, std::uint64_t{42});

    std::cout << run.counts.discarded << '\n';
    std::cout << run.counts.logical_errors << '\n';
}
```

For CUDA FP64 on an RTX 4090:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=ON -DSYMFT_CPP_CUDA_REAL_DOUBLE=ON \
  -DSYMFT_CPP_CUDA_ARCH=89
cmake --build build-cuda -j 8
```

Choose the architecture for your own GPU instead of `89` when needed. Further
options are in the [CUDA guide](docs/optimization/CUDA.md).

## Development

Run the C++ tests with CTest as above. For the Python tests:

```bash
cd python
python setup.py build_ext --inplace
python -m pip install pytest
PYTHONPATH=src python -m pytest tests -q
```

The implementation is organized under `cpp/src`: `core` contains Pauli algebra
and symbolic frames; `factored` contains the stabilizer-coordinate planner;
`sampler`, `simd`, and `cuda` contain the execution backends. Python bindings
live in `python/src/symft`, and benchmark inputs live in `benchmark/circuit`.

Run `python3 tools/check_documentation.py` from the repository root to check
documentation links and benchmark records.

## AI acknowledgement

The authors used ChatGPT and OpenAI Codex for implementation, exploratory
coding, preliminary literature searches, and documentation editing. The authors
reviewed the resulting code, tests, benchmarks, and documentation, and take
responsibility for the contents of this repository.

## License

SOFT v2 and its SymFT implementation are licensed under the
[Apache License 2.0](LICENSE). The Clifft-derived benchmark inputs retain
their original attribution and [separate license](benchmark/LICENSE-Clifft-paper).
