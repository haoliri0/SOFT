# SymFT

SymFT is an exact, high-throughput Python/C++ simulator for noisy, adaptive Clifford-dominated quantum circuits.
It accepts a Stim-style circuit format extended with non-Clifford Pauli rotations and supports stochastic Pauli noise, mid-circuit measurements, measurement-record-controlled feedback, detectors, observables, and postselection.

SymFT is the second-generation successor to [SOFT](https://github.com/haoliri0/SOFT).
It replaces SOFT's independent per-shot circuit evolution with a shared symbolic Clifford–Pauli frame and a compiled stabilizer-coordinate sampling plan.
The implementation provides single-core and multithreaded CPU sampling, runtime-dispatched SIMD kernels, an optional CUDA backend, and a typed Python API.

- [Python interface guide](python/README.md)
- [Benchmark circuits and methodology](benchmark/README.md)

## From SOFT to SymFT

SOFT evolves a private generalized-stabilizer tableau, sparse coefficient map, and noise history for every shot.
SymFT replaces this per-shot evolution with three design features:

- **Symbolic Clifford–Pauli frame factorization.**
  SymFT pulls Pauli rotations and measurement projectors through a symbolic Clifford-Pauli frame.
  Shot-dependent Pauli noise and feedback remain as symbolic signs, so their effects can be
  evaluated without replaying the Clifford circuit for every shot.
- **Adaptive stabilizer-coordinate planning.**
  A shared tableau defines the basis, while a dynamically sized dense vector stores only the active non-stabilizer degrees of freedom.
  SymFT resolves basis changes once and emits direct multi-coordinate sampling instructions, avoiding per-shot tableau updates and Clifford localization.
- **Compile once, sample many times.**
  Preprocessing emits a compact sampling instruction stream that can be reused across shots.
  Sampling evaluates symbolic signs, updates active coefficients and measurement records, and accumulates detector and observable results without revisiting the original
  circuit or reconstructing the planning stabilizer tableau.

## Performance

The following results are taken from the current
[benchmark suite](benchmark/README.md). They report attempted shots per second
through the public sampling paths, not isolated kernel rates. CPU measurements
use one pinned core of an Intel Xeon Gold 5218R and complex FP64 arithmetic.
Each entry is the arithmetic mean of two sampling-only runs of approximately
60 seconds; compilation and planning are excluded.

For pure-Clifford circuits, the most relevant baseline is
[Stim](https://github.com/quantumlib/Stim). For magic-state cultivation (MSC),
the most relevant CPU baseline is
[Clifft](https://github.com/unitaryfoundation/clifft).

| Regime | Circuit | Baseline | SymFT | Speedup |
| --- | --- | ---: | ---: | ---: |
| Pure Clifford | Surface code `d=7, r=7` | Stim: 816.93k | **2.06M** | **2.52×** |
| Pure Clifford | Surface code `d=9, r=9` | Stim: 350.34k | **899.20k** | **2.56×** |
| MSC | `d=3` cultivation | Clifft: 502.3k | **1.762M** | **3.51×** |
| MSC | `d=5` cultivation | Clifft: 42.67k | **107.35k** | **2.52×** |

The pure-Clifford circuits do not use detector postselection. The MSC circuits
postselect all detectors and contain the injection and cultivation stages, but
not the subsequent Clifford-only escape stage. The throughput suffixes `M` and
`k` denote `10^6` and `10^3` shots/s.

The GPU results on an NVIDIA GeForce RTX 4090 also include [Tsim](https://github.com/QuEraComputing/tsim):

| Circuit | SOFT FP64 | Tsim CUDA | SymFT CUDA FP64 | SymFT vs SOFT |
| --- | ---: | ---: | ---: | ---: |
| MSC `d=3` | 331.50k | 26.93k | **68.04M** | **205×** |
| MSC `d=5` | 5.03k | DNC | **2.61M** | **518×** |

SOFT and SymFT use FP64. Tsim's effective CUDA path retains FP32 and complex64
intermediates despite JAX x64 mode, so its result is not a precision-matched
comparison. DNC means compilation did not finish within 300 seconds.

The compared tools expose different output forms, so these numbers compare the
tested public sampling paths rather than identical-width output kernels. See
the [benchmark documentation](benchmark/README.md) for the circuits,
configuration, hardware details, and measurement protocol.

## Installation

The Python package requires Python 3.9 or newer, NumPy 1.20 or newer, and a
C++20 compiler. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ./python
```

The extension compiles the C++ sources directly; CMake is not required for the
Python build.

To include the optional CUDA counts backend, install a CUDA toolkit with
`nvcc` and run:

```bash
SYMFT_PY_ENABLE_CUDA=1 python -m pip install -e ./python
```

See the [Python interface guide](python/README.md#cuda-counts-backend) for CUDA
architecture, precision, and execution-mode options.

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

Use `Circuit.sample` when full measurement records are needed,
`Circuit.sample_detectors` for detector records, and `Circuit.sample_counts`
or a compiled counts sampler for high-throughput aggregate statistics.

## Supported circuit model

The frontend supports:

- Clifford gates and Pauli-product operations;
- `T`, `T_DAG`, arbitrary-axis Pauli rotations, and `U`/`U3`;
- stochastic Pauli channels and correlated errors;
- Pauli measurements, resets, and measurement-record-controlled feedback;
- repeat blocks, detectors, observables, and detector postselection.

The format is a substantial Stim-style subset with non-Clifford extensions,
not a drop-in parser for every Stim instruction. Following Clifft's convention,
a rotation parameter `alpha` represents `alpha * pi` radians; for example,
`R_Z(0.02)` rotates by `0.02 * pi`.

See the [Python interface guide](python/README.md) for the complete operation
list, API reference, result schemas, and error behavior.

## C++ build

CMake 3.20 or newer and a C++20 compiler are required:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYMFT_CPP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the simple circuit summary tool:

```bash
./build/cpp/symft_cli benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim 1000
```

Run the configurable single-shot/batch rate harness:

```bash
./build/cpp/symft_rate_bench \
  --sampler both \
  --circuit benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim \
  --shots 1000000 \
  --batch-size auto \
  --threads 1
```

The library target is `symft_cpp`, and its public source-tree headers are under
`cpp/src`. The prepared counts API can be used directly:

```cpp
#include "frontend/stim_prepared_sampler.hpp"

#include <cstdint>
#include <iostream>

int main() {
    symft::CircuitSamplingOptions options;
    options.observable = 0;
    options.threads = 4;
    options.postselect_detectors = true;

    auto sampler =
        symft::prepare_batch_sampler_from_stim_file(
            "benchmark/circuit/msc_d3_inject_cultivate_p1e-3.stim", options);
    auto run = sampler.sample(1'000'000, std::uint64_t{1});

    std::cout << run.counts.discarded << '\n';
    std::cout << run.counts.logical_errors << '\n';
}
```

Enable the C++ CUDA target with:

```bash
cmake -S . -B build-cuda \
  -DSYMFT_CPP_ENABLE_CUDA=ON \
  -DSYMFT_CPP_CUDA_REAL_DOUBLE=OFF
cmake --build build-cuda -j
```

## Development

Run the C++ tests:

```bash
cmake -S . -B build -DSYMFT_CPP_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the Python tests:

```bash
cd python
python setup.py build_ext --inplace
PYTHONPATH=src python -m unittest discover -s tests -v
```

Repository layout:

```text
cpp/src/core/       Pauli algebra, symbolic expressions, and frames
cpp/src/circuit/    Circuit IR and lowering
cpp/src/factored/   Stabilizer-coordinate state and planner
cpp/src/sampler/    Single-shot, batch, and prepared samplers
cpp/src/frontend/   Stim-style parser and sampling frontend
cpp/src/simd/       Scalar and CPU SIMD kernels
cpp/src/cuda/       Optional CUDA sampler
python/src/symft/   Native Python binding, type hints, and package API
benchmark/          Stim fixtures and benchmark inputs
cpp/tests/          C++ correctness tests
python/tests/       Python interface tests
```

## AI Acknowledgement

The project authors used ChatGPT Pro (GPT-5.5/5.6) and OpenAI Codex for implementation, exploratory coding, preliminary literature searches, and documentation editing.
The project authors reviewed and verified the resulting code, tests, benchmark results, and documentation, and take full responsibility for the contents of this repository.

## License

SymFT is licensed under the [Apache License 2.0](LICENSE).
The Clifft-derived benchmark inputs retain their original attribution and are accompanied by a separate [Apache-2.0 license](benchmark/LICENSE-Clifft-paper).