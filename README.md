# SOFT

**A high-performance simulator for fault-tolerant quantum circuits.**

## symft_26_10_08: faster CPU and CUDA sampling

**MSC d=5: 3.04x single-core CPU throughput and approximately 9.44x CUDA
throughput versus old SymFT.** The CPU backend gains a further **28.9%** over
our September optimized implementation in the fresh October comparison.

| FP64 workload | Old SymFT | Optimized SymFT | Speedup | Evidence |
| --- | ---: | ---: | ---: | --- |
| MSC d=5, one CPU core | 114,864 shots/s | **348,896 shots/s** | **3.04x** | Fresh 2026-10-08, seven-run medians |
| MSC d=3, one CPU core | 1.787 M shots/s | **2.131 M shots/s** | **1.19x** | Fresh 2026-10-08, seven-run medians |
| MSC d=5, RTX 4090 | 2.619 M shots/s | **24.735 M shots/s** | **9.44x** | Archived September GPU measurements |

CPU: Xeon Gold 5218R, one pinned worker, Release/native+LTO. The old CPU
baseline is unmodified SymFT commit `e86c6a9`, rebuilt with the same settings.
All rows use canonical MSC injection+cultivation, `p=0.001`, FP64, all-detector
postselection, and **attempted** shots/s; preparation is excluded. GPU compares
three old 50M timing replays with the optimized 50B run, not equal-duration
paired repeats. These are workload-specific gains, **not a comparison to SOFT v1**.
[Full measurements, ranges, provenance, and limitations](docs/PERFORMANCE.md).

To obtain these accelerated paths, explicitly enable
[`cpu_backend="compiled"`](docs/optimization/CPU.md) or
[CUDA JIT with FP64](docs/optimization/CUDA.md). Existing defaults are preserved.
This release adds AVX-512 real-state CPU kernels, retaining AVX2/scalar fallbacks.
Release name: **`symft_26_10_08`**; Python package version: **`2026.10.8`**.
Prepared for review; no GitHub/PyPI publication is implied.

This checkout ports the optimizations onto `main` commit `c89b985` on branch
**`symft-26-10-08`**, ready for a normal PR to `main` after review and commit.
The table above describes the original optimization experiments against old
SymFT `e86c6a9`, not a fresh speed comparison against current `main`.
See [main integration and fresh checks](docs/MAIN_INTEGRATION.md).

## About SOFT and SymFT

SOFT is the project and software family. **SOFT v2 is powered by SymFT**, its
second-generation symbolic/compiled sampling architecture. SymFT names the
method and implementation architecture; it does not restrict SOFT to a single
future algorithm. The repository remains `haoliri0/SOFT`, and the Python
package and import remain `symft`.

SOFT v2 provides exact, finite-precision Python/C++ simulation of noisy,
adaptive Clifford-dominated circuits: stochastic Pauli noise, non-Clifford
Pauli rotations, mid-circuit measurements, record-controlled feedback,
detectors, observables, and postselection. It offers CPU sampling and an
optional CUDA counts backend. Exact here means no state truncation or
sampling-model approximation, not infinite-precision arithmetic.

The integration retains main's reference-normalized CPU counts, non-destructive
`EXP_VAL` sampling on CPU/CUDA, tableau/frame factorization, CUDA global-workspace
fallback, the [SOFT v1 archive](legacy/softv1/), and wheel release workflow.
Compiled CPU counts normalize detector/observable references before optimization;
expectation probes use the original executor. CUDA record/expectation requests
also retain the original executor even when JIT is enabled.

- [Documentation index](docs/README.md)
- [Python interface and installation](python/README.md)
- [CPU optimization](docs/optimization/CPU.md) and [CUDA optimization](docs/optimization/CUDA.md)
- [Measured performance and statistical validation](docs/PERFORMANCE.md)
- [Benchmark inputs and methodology](benchmark/README.md)
- [SOFT v1, naming, and compatibility](docs/PROJECT.md)
- [Release changes](CHANGELOG.md) and [maintainer review checklist](docs/RELEASE_CHECKLIST.md)

## Installation

To use the changes in this branch, build **this checkout**. An already installed
PyPI wheel does not necessarily contain these unreleased optimizations.
Python 3.9+, NumPy 1.20+, and a C++20 compiler are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./python
```

For a portable CPU build, set `SYMFT_PY_NATIVE=0`. The default source build
enables host-native optimization and is not a portable wheel configuration.

CUDA is optional. For the FP64 configuration used in the validation results:

```bash
SYMFT_PY_ENABLE_CUDA=1 SYMFT_PY_CUDA_REAL_DOUBLE=1 \
  python -m pip install ./python
```

This requires a CUDA toolkit including NVCC and NVRTC, the CUDA driver library,
and a compatible NVIDIA driver/device. The CUDA precision default is otherwise
FP32; enabling JIT does not automatically select FP64. See the
[build options](python/README.md#installation-and-build).

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
print(result["discard_rate"])
print(result["logical_error_rate"])
```

`discard_rate = discarded / attempted`; `logical_error_rate = logical_errors /
accepted`. A zero denominator produces `nan`. Counts partition attempted
shots into accepted and discarded even when early detector rejection is
disabled. `postselect_detectors=True` enables that early rejection.

For a compatible counts-only CPU workload, explicitly select the new backend:

```python
sampler = circuit.compile_counts_sampler(
    cpu_backend="compiled",
    postselect_detectors=True,
    threads=1,
)
print(sampler.info["cpu_compiled"], sampler.info["cpu_fallback_reason"])
result = sampler.sample(shots=100_000, stream_id=42)
```

The existing CPU backend remains the default. Compiled CPU sampling currently
requires one worker, dense state, detector postselection, and `max_k <= 10`;
unsupported configurations fall back to the existing backend. Its measurement
RNG contract is `cpu-shot-v1`, not seed-identical to the existing CPU path.
[CPU details and limitations](docs/optimization/CPU.md).

CUDA JIT is also opt-in and currently specializes compatible
`gpu_presample_expressions` counts workloads with `max_k <= 10`.
[CUDA activation and tuning](docs/optimization/CUDA.md).

## Archived large-shot validation

These are **archived measurements of the optimized source**, not new long runs
performed while preparing this branch. All entries below use FP64, the
canonical MSC injection+cultivation fixtures at `p=0.001`, observable 0,
all-detector postselection, and attempted shots/s. No escape stage is included.

| Workload and configuration | Measured throughput | Measurement |
| --- | ---: | --- |
| MSC d=5, one CPU socket, 40 processes | 2.959 M/s | 50 billion shots; 20 physical cores / 40 logical CPUs |
| MSC d=3, RTX 4090, CUDA JIT | 147.224 M/s | One 1-billion-shot run, approximately 6.79 s; not a long stability test |
| MSC d=5, RTX 4090, CUDA JIT | 24.735 M/s | 50 billion shots; approximately 33.69 minutes of sampling |

CPU: Intel Xeon Gold 5218R. Single-core rates exclude preparation; the socket
rate uses wall-clock sampling duration across persistent processes, excluding
their initial preparation/warmup. The GPU d=5 rate uses summed sampling-call
times, including noise, evolution, reduction, synchronization, and count
transfer, but excluding one-time preparation/JIT. These are distinct timing
contracts, not interchangeable end-to-end timings.
The CPU socket run predates the October AVX-512 update and is not a new 50B
validation of this release; its speed must not be scaled by the single-core gain.

The 40-process result is **not** native `cpu_backend="compiled", threads=40`
support. Do not multiply the single-core rate by logical CPU count to predict
socket throughput.

See [performance, exact counts, uncertainty, and provenance](docs/PERFORMANCE.md).
Older comparisons against SOFT v1, Stim, Clifft, and Tsim are retained separately
as a [historical snapshot](docs/HISTORICAL_BENCHMARKS.md).

## Architecture

SOFT v1 evolves an independent generalized-stabilizer state for each shot.
The SymFT architecture used by SOFT v2 instead:

1. Factors out a shared symbolic Clifford-Pauli frame.
2. Plans adaptive stabilizer-coordinate operations once per circuit.
3. Reuses that sampling plan across shots.

The opt-in CPU/GPU optimizations further resolve classical dependencies over
GF(2), schedule detector checks when their inputs become known, and use a
structurally proven real gauge where possible. Neither real-gauge FP64 nor
early rejection changes the intended circuit/noise/postselection model.
[Implementation details](docs/optimization/README.md).

## Circuit model

The frontend accepts a substantial Stim-style subset extended with non-Clifford
operations. It is not a drop-in parser for every Stim instruction. Supported
families include Clifford gates, Pauli-product operations, `T`/`T_DAG`,
Pauli rotations, `U`/`U3`, stochastic Pauli channels, correlated errors,
measurements, resets, record feedback, repeats, detectors, and observables.

Angles follow the half-turn convention: `R_Z(0.02)` means `0.02 * pi` radians.
See the [complete interface guide](python/README.md#supported-stim-operations).
The [d=7 proxy fixture](benchmark/README.md#why-the-distance-7-file-is-a-proxy)
is an unvalidated workload, not evidence of d=7 logical-error correctness.

## C++ build and tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSYMFT_CPP_BUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build --output-on-failure
```

For a portable scalar reference build, add `-DSYMFT_CPP_NATIVE=OFF`,
`-DSYMFT_CPP_ENABLE_AVX2=OFF`, and `-DSYMFT_CPP_ENABLE_AVX512=OFF`.

For CUDA FP64, use a separate directory:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=ON -DSYMFT_CPP_CUDA_REAL_DOUBLE=ON \
  -DSYMFT_CPP_CUDA_ARCH=89
cmake --build build-cuda -j 8
ctest --test-dir build-cuda --output-on-failure
```

Architecture 89 is the tested RTX 4090; select the architecture of your own GPU.
The C++ target and namespace remain `symft_cpp` and `symft`. No API/ABI rename
is implied by the SOFT project branding.

## Development

```bash
cd python
python setup.py build_ext --inplace
python -m pip install pytest
PYTHONPATH=src python -m pytest tests -q
```

Run `python3 tools/check_documentation.py` from the repository root to check
local documentation links, tracked artifact hygiene, and archived measurement
arithmetic. See [validation scope](docs/VALIDATION.md) and the
[review checklist](docs/RELEASE_CHECKLIST.md) before committing.

## AI acknowledgement

The project authors used ChatGPT Pro (GPT-5.5/5.6) and OpenAI Codex for
implementation, exploratory coding, preliminary literature searches, and
documentation editing. The project authors reviewed and verified the resulting
code, tests, benchmark results, and documentation, and take full responsibility
for the contents of this repository.

## License

SOFT v2, including its SymFT implementation, is licensed under
[Apache License 2.0](LICENSE). The Clifft-derived benchmark inputs retain their
original attribution and [separate license copy](benchmark/LICENSE-Clifft-paper).
