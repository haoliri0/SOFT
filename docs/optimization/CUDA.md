# CUDA JIT counts backend

This opt-in backend specializes a SymFT sampling plan for SOFT v2 aggregate
counts. It does not accelerate the full measurement-matrix return API.

## Build and activate

For the tested RTX 4090 FP64 configuration:

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=ON -DSYMFT_CPP_CUDA_REAL_DOUBLE=ON \
  -DSYMFT_CPP_CUDA_ARCH=89
cmake --build build-cuda -j 8

export SYMFT_CUDA_JIT=1
export SYMFT_CUDA_PACKED_NOISE=1
export SYMFT_CUDA_SCALAR_NOISE=1
export SYMFT_JIT_SYMBOLIC=1
export SYMFT_JIT_ROW_REDUCE=1
export SYMFT_JIT_COMPLEX=0
export SYMFT_JIT_REGS=128
export SYMFT_JIT_THREADS=128
export SYMFT_JIT_CACHE="$PWD/jit-cache"

build-cuda/cpp/symft_cuda_rate_bench \
  --circuit benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim \
  --postselect-detectors --gpu-presample-expressions \
  --shots-per-launch 524288 --shots 50000000 --stream-id 200
```

Choose your own GPU architecture instead of 89 when necessary. CPU-only builds
do not require CUDA. CUDA-enabled builds now link NVRTC and the CUDA driver
library as well as the runtime. The driver and toolkit must support the target
architecture. JIT compilation produces a cubin for the actual runtime device.

FP64 is a **build option**. The original CUDA default is FP32; none of the
environment flags above changes that. The measured results use FP64 only.

For Python, build this checkout with `SYMFT_PY_ENABLE_CUDA=1` and
`SYMFT_PY_CUDA_REAL_DOUBLE=1`, set the environment above before construction,
then use:

```python
import symft

circuit = symft.read_stim_file("benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim")
sampler = circuit.compile_counts_sampler(
    cuda=True, cuda_mode="gpu_presample_expressions",
    postselect_detectors=True, observable=0,
    shots_per_launch=524288, threads_per_block=128,
)
result = sampler.sample(shots=50_000_000, stream_id=200)
```

## Eligibility and failures

JIT selection requires `SYMFT_CUDA_JIT=1`, the GPU-presampled-expression mode,
and `max_k <= 10`, with no expectation probes and counts-only output.
Other modes, larger widths, `EXP_VAL` circuits, and full record/expectation
requests use the existing CUDA executor;
that executor still has its own device/resource limits. An incompatible real
gauge selects complex JIT arithmetic, not a lower-precision approximation.
NVRTC compilation or driver/module errors are reported as errors; there is no
claim of an automatic successful fallback after an arbitrary JIT failure.

Set all compilation/layout flags before constructing a sampler and do not
change them during its lifetime. In particular, changing packed-expression
layout mid-run is unsupported. `SYMFT_JIT_THREADS` controls the JIT state
kernel independently of the public `threads_per_block` option used elsewhere.

## Optimizations

1. **Circuit specialization.** [`cuda_jit.cpp`](../../cpp/src/cuda/cuda_jit.cpp)
   emits CUDA with instruction kinds, masks, indices, and phase coefficients
   resolved at compile time, avoiding per-shot interpretation.
2. **Warp-local state.** One warp handles one shot. Small active states stay
   primarily in registers. A structurally proven phase gauge uses real FP64
   amplitudes where valid; otherwise the full complex representation remains.
3. **Noise influence and packing.** The exogenous GF(2) influence table is
   transposed so sampled events XOR their actual effects. Expressions are
   packed per shot. Small noise vectors use one thread per shot, avoiding
   shared atomic updates; larger vectors retain the existing applicable path.
4. **Classical elimination.** Affine record/condition expansion separates
   noise from Born leaves. Ready detectors move earlier, including noise-only
   entrance checks. Row operations preserve the joint zero constraint.
5. **Selected-branch execution.** Projection constructs only the chosen state.
   Unused classical branch values are omitted while preserving the required
   RNG counter advancement. State-affecting measurements are not skipped.
6. **Device count reduction.** Discard/error flags are reduced on the GPU;
   two 64-bit counters are copied back. Accepted counts follow from attempted
   minus discarded. Reported throughput includes this aggregation/transfer.

The exogenous noise optimization preserves the original Philox/geometric/
categorical sampling model. Matching CPU/GPU seeds still does not imply a
common measurement tape. Bitwise noise checks and independent state/reference
tests complement, rather than follow from, distribution-level agreement.

## Flags and inspection

| Variable | Purpose |
| --- | --- |
| `SYMFT_CUDA_JIT=1` | Enable eligible JIT selection; off by default |
| `SYMFT_CUDA_PACKED_NOISE=1` | Packed per-shot expressions with JIT |
| `SYMFT_CUDA_SCALAR_NOISE=1` | Single-thread sparse-noise generation for 1–8 expression words |
| `SYMFT_JIT_SYMBOLIC=1` | GF(2) expansion, detector scheduling, live classical branches |
| `SYMFT_JIT_ROW_REDUCE=1` | Reduce simultaneous entrance detector checks |
| `SYMFT_JIT_COMPLEX=1` | Force complex JIT for ablation/reference checks |
| `SYMFT_JIT_REGS=128` | NVRTC register cap used in the measured configuration |
| `SYMFT_JIT_THREADS=128` | JIT block size; warp multiple in [32, 1024] |
| `SYMFT_JIT_CACHE=path` | Optional local cubin cache keyed by source/options/device/toolkit identity |
| `SYMFT_JIT_DUMP=path` | Dump generated source, compilation log, cubin, and resource data |
| `SYMFT_VALIDATE_NOISE=1` | Expensive bitwise comparison with original noise implementation; validation only |
| `SYMFT_CUDA_PROFILE=1` | Diagnostic stage timing; not production benchmarking |

Use a trusted local cache directory; generated binaries are build artifacts,
not source files to commit. The cache is not advertised as a cross-process
transactional store. Avoid simultaneous first-time writers to the same key
or use separate directories for concurrent validation jobs.

## Timing and evidence

Parsing, symbolic planning, NVRTC compilation, and initial allocation are
one-time costs and excluded from steady-state sampling rates. A cold d=5 JIT
can take seconds; report preparation/process wall time for short jobs rather
than treating that cost as free. A cache hit avoids recompilation but does not
make all preparation cost zero.

On the tested RTX 4090, the d=5 50B run sustained 24.735 M attempted shots/s;
the same-input pre-optimization remeasurement was about 2.619 M/s (roughly
9.44x, with different run durations). The d=3 1B run reported 147.224 M/s but
lasted only about 6.79 s, so it is not equivalent stability evidence.
See [counts, timing contracts, intervals, and caveats](../PERFORMANCE.md).
