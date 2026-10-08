# Compiled CPU counts backend

This is an opt-in SOFT v2 executor using the SymFT sampling architecture.
`cpu_backend="legacy"` names the pre-existing v2 CPU executor, not SOFT v1.

## Selection and fallback

Python:

```python
import symft

circuit = symft.read_stim_file("benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim")
sampler = circuit.compile_counts_sampler(
    cpu_backend="compiled", postselect_detectors=True, threads=1,
)
print(sampler.info)
result = sampler.sample(shots=1_000_000, stream_id=901)
```

The same keywords work with `Circuit.sample_counts(..., seed=901)`.
In C++, set `CircuitSamplingOptions.cpu_compiled = true` and use the prepared
batch counts sampler.

| Condition | Behavior |
| --- | --- |
| `batch=False` or `cuda=True` with compiled CPU requested in Python | Explicit argument error |
| No detector postselection | Existing CPU executor; `postselection_disabled` |
| Requested worker count is not 1 | Existing CPU executor; `single_worker_only` |
| Maximum active width exceeds 10 | Existing CPU executor; `max_k_exceeds_10` |
| Product-component active-state mode selected | Existing CPU executor; `active_components_enabled` |
| Real-gauge proof fails | Compiled complex FP64; classical optimizations remain enabled |

Inspect `sampler.info["cpu_compiled"]`, `cpu_real_gauge`, and
`cpu_fallback_reason`; do not infer actual selection from the requested option.
Full measurement/detector matrix APIs are not accelerated by this backend.

## Implementation

### Classical plan and entrance filtering

[`cpu_sampling_plan.cpp`](../../cpp/src/sampler/cpu_sampling_plan.cpp) expands
records, conditions, detectors, and the selected logical observable into GF(2)
affine expressions. A detector is scheduled after its last Born branch becomes
available. Detectors with no Born dependence are checked before quantum work.
Entrance rows are reduced without changing their simultaneous zero-test.

Affine noise columns are evaluated and deduplicated in packed 64-shot words.
Only surviving bits enter the per-shot state executor. This keeps rejected
attempts in the original sampling population and retains the original external
noise generator. Live-operation and branch maps remove unused classical storage.
State-affecting measurements remain even if their recorded bit is unused.

### Real-gauge FP64 and vector kernels

[`real_active.cpp`](../../cpp/src/sampler/real_active.cpp) checks whether the
full plan preserves

```text
psi[b] = global_phase * i^parity(b & gauge) * real[b]
```

The plan proves compatibility for rotations, promotions, probabilities, and
projections, including gauge updates after removing a pivot. The test is
structural, not a tolerance-based approximation. Incompatible plans use the
compiled complex executor throughout.

For d=5, 1024 doubles occupy 8 KiB; state and scratch together occupy 16 KiB.
AVX2/FMA uses lane shuffles for low-bit pairs and contiguous blocks for
high-bit pairs; irregular measurement patterns retain a scalar/gather fallback.
Projection swaps state/scratch buffers instead of copying the result back.

Real-vector SIMD kernels are selected at compile time from the translation
unit's ISA support. A portable non-native build uses their scalar fallback;
the pre-existing complex kernels have their own separate runtime SIMD dispatch.
Do not assume a portable wheel reproduces native-build throughput.

### `symft_26_10_08`: wider real-state kernels

The October update adds AVX-512 rotations and promotions on supported native
builds, retaining the existing AVX2/FMA and scalar paths. Profiling the d=5
workload identified width-10 rotations as the dominant execution cost.

- Eight FP64 amplitudes are processed per vector. When the highest Pauli
  partner bit is below 3, partners stay within one eight-lane register;
  an XOR-index permutation supplies them without gathers.
- Higher partner bits use two contiguous eight-amplitude blocks. Both members
  of each disjoint pair are loaded before either is overwritten, then stored
  in their original basis order.
- Promotion splits the gauge parity into fixed low-three-bit lane signs and
  one high-bit parity per block. This replaces eight scalar parity/branch
  operations with one block parity and a mask selection.
- Born probability reductions, detector scheduling, external-noise generation,
  the `cpu-shot-v1` tape, and FP64 precision are unchanged. There is no
  state truncation, probability threshold, or approximate shortcut.

The guards require compiler AVX-512F support (and FMA for rotations); these
instructions are not injected into portable binaries. Native builds remain
host-specific. Wider vectors do not guarantee a gain on every processor:
AVX frequency policies, workload width, and noise overhead still matter.
The [dated comparison](../PERFORMANCE.md#symft_26_10_08-single-core-update)
separates incremental speedup from the total gain over old SymFT.

### Reusable workspace

[`prepared_sampler.cpp`](../../cpp/src/sampler/prepared_sampler.cpp) owns the
compiled plan and reusable buffers. Preparation and sampling are separate.
The default existing CPU executor and its multithreading behavior remain
unchanged.

## Randomness contract: `cpu-shot-v1`

External noise retains the original chunk-seeded generator. Measurement
randomness is a SplitMix64 stream derived from `(stream_id, global_shot)`.
Each original random site occupies its draw position, including deterministic
active measurements; removing an unused dormant draw advances its counter.
Early termination of one shot therefore does not shift another shot's
measurement randomness.

- Repeating the same stream, circuit, backend, and chunk configuration
  reproduces the same run within the supported build/version contract.
- Real/complex and detector-hoisting variants can be compared on the same tape.
- The same identifier does **not** promise legacy-CPU or GPU-identical counts.
- Changing `sample_chunk_shots` can change noise tapes. Cross-version plans and
  floating-point platforms are not promised to be bitwise stable.
- Warmups and same-stream replays must not be added as fresh independent shots.

## Build and benchmark

From the repository root, on the tested Linux host:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=OFF -DSYMFT_CPP_NATIVE=ON \
  -DSYMFT_CPU_DIAGNOSTICS=OFF
cmake --build build-cpu -j 8
ctest --test-dir build-cpu --output-on-failure

taskset -c 5 build-cpu/cpp/symft_rate_bench \
  --circuit benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim \
  --shots 1500000 --sampler batch --threads 1 \
  --postselect-detectors --cpu-backend compiled --stream-id 901
```

Choose an allowed idle logical CPU on your machine. LTO is optional: use a
different build directory and add `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`.
It changes code generation, not precision. Rebuild and test every configuration;
never overwrite a frozen benchmark baseline.

The CLI reports actual backend, RNG, gauge selection, entrance detector/check
counts, fallback reason, preparation wall time, and sampling time.
Sampling includes noise, expressions, state work, postselection, and counts.

## Ablations and diagnostics

- `--cpu-backend legacy`: pre-existing v2 executor.
- `--cpu-complex`: disable the real gauge; Python `cpu_real_gauge=False`.
- `--cpu-no-hoist`: preserve detector positions; Python `cpu_hoist_detectors=False`.
- Build separately with `-DSYMFT_CPU_DIAGNOSTICS=ON`, then use `--cpu-profile`
  for instrumentation. Never report this instrumented rate as production speed.

Diagnostic fields accumulate across repeats and are not a fully additive time
decomposition. Measurement time includes probability/projection; some affine
noise work is included in execution rather than presampling. Use total sample
time for end-to-end claims.

## Evidence and remaining work

The September experiment measured about 1.20x for d=3 and 2.41x for
d=5 with native+LTO versus the paired existing CPU path. The October release
has a separate, freshly measured comparison linked above. The earlier d=3
1.5–3x target was not reached; it must not be presented as an achieved result.
The 50B socket test used 40 independent one-worker processes on **20 physical
cores / 40 logical CPUs**, not a native 40-thread compiled sampler.

See [performance and intervals](../PERFORMANCE.md),
[correctness evidence](../VALIDATION.md),
[socket reproduction](../../benchmark/validation/README.md), and
[remaining experiments](ROADMAP.md).
