# SymFT benchmarks

This directory contains the circuit inputs and compact Python benchmark
harnesses used for the paper's near-Clifford and pure-Clifford results.
[`circuit/manifest.json`](circuit/manifest.json) records metadata for the
original generated circuit corpus; the executable CPU and GPU workload lists
are defined by [`config.json`](config.json) and
[`GPU_config.json`](GPU_config.json), respectively.

The circuit files are checked-in benchmark inputs. This directory does not
provide a circuit-regeneration script.

## Files

| Path | Purpose |
| --- | --- |
| [`circuit/`](circuit/) | Versioned Stim and extended Stim-like circuit inputs |
| [`config.json`](config.json) | CPU cases, per-call shot counts, logical CPU, duration, repeats, seed, and output path |
| [`benchmark.py`](benchmark.py) | Sequential single-logical-CPU benchmark for Stim, Clifft, Tsim, and SymFT |
| [`GPU_config.json`](GPU_config.json) | GPU cases, per-call shot counts, GPU selection, and per-circuit SymFT CUDA tuning |
| [`GPU_benchmark.py`](GPU_benchmark.py) | Sequential GPU benchmark for Tsim and SymFT |
| [`LICENSE-Clifft-paper`](LICENSE-Clifft-paper) | License for the vendored Clifft-derived inputs |

## Important: two circuit dialects

The `.stim` suffix describes the text format, not universal compatibility with
Stim.

- `pure_surface_*` are genuine Stim circuits. They contain only Clifford
  operations, Pauli noise, measurements, detectors, and observables.  Use these
  for SymFT/Stim/Clifft/Tsim comparisons.
- `msc_*`, `MSC_circuit_*`, `coherent_surface_*`, and `distillation.stim` use
  the extended SymFT/Clifft/Tsim Stim-like dialect. They contain operations such
  as `T`, `T_DAG`, `R_X`, or `R_Z`, so **Stim cannot parse or sample them**.
  Use these for near-Clifford simulators only.

SymFT and Clifft specify rotation angles in half-turns.  Therefore
`R_Z(0.02)` means an angle of `0.02*pi` radians.

## Circuit matrix

| Family | Configurations | CPU tools | GPU tools |
| --- | --- | --- | --- |
| Magic-state cultivation | `d=3`, `d=5` | Tsim, Clifft, SymFT | Tsim, SymFT |
| MSC distance-7 workload | `d=7` | Clifft, SymFT | Not configured |
| Coherent surface code | `(d,r)=(3,1),(3,3),(5,1),(5,5)` | Tsim, Clifft, SymFT | Tsim, SymFT |
| Magic-state distillation | 85-qubit `[17,1,5]` workload | Tsim, Clifft, SymFT | Tsim, SymFT |
| Pure-Clifford surface code | `(d,r)=(7,7),(9,9)` | Stim, Clifft, Tsim, SymFT | Tsim, SymFT |

The same circuit file is used for CPU and GPU runs. Precision is a backend
setting, not a circuit property. The current harnesses use FP64/JAX x64 for
the near-Clifford comparisons.

## Magic-state-cultivation provenance

`msc_d3_inject_cultivate_p1e-3.stim` and
`msc_d5_inject_cultivate_p1e-3.stim` are byte-for-byte copies of Clifft's
published `cultivation_d3.stim` and `cultivation_d5.stim` at commit
`db7dc9f13a2c2854690e92390c779048a1ac1400`. Their SHA-256 hashes are recorded
in [`circuit/manifest.json`](circuit/manifest.json). Clifft's construction is
based on Gidney, Shutty, and Jones'
magic-state-cultivation code at commit
`871e68ff6df2f75190b1bfd6351459d1b5a037e3`.

### Why the distance-7 file is a proxy

Gidney's released generator has specialized constructions only for `d=3` and
`d=5`.  In particular, the `d=5` construction includes hand-written
`d=3 -> d=5` growth, double-cat verification, feed-forward corrections, and
detector-healing rules.  There is no parameter-only route to a correct `d=7`
cultivation circuit.

`msc_proxy_d7_unverified_p1e-3.stim` is therefore deliberately named as a
proxy.  It was constructed as follows:

1. Start from the canonical `d=5` unitary inject+cultivate construction.
2. Re-express the logical-Y flow for Gidney's generic growth primitive.
3. Apply generic color-code growth from `d=5` to `d=7`.
4. Apply five distance-7 superdense syndrome cycles.
5. Apply the same S-to-T conversion/noise conventions used by Clifft.

This gives a 76-qubit, 310-measurement, 299-detector workload with an MSC-like
active-space profile.  It does **not** contain a newly derived and validated
distance-7 double-cat gadget.  Do not use its detector statistics, acceptance
rate, or logical observable for correctness or logical-error claims.  It is
only suitable for the explicitly qualified performance experiment described
in the paper draft.

## Coherent-noise construction

Each coherent-noise file begins with Stim's
`surface_code:rotated_memory_z` circuit using the same four noise settings as
the pure-Clifford files:

```text
after_clifford_depolarization = 1e-3
after_reset_flip_probability = 1e-3
before_measure_flip_probability = 1e-3
before_round_data_depolarization = 1e-3
```

Following Clifft's benchmark, every generated `DEPOLARIZE1(...)` and
`DEPOLARIZE2(...)` instruction is replaced by `R_Z(0.02)` on the same targets.
Reset and measurement flip errors remain stochastic.  The `d=5,r=5` case is
already extremely expensive for exact near-Clifford simulation, so no
coherent `d=7` case is included.

## Environment

Use a Python environment in which `stim`, `clifft`, `tsim`, and `symft` can all
be imported. Activate that environment before running the commands below. The
CPU harness forces the relevant thread-pool environment variables to one before
importing the simulators. The GPU harness additionally requires CUDA-enabled
JAX for Tsim and a CUDA-enabled SymFT Python build.

Both scripts use only relative default paths: a relative `--config` path is
resolved from the directory containing the script, and `circuit_dir` and
`report` are then resolved from the directory containing that configuration
file.

## CPU benchmark

[`benchmark.py`](benchmark.py) runs every selected case sequentially and pins
the process to one logical CPU. It also sets Clifft and SymFT to one thread and
checks SymFT's active thread count.

From the repository root:

```bash
python benchmark/benchmark.py --list
python benchmark/benchmark.py --cpu 42
```

Run one case or override the configured duration and repeat count:

```bash
python benchmark/benchmark.py \
  --cpu 42 \
  --only '^coherent_d5_r1__symft$' \
  --seconds 60 \
  --repeats 2
```

The `--only` value is a regular expression matched against
`<circuit-id>__<tool>`. Command-line values override the corresponding entries
under `run` in [`config.json`](config.json). Results are written to the
configured `report` path, which is `performance.md` by default.

## GPU benchmark

[`GPU_benchmark.py`](GPU_benchmark.py) benchmarks Tsim and SymFT only; SOFT is
intentionally not included. Each case runs in a fresh child process so its GPU
memory is released before the next case. The selected physical device is
exposed through `CUDA_VISIBLE_DEVICES`.

From the repository root:

```bash
python benchmark/GPU_benchmark.py --list
python benchmark/GPU_benchmark.py --gpu 0
```

For a single SymFT GPU case:

```bash
python benchmark/GPU_benchmark.py \
  --gpu 0 \
  --only '^coherent_d5_r1__symft$' \
  --seconds 60 \
  --repeats 2
```

[`GPU_config.json`](GPU_config.json) also supplies SymFT's CUDA mode,
`threads_per_block`, and `shots_per_launch` for each circuit. Results are
written incrementally to `GPU_performance.md` by default.

## Shot counts and timing

The shot-count sections specify the number of shots passed to one public Python
API call:

- `stim_batch` for Stim;
- `tsim_call` for Tsim;
- `clifft_call` for Clifft;
- `symft_call` for SymFT.

For each repeat, the harness keeps making calls of that fixed size until the
sum of measured sampling time reaches `sample_seconds`. Consequently, a repeat
can exceed the target duration by the time of its final API call. Compilation,
sampler construction, and warmup are performed before the measured loop and
reported separately. Throughput is attempted shots divided by measured sample
time; the final value is the arithmetic mean across repeats.

## SymFT planning check

To verify SymFT parsing and planning without allocating an active-state vector
or sampling any shots, build and run the dedicated planner tool:

```bash
cmake -S . -B build -DSYMFT_CPP_ENABLE_CUDA=OFF
cmake --build build --target symft_plan
build/cpp/symft_plan benchmark/circuit/coherent_surface_d5_r5_p1e-3_rz0p02.stim
```

`symft_plan` reports parse time, planning time, maximum active width, pending
operation counts before and after optimization, rotation fusions, measurement
left-swaps, and peak resident memory. Use this tool—not `symft_bench` with zero
shots—for large-width preprocessing checks, because benchmark executors
allocate active state storage independently of the requested shot count.

## Benchmarking notes

- Keep compilation/preprocessing time separate from steady-state sampling
  throughput, and compile once before repeated sampling.
- Use identical detector postselection and output contracts across tools.
- Report attempted shots/s and, when postselecting, survivor rate separately.
- Record simulator commit, compiler flags, thread affinity, CPU/GPU model, and
  FP32/FP64 mode with every result.
- Treat any proxy or otherwise unverified workload label and correctness
  limitation as part of every table or figure caption that includes it.

## References and license

- [Clifft paper benchmark circuits](https://github.com/unitaryfoundation/clifft-paper/tree/main/qec_bench/circuits)
- [Magic state cultivation](https://github.com/Strilanc/magic-state-cultivation)
- [Stim circuit generation API](https://github.com/quantumlib/Stim)

The two canonical MSC inputs and the construction used for the proxy derive
from the Apache-2.0-licensed Clifft paper repository.  A copy of that license is
provided in [`LICENSE-Clifft-paper`](LICENSE-Clifft-paper).
