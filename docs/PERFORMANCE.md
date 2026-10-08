# Measured performance and statistical validation

The **`symft_26_10_08` single-core comparison was measured on 2026-10-08**.
The GPU and 50B socket results remain archived September measurements, not
measurements of the main-based integration. Fresh integration checks are
separately recorded in [MAIN_INTEGRATION.md](MAIN_INTEGRATION.md). They are not
new 50B runs of this release. Fresh CPU evidence is in
[`cpu_20261008.json`](../benchmark/results/cpu_20261008.json).
The September counts, per-repeat CPU timings, confidence intervals,
configurations, and source hashes are in
[`optimization_202609.json`](../benchmark/results/optimization_202609.json).
The review checkout is tested separately in [VALIDATION.md](VALIDATION.md).

## Definitions and workload

- Throughput means **attempted shots/s**, not accepted shots/s.
- Discard rate is `discarded / attempted`.
- Error rate after postselection is `logical_errors / accepted`.
- `accepted + discarded == attempted` and `logical_errors <= accepted`.
- Measurements use FP64, physical noise parameter `p=0.001`, observable 0,
  and postselection on all detectors.
- The canonical MSC inputs include **injection and cultivation, not escape**.
  They are not the separately labeled unverified d=7 proxy.
- Warmup, same-stream replays, and previous runs are not counted again as fresh
  independent samples. CPU and GPU results are reported separately, not pooled.

The d=5 circuit SHA-256 is
`c2b4566917bd9bf27a5705284dac02700ef0dcc7c03c91066670db376d633a6d`;
the d=3 SHA-256 is
`90a7d841e003e5ee38137cd9a3eb6529bb552e49c424bc6b0932a27d97cdb41f`.
See [circuit provenance](../benchmark/README.md#magic-state-cultivation-provenance).

## symft_26_10_08 single-core update

Intel Xeon Gold 5218R, logical CPU 5, one worker, GCC 13.3, FP64,
Release/native+LTO, diagnostics off. Three versions were interleaved, rotating
their order over seven repeats; each had a separate warmup. Each timed repeat
used 12M d=3 or 1.5M d=5 attempts. This is the same circuit, precision,
postselection, thread count, and build policy for all three versions.

| CPU version | d=3 median, M/s | d=3 min–max, M/s | d=5 median, k/s | d=5 min–max, k/s |
| --- | ---: | ---: | ---: | ---: |
| Old SymFT, unmodified `e86c6a9` | 1.787010 | 1.769120–1.805840 | 114.864 | 112.760–115.547 |
| September compiled implementation, remeasured | 2.089110 | 2.056860–2.116760 | 270.647 | 262.030–274.955 |
| **`symft_26_10_08`, compiled + AVX-512** | **2.130850** | 2.074960–2.142330 | **348.896** | 344.450–351.959 |

Ratios of medians against old SymFT are **1.192x (d=3)** and **3.037x (d=5)**.
Against the already optimized September implementation, the incremental gains
are **2.0%** and **28.9%**, respectively. The d=3 increment is small relative
to host variability; do not describe it as a large new improvement. These are
new ratios, not products of speedups measured on different dates.

The old baseline was rebuilt directly from commit
`e86c6a92525650744ce0dc2a65126e1ab951882d`; it is old **SymFT / SOFT v2**, not
SOFT v1. Its original CLI has no stream-ID option, so its timing repeats replay
stream 0. The two compiled variants use streams 261008100–261008106 and have
identical counts on every paired repeat. Replays are not independent samples.

Sampling timers include noise, affine expressions, state evolution,
postselection, and count aggregation; parsing/planning are excluded. The host
was not reserved exclusively, and unrelated work plus CUDA review tasks on
the other socket could run concurrently. Pinned-core and SMT-sibling CPU-time
deltas are included in the evidence. Min–max ranges are descriptive, not
confidence intervals or cross-machine guarantees.

The fresh compiled d=3 runs observed 43 accepted errors in 57,690,450 accepted
shots; d=5 observed zero in 1,513,253 accepted shots. Those d=5 counts are far
too small to estimate a rate near 1e-9, and zero errors do not establish zero
error probability. The separate archived 50B runs below remain the rare-event
evidence. Two historical nonzero-error d=5 segments were also replayed with
the new AVX-512 worker: each reproduced its original single logical error,
accepted count, and discard count. Those 10M replays are not new production data.

The new CPU code only changes exact real-state rotation/promotion kernels;
it does not change FP64 precision, Born probability reductions, noise, detector
conditions, or RNG. See [implementation](optimization/CPU.md#symft_26_10_08-wider-real-state-kernels),
[test scope](VALIDATION.md), and the
[three-way reproduction recipe](../benchmark/validation/README.md#dated-single-core-comparison).

## September single-core CPU: paired experiment

Intel Xeon Gold 5218R, logical CPU 5, one worker, GCC 13.3, Release/native,
diagnostics off. Seven interleaved repeats used streams 901–907; each repeat
used 12 million d=3 or 1.5 million d=5 attempts. Each variant had separate
warmup. The host was not exclusively reserved; sibling-core contention was
recorded. Rates exclude preparation and use the native sampling timer.

| SOFT v2 CPU executor | d=3 median, M/s | d=3 min–max, M/s | d=5 median, k/s | d=5 min–max, k/s |
| --- | ---: | ---: | ---: | ---: |
| Existing executor (`legacy`) | 1.656350 | 1.629000–1.666160 | 105.713 | 102.394–111.968 |
| Compiled, native | 1.846280 | 1.822820–1.856580 | 247.562 | 234.683–256.522 |
| Compiled, native + LTO | 1.981020 | 1.899340–2.027420 | 254.392 | 234.863–264.391 |

Ratios of medians versus the paired existing executor: **1.115x / 2.342x**
for d=3 / d=5 without LTO, and **1.196x / 2.406x** with LTO. This is not a
comparison against SOFT v1 or against the older cross-tool README table.
The d=3 improvement did not reach the original 1.5–3x aspiration.

Preparation medians (existing / compiled / compiled+LTO) were
29.36 / 28.65 / 27.34 ms for d=3 and 937.78 / 940.44 / 904.00 ms for d=5.
Small preparation/LTO differences should not be treated as guaranteed gains.
Portable non-native builds can use scalar real-vector kernels and need their
own performance measurement.

Ordinary compiled and compiled+LTO repeats had identical counts; the latter
are replays, not additional independent shots. The seven ordinary d=5 runs
observed zero errors in 1,512,458 accepted shots: this does **not** establish
a zero physical error rate or estimate a rate near 1e-9 precisely.

## GPU performance

RTX 4090, FP64, GPU-presampled expressions, JIT, packed/scalar sparse noise,
symbolic scheduling, and entrance row reduction. The measured configuration
uses 524,288 shots/launch, register cap 128, and 128 JIT threads/block.
[Exact activation recipe](optimization/CUDA.md).

| Workload | Attempts | Sampling time | Attempted shots/s |
| --- | ---: | ---: | ---: |
| d=3 | 1,000,000,000 | 6.79237 s (printed) | 147.224 M/s (printed) |
| d=5 | 50,000,000,000 | 2,021.420764 s | 24.735078 M/s |

The d=3 row is one short run, not equivalent to the d=5 long stability test.
Its printed throughput and time are rounded independently. d=5 rates include
noise generation, state evolution, detector/observable work, device reduction,
synchronization, and count transfer. One-time parsing/planning/JIT and
between-call journaling are excluded from the summed sampling-call timer.

The previous v2 CUDA executor was remeasured on the same d=5 fixture and
launch size in three 50M timing replays: median 2.619360 M/s. Comparing that
median with the optimized 50B sampling rate gives approximately **9.44x**;
the different run durations are disclosed, and this is not a new controlled
cross-tool comparison. The prior 20B GPU run is a separate experiment, not
part of either 50B total.

## One CPU socket: 50B attempts

One Xeon Gold 5218R socket has **20 physical cores / 40 logical CPUs**.
The run used 40 persistent, independently pinned compiled/LTO processes on
logical CPUs `0–19,40–59`, not one compiled sampler with `threads=40`.
It completed 10,000 unique 5M-shot segments in one session without a worker
restart. Each worker prepared its sampler and performed a separate 500k-shot
warmup before the production barrier.

- Common sampling wall span: **16,897.163955 s**, approximately 4 h 41 min 37 s.
- Throughput: **2,959,076.454 attempted shots/s**.
- Total invocation wall time including startup/preparation/warmup: 16,905.602122 s.
- Interior 600-second window min/median/max: **2.958311 / 2.960421 / 2.976156 M/s**.
- Steady worker CPU time: approximately **39.984 / 40 logical-CPU equivalents**
  (99.96%); this is worker CPU-time utilization, not a hardware instruction
  throughput or physical-core efficiency claim.

The socket timer includes dispatch gaps and the final drain. Window boundaries
linearly allocate partially overlapping segments; these window rates are
estimates, while the overall common-wall-span rate uses the exact total/span.
Background contention is included. This result does not imply linear scaling
from a single core to 40 logical CPUs.

## Discard and postselected error statistics

| Run | Attempted | Discarded | Accepted | Accepted logical errors |
| --- | ---: | ---: | ---: | ---: |
| CPU d=5 | 50,000,000,000 | 42,802,205,779 | 7,197,794,221 | 25 |
| GPU d=5 | 50,000,000,000 | 42,802,148,564 | 7,197,851,436 | 28 |
| GPU d=3 | 1,000,000,000 | 313,194,348 | 686,805,652 | 616 |

| Run | Discard rate | Error / accepted | 95% Clopper–Pearson interval for error / accepted |
| --- | ---: | ---: | --- |
| CPU d=5 | 85.604411558% | 3.473286e-9 | [2.247728e-9, 5.127256e-9] |
| GPU d=5 | 85.604297128% | 3.890050e-9 | [2.584910e-9, 5.622205e-9] |
| GPU d=3 | 31.319434800% | 8.969058e-7 | [8.274669e-7, 9.706161e-7] |

Discard 95% Wilson intervals, in probability units, are respectively
[0.8560410386, 0.8560471925], [0.8560398943, 0.8560460483], and
[0.3131656031, 0.3132230944]. All displayed rates are derived from integer
counters; their decimal precision does not imply equally small uncertainty.

The matched CPU/GPU d=5 comparison gives two-sided discard p=0.6063 and
accepted-error Fisher p=0.7838. This supports statistical compatibility within
the reviewed scope; **non-rejection is not proof of identical distributions**.
Only 25/28 errors remain a broad rare-event estimate. Independent trajectory,
kernel, and noise tests are separate correctness evidence.

Two CPU segments containing errors (683 and 1377) were replayed with complex
FP64 and detector hoisting disabled; both matched their original counts.
Those 10M replayed shots are excluded from the production 50B total.
Historical CPU 30B logs did not freeze their original binary/input and are not
the primary comparator; they cannot override the matched-fixture evidence.

## Provenance and reproduction

The [October CPU record](../benchmark/results/cpu_20261008.json) contains all
42 timed rows, binary identities, measured source hashes, and per-core timing
telemetry. Its raw archive SHA-256 links it to the frozen local experiment.
Check that record with `python3 benchmark/validation/check_cpu_comparison.py`;
add `--check-current-sources` when verifying this same release's measured code.

The [curated JSON](../benchmark/results/optimization_202609.json) preserves
counts, relevant timings, intervals, 42 non-warmup CPU repeat records, build
hashes, circuit hashes, and hashes of the original analysis/metadata files.
The original archive paths are recorded relative to the experiment workspace;
they are provenance identifiers, not files claimed to be bundled in this repo.
Full journals, frozen executables, caches, and raw telemetry remain in that
workspace. Source hashes alone do not replace access to those raw records.

Run these repository-local checks:

```bash
python3 tools/check_documentation.py
python3 benchmark/validation/check_results.py
```

When the original archive is available, independently check the exported
values, raw journal totals, unique segment/stream IDs, circuit hashes, and
source-file hashes without rewriting the archive:

```bash
python3 benchmark/validation/check_results.py --experiment-root /path/to/SymFT_CUDA_Astra
```

For new sampling runs, use the [CPU](optimization/CPU.md),
[CUDA](optimization/CUDA.md), and [long-run recipes](../benchmark/validation/README.md).
Record a new revision/build identity and new timing evidence; do not relabel
archived results as a fresh measurement of a different build or machine.
