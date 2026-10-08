# Historical cross-tool benchmark snapshot

This table is retained from the initial `symft` branch README. It is not a
measurement of the new compiled CPU or CUDA JIT paths. The original cross-tool
experiments were not rerun during this integration; do not combine their
baselines with the September optimization measurements to claim new speedups.
See [current optimization measurements](PERFORMANCE.md) for the separate,
configuration-specific results.

The following results are taken from the earlier
[benchmark suite](../benchmark/README.md). They report attempted shots per second
through the public sampling paths, not isolated kernel rates. CPU measurements
use one pinned core of an Intel Xeon Gold 5218R and complex FP64 arithmetic.
Each entry is the arithmetic mean of two sampling-only runs of approximately
60 seconds; compilation and planning are excluded.

For pure-Clifford circuits, the most relevant baseline is
[Stim](https://github.com/quantumlib/Stim). For magic-state cultivation (MSC),
the most relevant CPU baseline is
[Clifft](https://github.com/unitaryfoundation/clifft).

| Regime | Circuit | Baseline | SOFT v2 (SymFT, pre-optimization) | Speedup |
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

| Circuit | SOFT v1 FP64 | Tsim CUDA | SOFT v2 CUDA FP64 (pre-optimization) | v2 vs v1 |
| --- | ---: | ---: | ---: | ---: |
| MSC `d=3` | 331.50k | 26.93k | **68.04M** | **205×** |
| MSC `d=5` | 5.03k | DNC | **2.61M** | **518×** |

SOFT v1 and the SymFT-based SOFT v2 use FP64. Tsim's effective CUDA path retains FP32 and complex64
intermediates despite JAX x64 mode, so its result is not a precision-matched
comparison. DNC means compilation did not finish within 300 seconds.

The compared tools expose different output forms, so these numbers compare the
tested public sampling paths rather than identical-width output kernels. See
the [benchmark documentation](../benchmark/README.md) for the circuits,
configuration, hardware details, and measurement protocol.
