# Implemented work and remaining experiments

This is a record of implementation status, not a promised performance schedule.
The original detailed planning/laboratory notes remain in the experiment
workspace; this page distinguishes achieved results from hypotheses.

| Area | Implemented | Still exploratory or absent |
| --- | --- | --- |
| CPU measurement | Frozen baselines, phase diagnostics, interleaved A/B, reference tapes | Reliable hardware-counter study; complete byte-traffic accounting |
| CPU classical work | GF(2) expansion, ready detectors, entrance masks, live branches | Direct event-to-final-expression noise generation |
| CPU state | Exact real gauge, complex fallback, scalar + AVX2/FMA; October AVX-512 rotation/promotion | AVX-512 measurement kernels; broader ISA/hardware search |
| CPU planning | Reusable compact operation/branch plan | Small-width AOT/JIT, hot/cold instruction splitting |
| CPU parallelism | Existing executor threads; compiled independent processes | Native multi-worker compiled counts, cross-shot AoSoA SIMD |
| CPU build | Optional tested LTO configuration | PGO and state caching |
| GPU | Circuit JIT, packed/sparse noise, symbolic checks, real gauge, reduction | Wider active spaces, broader device/platform coverage, robust shared-cache concurrency |

## Lessons from the measured prototypes

- A scalar real/GF(2) prototype did not automatically improve throughput;
  smaller state storage alone does not remove scheduling and expression cost.
- The September d=5 experiment reached approximately 2.41x single-core speedup
  with native+LTO. The October AVX-512 update has its own
  [fresh comparison](../PERFORMANCE.md#symft_26_10_08-single-core-update).
  September d=3 reached approximately 1.20x, below the original
  1.5–3x aspiration. That aspiration is not a measured result.
- Disabling real gauge or detector hoisting reduced speed in the ablations;
  their individual ratios cannot be multiplied to invent a total speedup.
- The d=5 incremental LTO benefit was small relative to runtime variability;
  repeat it before relying on a precise extra percentage.

For further work, keep FP64/model/postselection semantics fixed, retain an
unmodified baseline, replay complete trajectories against an independent
reference, and measure repeated end-to-end attempted-shot throughput. Prefer
small-width scheduling or cross-shot SIMD experiments for d=3. Keep unsupported
paths explicit and preserve the existing defaults until evidence warrants a
separate policy change.
