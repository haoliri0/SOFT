# SymFT optimization architecture in SOFT v2

These optimizations target aggregate-count sampling, particularly the canonical
MSC d=3/d=5 injection+cultivation circuits. They do not replace every public
sampling API or claim equal speedups for every circuit.

## Shared ideas

1. **Compile classical dependencies.** Expand record assignments, feedback,
   detectors, and the selected observable into affine expressions over GF(2).
   Exogenous noise and genuinely stochastic measurement outcomes remain
   distinct leaves; Born probabilities are still evaluated from the state.
2. **Reject only when justified.** Check each detector once its last dependency
   is known. Noise-only detector checks can precede quantum-state evolution.
   Rejected attempts still contribute to the attempted/discarded counts; they
   are not replaced by additional accepted shots.
3. **Prove a smaller representation.** When all operations preserve a real
   gauge, represent the same complex state as a real vector with known phases.
   If the proof fails, use complex arithmetic instead. This is not amplitude
   truncation or a reduction in numeric precision.
4. **Keep only live execution state.** Eliminate unused record storage and
   classical branches while preserving the selected backend's random-stream
   contract and all state-affecting measurements.

The d=5 fixture has 107 detectors, of which 93 are noise-only; d=3 has 20,
of which 16 are noise-only. Row reduction preserves the simultaneous zero
constraint. In these fixtures it does not reduce those entrance row counts
further, so it must not be credited with an additional factor-of-two gain.

## Different implementation choices

| CPU compiled counts | CUDA JIT counts |
| --- | --- |
| Reusable C++ operation stream | Circuit-specific CUDA source compiled by NVRTC |
| One state per surviving shot | One warp per shot, register-oriented state |
| 64-shot entrance masks | Packed per-shot noise expressions |
| Real scalar, AVX2/FMA, or native AVX-512 kernels | Warp operations and device counter reduction |
| Versioned `cpu-shot-v1` measurement RNG | GPU-specific measurement/noise stream contract |

CPU details: [CPU.md](CPU.md). GPU details: [CUDA.md](CUDA.md).

## Why vector length is not end-to-end complexity

Although the maximum active vector contains 16 entries for d=3 and 1024 for
d=5, those sizes describe a state-update component, not all work per attempted
shot. A useful decomposition is noise/expressions + detector checks +
survivor-weighted state evolution + aggregation. Width changes during the
circuit, rejected shots stop at different stages, and GPU parallelism and
memory/register behavior also differ. Thus a 64x ratio of maximum vector
lengths does not imply a 64x ratio of total sampling time.

The measured d=3/d=5 GPU rates also use different run lengths, so their ratio
is descriptive rather than a controlled microbenchmark of vector operations.
See [measurement contracts](../PERFORMANCE.md).

## Precision and scope

The published optimization measurements use FP64. There is no fast-math,
FP32 substitution, small-amplitude cutoff, omitted noise, changed detector
selection, or survivor-only throughput denominator in these measurements.
Finite-precision results need not be bitwise identical across real/complex
representations or different hardware. Explicit opt-in, fallback paths,
independent reference tests, and [validation limits](../VALIDATION.md) remain
part of the feature contract.
