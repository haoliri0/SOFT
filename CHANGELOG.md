# Changelog

## symft_26_10_08 — 2026-10-08 (prepared, not yet published)

Python package version: `2026.10.8`. Release identifier: `symft_26_10_08`.
SOFT remains the project name and `import symft` remains unchanged.

### Further CPU acceleration

- Added native AVX-512 FP64 real-state rotation and promotion kernels, while
  retaining AVX2/FMA and scalar fallbacks. No sampling-model or RNG change.
- Added 2,556 promotion gauge/sign/width cases, including signed zero and
  output-boundary guards, alongside the independent trajectory/kernel tests.
- Added a reproducible, pinned three-way benchmark: original SymFT commit,
  September optimized CPU, and this release. See the
  [measured comparison](docs/PERFORMANCE.md#symft_26_10_08-single-core-update).

### Added

- Opt-in compiled CPU counts plan: GF(2) classical dependency expansion,
  early detector checks, packed entrance filtering, and compact live branches.
- Structurally checked real-gauge FP64 execution with scalar, AVX2/FMA and AVX-512
  kernels; compiled complex fallback when the gauge is incompatible.
- Explicit `cpu-shot-v1` measurement RNG contract and CPU diagnostic switches.
- Opt-in CUDA NVRTC circuit specialization, packed/sparse noise expressions,
  symbolic detector scheduling, real-gauge execution, and device count reduction.
- CPU plan and CUDA JIT correctness tests, portable-source build checks,
  reproducible validation helpers, and compact archived performance evidence.

### Documentation

- SOFT remains the project/repository brand; SymFT names the v2 architecture.
- Documented actual activation flags, precision defaults, fallback conditions,
  and same-seed limitations. Existing public identifiers remain unchanged.
- Separated historical cross-tool comparisons, single-core A/B experiments,
  GPU measurements, and the 40-process CPU socket validation.
- Recorded the CPU and GPU 50B counts separately, including accepted-shot
  denominators and confidence intervals. These are archived runs, not reruns
  of a newly published release.

### Compatibility and limits

- The existing CPU backend remains the default; compiled CPU is counts-only,
  single-worker, dense, postselected, and limited to `max_k <= 10`.
- The 40-process benchmark is not native compiled `threads=40` support.
- CUDA JIT is opt-in for GPU-presampled-expression counts with `max_k <= 10`.
  CUDA FP64 must be selected at build time; its default remains FP32.
- This is an update to the existing `symft` branch, not a merge of `main`.
  Package metadata is updated, but no Git tag, commit, push, or publication
  is performed as part of preparing this review tree.
