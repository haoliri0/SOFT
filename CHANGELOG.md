# Changelog

## v0.1.0 — 2026-10-24

First GitHub Release, containing the `symft_26_10_08` update.

Python package version: `2026.10.8`. Release identifier: `symft_26_10_08`.
SOFT remains the project name and `import symft` remains unchanged.

### Release packaging

- Publish portable CPU wheels for CPython 3.9–3.13 on Linux x86_64/aarch64,
  macOS arm64, and Windows amd64, plus a source distribution for CPU/CUDA builds.
- Disable host-native optimization in distributable wheels and test each wheel
  with the Python API suite; test installation from the source distribution.
- Gate publication on CPU tests, documentation checks, package identities and
  the complete wheel matrix. Manual workflow runs build without publishing.
- Attach distribution files, a source/asset manifest and SHA256 checksums.
- Include the previously omitted October 9 distillation benchmark evidence.

### Main-based integration — 2026-10-09

- Ported onto main `c89b985` on branch `symft-26-10-08`, without merging the
  unrelated old symft history or dropping main-only functionality.
- Added compiled CPU reference normalization and explicit expectation-probe
  fallback; retained CUDA record/expectation output with JIT flags enabled.
- Fixed main's scalar CUDA record-copy path so each shot-owning thread writes
  its records; added regression coverage for chunk boundaries.
- Preserved the SOFT v1 archive, factorization code, and wheel release workflow.
- Added cross-feature regression tests and pytest discovery in CI.
  See [fresh integration checks](docs/MAIN_INTEGRATION.md).

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
- This release uses the main-based integration merged in PR #19.
  GitHub `v0.1.0` maps to Python `2026.10.8`; older Python `0.1.x` files and
  historical SOFT v1 tags retain their original identities.
