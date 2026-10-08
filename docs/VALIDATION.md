# Correctness evidence and review validation

Two kinds of evidence are kept separate: archived optimization experiments and
checks rerun on the prepared `symft` working tree. Neither means every circuit,
platform, compiler, or device has been exhaustively verified.

## symft_26_10_08 release checks

Fresh review date: **2026-10-08**, on the same Linux/Xeon Gold 5218R/RTX 4090
host. The dated checklist and artifact identities are in
[release-26-10-08-checks.json](release-26-10-08-checks.json). The September
record below is preserved as history, not silently relabeled as a new test.

- Native AVX-512+LTO, AVX2-only real kernels, and fully scalar C++ builds pass
  their CPU suites. Native ASan/UBSan also passes, including the new kernels.
- Every CPU configuration checks 2,556 promotion gauge/sign/width cases,
  8,364 independent gauge-kernel cases, 3,702 reference trajectories, and
  101,922 intermediate states. Promotion tests include signed zero and
  output-boundary guards. The native maximum state/probability discrepancies
  remain 6.68e-15 / 1.11e-14 against the long-double complex reference.
- All seven paired before/after repeats for each distance have identical
  discard/accepted/error counters. This supplements, not replaces, reference
  trajectory testing.
- New AVX-512 replay of historical d=5 segments 683 and 1377 reproduces
  `(discarded, accepted, errors)` = `(4,279,804, 720,196, 1)` and
  `(4,279,348, 720,652, 1)`. Those 10M replays are not pooled into the 50B run.
- A newly built native CPU Python extension and a portable wheel built from
  the new source distribution each pass all 49 API tests. Installed metadata
  and the public `__version__` / `__release__` identifiers agree.
- Persistent worker/journal smoke tests, evidence arithmetic, version labels,
  and local documentation links are checked separately. CUDA-linked CPU and
  actual GPU checks are recorded in the dated checklist.

This release does **not** rerun the 50B socket/GPU experiment or extrapolate a
new socket rate from the AVX-512 single-core result. There is no fresh claim
about other CPUs, GPUs, operating systems, or cold-JIT performance.

## Archived optimization evidence

### CPU

- 8,364 real-gauge kernel cases covering widths 1–10, masks, signs, pivots,
  and both projection branches.
- 3,702 complete trajectories and 101,922 intermediate state comparisons using
  an independent long-double complex reference. The reference does not call
  the compiled GF(2) eliminator or real/SIMD probability kernels.
- Native/LTO maximum state and probability discrepancies of approximately
  6.68e-15 and 1.11e-14; global phase is accounted for.
- Scalar ASan/UBSan checks, noise probabilities 0/1/high, feedback, incompatible
  gauges, contradictory detectors, multiword branches, tail sizes, move/reuse,
  empty requests, and fallback behavior.
- Same-stream legacy replay checks and compiled real/complex/no-hoist ablations.
- The separate 50B socket run and two nonzero-error segment replays described
  in [PERFORMANCE.md](PERFORMANCE.md).

### CUDA

- Independent long-double complex reference comparisons for synthetic feedback,
  complex-gauge, detector, and full d=3/d=5 state-evolution cases.
- Packed and unpacked expression layouts, symbolic detector scheduling,
  contradictory/duplicate rows, multiple branch words, and forced complex JIT.
- Bitwise comparison of optimized external-noise expressions with the original
  implementation, plus boundary/high-noise checks.
- Historical CUDA memory/synchronization checks and separate 20B/50B runs.
  Historical sanitizer results are not labeled as reruns during this review.

Long-run statistical agreement complements these checks; it does not replace
an independent state/noise reference. Finite-precision exact simulation does
not imply bitwise agreement across all implementations.

## Checks on the prepared working tree

Review date: 2026-09-30. Host: Linux x86_64, Xeon Gold 5218R, GCC 13.3,
RTX 4090, CUDA toolkit 13.0. Local build directories are ignored by Git.
The September review status and command details are recorded in
[review-checks.json](review-checks.json).

The review includes native/LTO CPU and fully scalar non-native C++ builds,
CPU Python bindings with native optimization disabled, and a CUDA FP64 build
with actual device execution. Additional checks cover the symbolic/packed JIT,
source packaging, the persistent worker/journal helper, local documentation
links, and archived evidence arithmetic. Consult the status record rather than
assuming that a configured check necessarily passed.

Reproduce the C++ checks from the repository root:

```bash
cmake -S . -B build-check-cpu -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=OFF -DSYMFT_CPP_NATIVE=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build-check-cpu -j 8
ctest --test-dir build-check-cpu --output-on-failure

cmake -S . -B build-check-scalar -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=OFF -DSYMFT_CPP_NATIVE=OFF \
  -DSYMFT_CPP_ENABLE_AVX2=OFF -DSYMFT_CPP_ENABLE_AVX512=OFF
cmake --build build-check-scalar -j 8
ctest --test-dir build-check-scalar --output-on-failure

cmake -S . -B build-check-cuda -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=ON -DSYMFT_CPP_CUDA_REAL_DOUBLE=ON \
  -DSYMFT_CPP_CUDA_ARCH=89
cmake --build build-check-cuda -j 8
ctest --test-dir build-check-cuda --output-on-failure
```

Run CTest without externally forcing JIT/packed/symbolic flags: registered
test variants select their own layouts. CUDA JIT tests include NVRTC compilation
and can take several minutes even though each reference case has few shots.

For explicit runtime noise validation, use a short d=5 CUDA CLI run with the
[full optimization environment](optimization/CUDA.md) plus
`SYMFT_VALIDATE_NOISE=1`. Do not use this diagnostic throughput as a benchmark.

Python checks are in [the interface guide](../python/README.md#development-and-testing).
The [long-run helper](../benchmark/validation/README.md) has its own small
actual-worker smoke test; it does not automatically start a 50B run.

## Boundaries

- This review does not run a new 50B CPU/GPU experiment or recertify the archived
  timing on a different compiler, machine, or generated binary.
- Native compiled CPU multithreading, large active widths, all adaptive
  circuits, and full measurement-array acceleration are not claimed.
- macOS, Windows, ARM, other GPU architectures, and a published wheel matrix
  are not certified by these local Linux checks. Portable scalar on x86_64
  is not a substitute for those platform tests.
- Repository-local Markdown links are checked automatically; external URLs
  and a rendered GitHub preview have not been exhaustively verified.
- Remote freshness remains a separate connectivity issue, not a correctness
  result. Follow the [manual fetch/review/push procedure](RELEASE_CHECKLIST.md).
