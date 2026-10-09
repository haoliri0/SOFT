# Main-based integration: symft-26-10-08

Prepared on 2026-10-09 from main commit
`c89b98514a919240b8afa53a271e08d926d3c987`, porting the optimization delta from
`symft` release `89c00eff971847f6aa8cbbbaf15592a724f5bd3d`. The branch shares
main's history; no unrelated-history merge, force push, or removal of main-only
features is used. Changes remain uncommitted for maintainer review.

## Compatibility work

- CPU compiled counts include expected detector bits before GF(2) elimination
  and detector hoisting, and normalize the selected logical observable.
  Regression coverage includes nonzero references, multiple packed words,
  contradictory constraints, noise, and quantum-dependent detectors.
- CPU expectation probes use the original executor with explicit fallback
  reason `expectation_values_present`; they must not be treated as destructive
  measurements by the counts-only optimizer.
- CUDA JIT is used only for compatible counts-only calls. `EXP_VAL` programs
  and full record/expectation output use main's general runtime, with unpacked
  expressions and without the counts-only early return.
- Fixed a pre-existing main CUDA record-copy issue: the scalar executor owns
  one shot per thread and must write every thread's records, while the persistent
  block executor writes through thread zero. Counts-only kernels are unaffected.
- Main's tableau/frame/pullback code, CUDA global-workspace fallback,
  `legacy/softv1/`, and `.github/workflows/release.yml` are retained.
- Python's reference keywords retain their existing positional order; the
  three opt-in CPU keywords are appended. CPU defaults and CUDA reference-count
  restrictions remain unchanged. Pytest discovers both function-style and
  unittest-style API tests in CI.

## Fresh validation

Completed on 2026-10-09 on Linux x86_64, Xeon Gold 5218R, RTX 4090,
GCC 13.3.0, CUDA 13.0.88, and Python 3.12.3. The
[machine-readable record](main-integration-checks.json) contains source and
artifact hashes, individual CPU timing runs, GPU counters, and test outcomes.
Historical checks and 50B measurements are not relabeled as fresh runs.

| Configuration | Result |
| --- | --- |
| Unmodified main, Release/native/LTO CPU C++ | 4/4 passed, before applying the port |
| Integrated Release/native/LTO CPU C++ | 5/5 passed |
| Integrated portable scalar CPU C++ | 5/5 passed, AVX2/AVX-512 disabled |
| Integrated native ASan/UBSan C++ | 5/5 passed, leak detection enabled |
| Integrated CUDA FP64 C++ | 9/9 passed, including actual device/JIT execution |
| Native CPU Python extension | 92 passed; 21 CUDA-only tests skipped; 4 subtests passed |
| Portable CUDA FP64 Python extension | 113 passed; 4 subtests passed |
| Portable CPU wheel built from the source distribution | 92 passed; 21 CUDA-only tests skipped; 4 subtests passed |
| Persistent worker, journal, and statistics helpers | 16/16 passed |

The independent CPU reference suite compares 4,092 trajectories and 102,702
intermediate states; maximum state/probability discrepancies are
`6.68e-15` / `1.11e-14`. The new reference-normalization tests include packed-word
boundaries and do not reuse the optimized detector-index calculation.
GPU noise expressions match the original generator bit-for-bit in a 100,003-shot
d=5 diagnostic, including a 34,467-shot tail. Separate CUDA regression tests
cover record copying, counts aggregation boundaries, and expectation probes.
Cold NVRTC tests passed during integration; the final CUDA suite reused the
JIT cache. No fresh CUDA memory-sanitizer result is claimed.

The source distribution was used to build a non-native CPU wheel, which was
installed into an isolated target and tested. Installed package metadata,
`__version__`, and `__release__` agree. No package was uploaded. Main's archive,
factorization sources, and release workflow are unchanged; there are no
unmerged index entries or deleted main files. Documentation links and archived
measurement arithmetic are checked with `tools/check_documentation.py`.

## Short performance checks against current main

These are **integration smoke measurements**, not replacements for the archived
long runs. All rows use canonical MSC injection+cultivation at `p=0.001`, FP64,
all-detector postselection, and attempted shots/s. Sampling timers exclude
preparation and JIT compilation. The host was not exclusively reserved.

### CPU, one pinned worker

Three interleaved repeats per variant, pinned to logical CPU 5, with warmups
excluded. Every build uses Release/native/LTO. Each repeat attempts 2M shots
for d=3 or 400k for d=5. Rates below are medians, with min/max in parentheses.

| Workload | Unmodified main `c89b985` | Previous optimized `symft` `89c00ef` | Main-based compiled CPU | Speedup vs main |
| --- | ---: | ---: | ---: | ---: |
| d=3 | 1.800 M/s (1.789–1.802) | 2.093 M/s (2.083–2.110) | 2.019 M/s (2.007–2.034) | 1.12x |
| d=5 | 121,328/s (121,299–122,489) | 346,918/s (345,722–350,859) | 360,407/s (350,149–360,536) | 2.97x |

Against the previous optimized branch, d=3 is 3.5% slower and d=5 is 3.9%
faster in this short comparison; this is not a claim that the port improves
every workload. The two compiled variants use matching streams
`261009100`–`261009102`: attempted/discarded/accepted/error counters match in
every pair, including eight total d=3 logical errors. Main's timing baseline
replays stream zero because its CLI lacks the new stream option; those repeats
are not independent rare-error statistics. Additional integrated **default**
CPU runs exactly reproduce main's stream-zero counters for both distances.

### GPU, d=5

The same integrated binary runs the existing GPU-presampled-expression path
and opt-in packed/scalar/symbolic JIT, each with three 10M-shot streams starting
at `261009200`, using 524,288 shots per launch. The CLI reports the average
sampling time per 10M-shot repeat; it does not expose per-repeat timing ranges.

| Path | Mean sampling time | Attempted throughput | Discarded / accepted / errors, across 30M shots |
| --- | ---: | ---: | --- |
| Existing CUDA path, optimizations disabled | 4.27946 s | 2.337 M/s | 25,679,956 / 4,320,044 / 0 |
| Opt-in CUDA JIT | 0.405311 s | 24.672 M/s | 25,679,957 / 4,320,043 / 0 |

This is a **10.56x** sampling-throughput ratio with a warm JIT cache, not a
cold-start or long-stability result. Aggregate discard counts differ by one
out of 30M shots; cross-backend seeded trajectories are not claimed to be
bitwise identical. Both discard fractions are approximately 85.60%.
Zero observed logical errors in these short tests does not establish zero
true postselection error probability. See the separate
[archived long-run evidence](PERFORMANCE.md).

## Reproduce the checks

Use the [C++ build/test commands](VALIDATION.md#checks-on-the-prepared-working-tree)
and [Python development instructions](../python/README.md#development-and-testing)
with this checkout. CUDA validation requires an actual compatible device;
CPU-only Python runs intentionally skip GPU tests. Python development headers
are required to compile the extension. From the repository root, the final
local test invocations were:

```bash
ctest --test-dir build-main-integration-cpu --output-on-failure
ctest --test-dir build-main-integration-scalar --output-on-failure
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 \
  ctest --test-dir build-main-integration-sanitize --output-on-failure
SYMFT_JIT_CACHE="$PWD/jit-cache" \
  ctest --test-dir build-main-integration-cuda --output-on-failure

PYTHONPATH=build-main-integration-python-cpu/lib \
  build-main-integration-venv/bin/python -m pytest python/tests -q
PYTHONPATH=build-main-integration-python-cuda/lib SYMFT_JIT_CACHE="$PWD/jit-cache" \
  build-main-integration-venv/bin/python -m pytest python/tests -q
PYTHONPATH=build-main-integration-packaging/installed \
  build-main-integration-venv/bin/python -m pytest python/tests -q
SYMFT_TEST_WORKER="$PWD/build-main-integration-cpu/cpp/symft_cpu_stream_worker" \
  build-main-integration-venv/bin/python -m unittest discover \
    -s benchmark/validation -p 'test_*.py' -v
python3 tools/check_documentation.py
git diff --check
```

These ignored build directories are local outputs, not included binaries.
The CPU smoke check uses
[`compare_cpu_versions.py`](../benchmark/validation/compare_cpu_versions.py)
with the three executables identified in the JSON record, `--cpu 5 --repeats 3
--d3-shots 2000000 --d5-shots 400000 --stream-start 261009100`. Raw local logs
and frozen CPU executables are retained under the ignored
`review-artifacts/main-integration-cpu/` directory. The CUDA check uses the
[documented optimization flags](optimization/CUDA.md#build-and-activate)
with `--shots 10000000 --repeats 3 --stream-id 261009200`; the baseline sets
JIT, packed/sparse/scalar noise, symbolic scheduling, and row reduction to zero.
Profiling and noise diagnostics are disabled for timing.

## Publication

Follow the [manual commit/push checklist](RELEASE_CHECKLIST.md). Publish branch
`symft-26-10-08`, then open a PR with base `main`. This preparation does not
commit, push, open a PR, create a tag, or publish a package.
