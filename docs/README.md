# SOFT documentation

SOFT is the project; SymFT is the second-generation sampling architecture.
The Python package is still `symft`. GitHub Release **`v0.1.0`** contains the
main-based **`symft_26_10_08`** update as Python package **`2026.10.8`**. See the
[release notes](releases/v0.1.0.md) for installation, platforms and limitations.

## Start here

- [Project overview, install, and quick start](../README.md)
- [Python API](../python/README.md)
- [Project naming, SOFT v1, and compatibility](PROJECT.md)
- [Changelog](../CHANGELOG.md)

## Implementation and measurement

- [Optimization overview](optimization/README.md)
- [Compiled CPU counts backend](optimization/CPU.md)
- [CUDA JIT counts backend](optimization/CUDA.md)
- [Implemented work and remaining experiments](optimization/ROADMAP.md)
- [Performance, exact counts, and uncertainty](PERFORMANCE.md)
- [Correctness tests and validation boundaries](VALIDATION.md)
- [Main-based integration and fresh smoke comparisons](MAIN_INTEGRATION.md)
- [Machine-readable main integration checks](main-integration-checks.json)
- [Circuit provenance and cross-tool harnesses](../benchmark/README.md)
- [Long-run reproduction tools](../benchmark/validation/README.md)
- [Machine-readable archived measurements](../benchmark/results/optimization_202609.json)
- [Fresh October CPU comparison](../benchmark/results/cpu_20261008.json)
- [Historical pre-optimization cross-tool tables](HISTORICAL_BENCHMARKS.md)

## Maintainers

- [Release checklist](RELEASE_CHECKLIST.md)
- [Documentation/evidence checks](../tools/check_documentation.py)

The original experiment workspace, frozen binaries, journals, failed attempts,
and detailed Chinese laboratory notes remain outside this Git checkout. Their
results are summarized here with source hashes, rather than checking in binary
artifacts, machine-specific build directories, caches, or massive logs. See
[provenance](PERFORMANCE.md#provenance-and-reproduction).
