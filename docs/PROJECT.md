# Project naming and compatibility

## Names and versions

| Name | Meaning |
| --- | --- |
| SOFT | Long-lived software project and GitHub repository |
| SOFT v1 | Original independent per-shot simulator and the corresponding research work |
| SOFT v2 | Second-generation software implementation |
| SymFT | Symbolic/compiled sampling method and architecture used by SOFT v2 |
| `symft_26_10_08` | Dated CPU/CUDA performance release prepared on 2026-10-08 |
| `symft` | Existing Python distribution/import name |

Use **SOFT v2 (SymFT)** on first mention in implementation/benchmark descriptions,
and **SymFT** when discussing the method. Use **SOFT v1** for comparisons against
the original implementation; an unqualified “SOFT vs SymFT” comparison obscures
the distinction between the project and an implementation generation.

The `v2` generation label is not the Python distribution version. This update
is named **`symft_26_10_08`**; its packaging-compatible calendar version is
**`2026.10.8`**, exposed as `symft.__version__`. The exact release name is
available as `symft.__release__`. The name does not imply that a Git tag,
GitHub release, or PyPI upload has already been created.

## Existing users

- Keep using `pip install symft` / `import symft` for released packages. Build
  this checkout to test the unreleased optimizations.
- C++ namespace `symft`, target `symft_cpp`, executable names, environment
  variables, and exception `SymFTError` are unchanged.
- Existing CPU sampling remains the default. The new compiled CPU path and
  CUDA JIT require explicit opt-in.
- A matching seed across different backends is not a matching random tape.
  The compiled CPU RNG contract is documented in the [CPU guide](optimization/CPU.md).
- Branding does not promise source/API compatibility between the original
  `soft` Python interface and `symft`. Follow the [current API guide](../python/README.md)
  when migrating an old script; do not just replace its import statement.

This branch's `cpu_backend="legacy"` means the existing **SOFT v2 CPU executor**,
not the original SOFT v1 implementation.

## Original SOFT and research attribution

The original implementation is preserved under
[`legacy/softv1`](../legacy/softv1/), retained unchanged from main.
For old-paper reproduction, use the
original paper's specified revision and input files; the archive directory
alone does not establish that every historical experiment is reproduced.

Keep the original SOFT and new SymFT research contributions distinct. Cite the
work corresponding to the method actually used, and record the software
revision, backend, precision, circuit hash, and postselection settings. No new
paper title, DOI, release tag, or citation metadata is invented in this update.
Maintainers should add verified bibliographic details when releasing the work.

## Branch scope

This integration targets **`symft-26-10-08`**, based directly on main commit
`c89b98514a919240b8afa53a271e08d926d3c987`. It ports the optimization delta from
`symft` release commit `89c00eff971847f6aa8cbbbaf15592a724f5bd3d` without importing
that branch's unrelated root history. Main's reference normalization,
expectation-value sampling, factorization code, archive, and release workflow
are retained. See [integration details](MAIN_INTEGRATION.md) and the
[manual review/push procedure](RELEASE_CHECKLIST.md).
