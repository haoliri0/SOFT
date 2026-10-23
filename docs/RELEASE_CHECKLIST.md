# Release checklist

## Version and source identity

GitHub **`v0.1.0`** maps to Python **`2026.10.8`** and update
**`symft_26_10_08`**. [release.json](../release.json) is the publishing mapping;
[the release notes](releases/v0.1.0.md) explain why these versions differ.
Python `0.1.0` and `0.1.1` already identify older published packages.

The baseline is main `3f718e9`, including PR #19's integration of the October 8
update. Use an isolated release branch/worktree. Later local development, such
as measurement-major output, must not enter this release accidentally.
Historical SOFT v1 tags and `legacy/softv1/` are preserved.

## Before tagging

- Commit every input referenced by documentation, including the October 9
  distillation benchmark JSON; check from a clean checkout.
- Keep the Python source version, package metadata, release mapping and notes
  consistent. Run `python tools/check_release.py` and
  `python tools/check_documentation.py`.
- Test portable SIMD and scalar C++ configurations, Python APIs and the
  persistent-worker/statistics helpers. Run CUDA FP32/FP64 and JIT tests on a
  real device and record the toolchain and source identity separately.
- Build and install the sdist in isolation. Test all supported wheel targets;
  public wheels use `SYMFT_PY_NATIVE=0` and `SYMFT_PY_ENABLE_CUDA=0`.
- Release-branch PRs and manual release-workflow runs build the full 20-wheel
  CPython 3.9–3.13 matrix without publishing. Other PRs build CPython 3.13.
- Check source/package licenses and the final notes. Keep historical benchmark
  claims separate from fresh validation; a new 50B run is not required.
- Merge the verified release preparation and confirm the exact final SHA.

## Publication

Pushing the tag recorded in `release.json` invokes the release workflow:

1. Validate the tag/package mapping and run the reusable CPU/docs CI.
2. Build and test wheels and the sdist, validate the full artifact matrix,
   metadata and embedded versions, and generate the manifest and checksums.
3. Create a GitHub draft and attach the verified assets.
4. Upload the same distribution files to PyPI through the existing trusted
   publisher (`release.yml`, environment `pypi`).
5. Publish the GitHub draft as the latest release after the PyPI job succeeds.

Manual workflow runs never upload to PyPI or create a GitHub release. Tag
publication is intentionally separate from a build-only rehearsal. Do not
retarget an existing release tag or overwrite published distribution files.
If publication partially fails, inspect which files are already public before
resuming; do not rebuild different bytes under the same version.

## After publication

Verify the public tag SHA, all GitHub assets and their hashes, PyPI filenames
and hashes, and a fresh `pip install symft==2026.10.8` outside the source tree.
Check installed package metadata against `symft.__version__` and
`symft.__release__`, then execute a counts-sampling smoke test.
