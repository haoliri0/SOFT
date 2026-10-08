# Maintainer review and manual commit/push

## Prepared scope

This checkout prepares the existing remote **`symft`** branch for review.
SOFT remains the repository/project name; SymFT names the v2 architecture.
Release name: **`symft_26_10_08`**; Python distribution version: **`2026.10.8`**.
Imports, C++ names, and default backends are not renamed or switched. No commit,
push, tag, GitHub settings change, or package publication is part of preparation.

Local branch: `symft`. Remote: `https://github.com/haoliri0/SOFT.git`.
Recorded branch base: `e86c6a92525650744ce0dc2a65126e1ab951882d`.
Optimizations came from the existing experiment checkout based on
`9ec5790322f93140e78bdb6d6620a2a43eceba0b`. Their source trees differ at baseline
only in two benchmark configuration/harness files; the branch's versions of
those files are preserved. This is not a merge of the separately evolving
`main` branch, and does not contain every later `main` API feature.

**Remote freshness is unverified.** HTTPS connections failed with TLS errors;
an SSH read-only attempt also failed. The branch base was obtained from an
existing local clone's `origin/symft`, not a successful fresh fetch. Do not
interpret the prepared checkout as proof that the remote still has that tip.

## Review the working tree

The prepared integration is staged for a single maintainer-reviewed commit.
Review both the index and any later edits; the original experiment checkout
and its pre-existing staged/unstaged changes remain separate and preserved.

```bash
git branch --show-current
git remote -v
git status --short
git diff --check
git diff --cached --check
git diff --cached --stat
git diff --cached
git ls-files --others --exclude-standard
python3 tools/check_documentation.py
```

Plain `git diff` omits staged and untracked content. Inspect the index and any
listed new files too. Build directories, native libraries,
JIT caches, raw journals, and local review artifacts must not enter the commit.

Read [validation scope](VALIDATION.md) and [performance caveats](PERFORMANCE.md).
Review CPU/GPU implementation changes separately from naming/documentation.
Logical commits can split shared build/bindings, CPU, CUDA, and documentation,
but both backends touch shared files: ensure each chosen split still builds.
A single reviewed integration commit is also possible.

## Check the actual remote before committing

These commands only fetch/read; they do not publish your changes:

```bash
git fetch origin refs/heads/symft:refs/remotes/origin/symft
git rev-parse origin/symft
git rev-parse HEAD
```

For the prepared state, both hashes should be the recorded base above.
If the remote moved, stop and reconcile those changes while preserving this
working tree. Do not reset it, force-push it, or assume an unrelated `main`
history is the correct rebase target. If a fetch fails, remote freshness is
still unknown.

## Commit and push yourself after review

Only after confirming the file list and remote base:

```bash
git add README.md CHANGELOG.md .gitignore .github docs tools benchmark cpp python
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Prepare symft_26_10_08 CPU/CUDA performance release"
git push -u origin HEAD:refs/heads/symft
```

The explicit destination prevents an accidental push to `main`. There is no
`--force`; a concurrent remote update should cause a normal non-fast-forward
rejection, at which point fetch/reconcile rather than overriding it.
If commit succeeds but the network prevents push, retain that local commit and
retry only the push after resolving connectivity and fetching the remote.

## Separate release decisions

- Updating the GitHub About text, publishing the prepared wheel,
  creating a release/tag, or integrating this branch into `main` are separate
  maintainer actions, not performed by this preparation.
- Proposed About: “SOFT: a high-performance simulator for fault-tolerant quantum
  circuits. SOFT v2 is powered by the SymFT sampling architecture.”
- Keep SOFT v1 research attribution and reproduction revisions distinct from
  the SymFT architecture and its optimization measurements.
- This branch's CI checks Linux CPU builds, Python behavior, documentation,
  and evidence arithmetic. It does not certify CUDA hardware or non-Linux
  wheel compatibility. Local CUDA checks are recorded separately.
