# Maintainer review and manual commit/push

## Prepared scope

Local branch: **`symft-26-10-08`**. PR target: **`main`**.
Remote: `https://github.com/haoliri0/SOFT.git`.
Release name: **`symft_26_10_08`**; Python version: **`2026.10.8`**.

The branch starts at main commit `c89b98514a919240b8afa53a271e08d926d3c987`.
Optimizations are ported from `symft` commit
`89c00eff971847f6aa8cbbbaf15592a724f5bd3d`; the unrelated symft history is
not merged. Main's archive, factorization code, reference/expectation APIs and
wheel release workflow are preserved. See [integration notes](MAIN_INTEGRATION.md).

Preparation does not create a commit, push, PR, tag, release, or PyPI upload.
Changes are left in the working tree for manual inspection and staging.
Fresh fetch/read-only remote checks succeeded on 2026-10-09; fetch again before
publication because a collaborator may subsequently update main or the branch.

## Review

```bash
git branch --show-current
git remote -v
git status --short
git diff --check
git diff --stat
git diff
git ls-files --others --exclude-standard
python3 tools/check_documentation.py
```

The branch name must be `symft-26-10-08`, not `main` or `symft`.
Inspect new files as well: unstaged diff does not include untracked content.
Local build directories, Python environments, native extensions, JIT caches,
and raw measurement logs are ignored. Main's already tracked SOFT v1 archive
is retained unchanged, including four historical object files verified by hash.

Read [validation scope](VALIDATION.md) and [performance limitations](PERFORMANCE.md).
Historical 50B runs and the October 8 performance table are not new 50B
measurements of this main-based source.

## Check the current remote

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD
git ls-remote --heads origin main symft-26-10-08
```

Before the first commit, HEAD is the main base above. After committing, HEAD
will be its descendant. The ancestry check should exit successfully while
remote main stays at the checked base. If main moves, review its new changes;
do not reset this work or force-push. A common ancestor solves the previous
unrelated-history problem, but cannot promise zero conflicts with future edits.

## Manual commit and push

After checking the files and test results:

```bash
git add README.md CHANGELOG.md .gitignore .github docs tools benchmark cpp python
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Port symft_26_10_08 CPU/CUDA optimizations onto main"
git push -u origin HEAD:refs/heads/symft-26-10-08
```

No force push is needed. If a remote branch with that name has advanced, a
normal push rejects the update: fetch and reconcile instead of overwriting it.

On GitHub, choose **Pull requests → New pull request**, with **base: main** and
**compare: symft-26-10-08**. Review the diff and CI results before merging.
Publishing packages or creating a release tag is a separate maintainer action.

The CPU CI includes pytest so main's function-style expectation tests are
actually discovered, in addition to the original unittest suites. Main's
multi-platform wheel workflow is retained; local Linux/CUDA success does not
certify Windows, macOS, ARM, or every GPU architecture.
