# Long-run validation and evidence checks

These helpers belong to SOFT v2 (SymFT). They are opt-in experiments, not part
of package installation and not a reason to run 50B shots during every review.
Use short smoke tests before allocating a CPU socket or a GPU for a long run.

## Check the archived evidence

```bash
python3 benchmark/validation/check_results.py
python3 benchmark/validation/check_results.py --experiment-root /path/to/SymFT_CUDA_Astra
```

The first command independently checks integer totals, rates, confidence
intervals, CPU repeat medians, comparison statistics, and circuit hashes in
the [curated evidence](../results/optimization_202609.json). The optional
archive check additionally reads the original source files and production
journals, checks their hashes and stream uniqueness, and recomputes counts and
timing totals. It never overwrites those archives. Interval helpers are scoped
to fewer than 1000 observed errors; they are not a general binomial library.

## Dated single-core comparison

[`compare_cpu_versions.py`](compare_cpu_versions.py) compares three separately
built binaries, on one allowed idle logical CPU, with rotating run order and
separate warmups. It freezes the executables, saves raw output and source/binary
hashes, and verifies identical counters for the before/after compiled pair.
The unmodified old CLI replays stream 0; no timing repeat is presented as
additional independent rare-event evidence.

Build the old commit and the current release with the same policy:

```bash
# New directories only; neither command changes the active checkout's branch.
git worktree add --detach build-old-symft-source e86c6a92525650744ce0dc2a65126e1ab951882d
cmake -S build-old-symft-source -B build-old-symft -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_NATIVE=ON -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build-old-symft --target symft_rate_bench -j 8

cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=OFF -DSYMFT_CPP_NATIVE=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build-release -j 8
ctest --test-dir build-release --output-on-failure

python3 benchmark/validation/compare_cpu_versions.py \
  --legacy-binary build-old-symft/cpp/symft_rate_bench \
  --before-binary /path/to/frozen-september-build/cpp/symft_rate_bench \
  --after-binary build-release/cpp/symft_rate_bench \
  --legacy-commit e86c6a92525650744ce0dc2a65126e1ab951882d \
  --cpu 5 --repeats 7 --d3-shots 12000000 --d5-shots 1500000 \
  --stream-start 261008100 --output review-artifacts/new-cpu-comparison
```

The September compiled source was an uncommitted experimental snapshot, not a
published tag. Its frozen binary and source were preserved locally before this
update; their provenance is recorded in the dated evidence. To independently
reproduce the incremental comparison, obtain that snapshot. Do not substitute
the current binary as `--before-binary`. The old-commit/current-release total
comparison can also be rerun directly with the two CLI executables above.

The optional `--export NEW_FILE.json` writes compact public evidence without
overwriting an existing file. The bundled dated record is checked by:

```bash
python3 benchmark/validation/check_cpu_comparison.py
# Use only while checking the same measured source revision:
python3 benchmark/validation/check_cpu_comparison.py --check-current-sources
```

These are quick read-only checks, not a request to rerun the benchmarks.

## Persistent CPU socket run (Linux)

The worker binds to its selected logical CPU **before** constructing a compiled
single-worker sampler. Every process prepares once, warms up with a separate
stream, and waits for the controller's production barrier. Segments have unique
indices/streams; completed records are fsynced to a journal. A per-run lock
prevents concurrent controllers. Resume replays only unfinished segments and
does not add them to counts until completion.

The helper deliberately targets the canonical d=5 fixture and requires the
compiled real-gauge backend. It refuses a silent backend fallback. It uses
Linux affinity, `/proc`, `/sys`, and `fcntl`; other operating systems are not
supported by this helper. **Do not run it with Python `-O`**, which would disable
its integrity assertions; the entry point rejects that mode.

Build from the repository root:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DSYMFT_CPP_ENABLE_CUDA=OFF -DSYMFT_CPP_NATIVE=ON \
  -DSYMFT_CPP_BUILD_VALIDATION_TOOLS=ON \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
cmake --build build-cpu -j 8
ctest --test-dir build-cpu --output-on-failure

SYMFT_TEST_WORKER="$PWD/build-cpu/cpp/symft_cpu_stream_worker" \
  python3 -m unittest discover -s benchmark/validation -p 'test_*.py' -v
```

First inspect topology with `lscpu -e=CPU,SOCKET,CORE,ONLINE`. The recorded
Xeon Gold 5218R socket used logical CPUs `0-19,40-59`: **20 physical cores,
40 logical CPUs**. Those IDs are machine-specific; choose the allowed IDs for
one socket on your host. The compiled backend itself still has `threads=1`.

Example short run on two allowed logical CPUs:

```bash
python3 benchmark/validation/long_run_cpu_socket.py \
  --cpus 0,1 --shots 100257 --segment-shots 50000 --warmup-shots 2049 \
  --stream-start 48000000 --warmup-stream-start 47900000 \
  --label cpu_smoke
```

The archived 50B configuration, **only when you intend a multi-hour run**:

```bash
python3 benchmark/validation/long_run_cpu_socket.py \
  --cpus 0-19,40-59 --shots 50000000000 --segment-shots 5000000 \
  --warmup-shots 500000 --stream-start 50000000 \
  --warmup-stream-start 49000000 --label cpu_d5_50B \
  --static-library build-cpu/cpp/libsymft_cpp.a
```

Use a new label for each new experiment. Add `--resume` with identical options
to resume an existing label; changing the configuration or controller identity
is rejected. `--worker` can select another built worker and `--output-root`
can select a results directory. The worker, input, and controller are frozen
inside that run directory. The binary hash is recorded; do not expect a newly
built worker hash to equal the archived 2026-09-28 binary.

Output includes `metadata.json`, `chunks.jsonl`, `summary.json`, warmup records,
telemetry, and per-worker stderr. Raw outputs under root `results/` are ignored
by Git. Keep warmup/replay counts separate from production. Aggregate rates
from total counts/time, never by averaging per-worker rates or adding parallel
worker durations as if they were common wall time.

## GPU long-run recipe

Build a CUDA FP64 Python extension and activate the environment in the
[CUDA guide](../../docs/optimization/CUDA.md). The archived d=5 long run used
100 calls of 500M attempts, streams 3,000,000 through 3,000,099, one reusable
sampler, and 524,288 shots/launch. A minimal reproduction loop is:

```python
import json
import symft

circuit = symft.read_stim_file("benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim")
sampler = circuit.compile_counts_sampler(
    cuda=True, cuda_mode="gpu_presample_expressions",
    observable=0, postselect_detectors=True,
    shots_per_launch=524288, threads_per_block=128,
)
for index in range(100):
    result = sampler.sample(shots=500_000_000, stream_id=3_000_000 + index)
    print(json.dumps({"index": index, "stream_id": 3_000_000 + index,
                      "result": result}), flush=True)
```

This is a minimal streaming recipe, **not** the original resumable archive
controller. Use a unique output file and retain each successful call before
continuing; after interruption do not blindly concatenate reruns of the same
streams as independent samples. Reusing the archived stream schedule is a
reproduction, not fresh independent evidence to pool with it. Record extension
and circuit hashes, environment flags, driver/toolkit/device, and separate
preparation and sampling times. Confirm FP64 in the build configuration: the
runtime backend-name string alone does not certify its precision.
