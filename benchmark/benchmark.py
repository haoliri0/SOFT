#!/usr/bin/env python3
"""Sequential single-CPU benchmark for Stim, Clifft, Tsim, and SymFT."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Callable


BASE = Path(__file__).resolve().parent
DEFAULT_SAMPLE_SECONDS = 60.0
SHOT_SECTIONS = {
    "stim": "stim_batch",
    "tsim": "tsim_call",
    "clifft": "clifft_call",
    "symft": "symft_call",
}

# Limit libraries before any simulator is imported.
os.environ.update(
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "JAX_ENABLE_X64": "true",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    }
)

Sample = Callable[[int, int], dict[str, Any]]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--cpu", type=int, help="logical CPU used by every tool")
    parser.add_argument("--only", help="regex matched against circuit__tool")
    parser.add_argument("--seconds", type=float, help="override seconds per repeat")
    parser.add_argument("--repeats", type=int, help="override repeat count")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def load_config(name: str) -> tuple[Path, dict[str, Any]]:
    path = Path(name)
    if not path.is_absolute():
        path = BASE / path
    path = path.resolve()
    return path, json.loads(path.read_text())


def make_cases(config: dict[str, Any], pattern: str | None) -> list[dict[str, Any]]:
    cases = []
    for circuit in config["circuits"]:
        for tool in circuit["tools"]:
            case = {
                **circuit,
                "tool": tool,
                "case_id": f"{circuit['id']}__{tool}",
                "shots": config["shots"][SHOT_SECTIONS[tool]][circuit["id"]],
                "seed": config["run"]["seed"],
            }
            if pattern is None or re.search(pattern, case["case_id"]):
                cases.append(case)
    return cases


def pin_cpu(cpu: int) -> None:
    allowed = os.sched_getaffinity(0)
    if cpu not in allowed:
        raise ValueError(f"CPU {cpu} is unavailable; choose from {sorted(allowed)}")
    os.sched_setaffinity(0, {cpu})
    if os.sched_getaffinity(0) != {cpu}:
        raise RuntimeError("failed to pin the process to one logical CPU")


class CompileTimeout(RuntimeError):
    pass


@contextmanager
def compile_timeout(seconds: int):
    """Stop simulator preparation after the configured deadline."""

    def expired(_signum: int, _frame: Any) -> None:
        raise CompileTimeout(f"compilation exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def count_output(
    detectors: Any, observables: Any, postselect: bool
) -> tuple[int, int]:
    """Count output outside the timed sampling region."""
    import numpy as np

    dets = np.asarray(detectors)
    obs = np.asarray(observables)
    discarded = (
        np.any(dets, axis=1)
        if postselect and dets.shape[1]
        else np.zeros(dets.shape[0], dtype=np.bool_)
    )
    errors = int(np.count_nonzero((~discarded) & obs[:, 0])) if obs.shape[1] else 0
    return int(np.count_nonzero(discarded)), errors


def stim_sampler(case: dict[str, Any], path: Path) -> tuple[Sample, dict[str, Any]]:
    import numpy as np
    import stim

    sampler = stim.Circuit.from_file(str(path)).compile_detector_sampler()
    sampler.sample(min(case["shots"], 1024), separate_observables=True, bit_packed=True)

    def sample(shots: int, _seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        _, obs = sampler.sample(shots, separate_observables=True, bit_packed=True)
        elapsed = time.perf_counter() - started
        errors = int(np.count_nonzero(obs[:, 0] & 1)) if obs.shape[1] else 0
        return counts(shots, 0, errors, elapsed)

    return sample, {"version": stim.__version__, "precision": "binary stabilizer"}


def tsim_sampler(case: dict[str, Any], path: Path) -> tuple[Sample, dict[str, Any]]:
    import jax
    import numpy as np
    import tsim

    if not jax.config.jax_enable_x64:
        raise RuntimeError("Tsim requires JAX_ENABLE_X64=true")
    circuit = tsim.Circuit(path.read_text())
    strategy = case.get("tsim_strategy", "cat5")
    sampler = circuit.compile_detector_sampler(strategy=strategy, seed=case["seed"])
    mask = np.ones(circuit.num_detectors, dtype=np.bool_) if case["postselect"] else None

    def sample(shots: int, _seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        dets, obs = sampler.sample(
            shots,
            batch_size=None,
            separate_observables=True,
            postselection_mask=mask,
        )
        elapsed = time.perf_counter() - started
        discarded, errors = count_output(dets, obs, case["postselect"])
        return counts(shots, discarded, errors, elapsed)

    sample(case["shots"], case["seed"] - 1)  # JAX warmup.
    return sample, {
        "version": getattr(tsim, "__version__", "unknown"),
        "precision": "JAX x64",
        "strategy": strategy,
    }


def clifft_sampler(case: dict[str, Any], path: Path) -> tuple[Sample, dict[str, Any]]:
    import clifft

    clifft.set_num_threads(1)
    circuit = clifft.parse_file(str(path))
    hir = clifft.trace(circuit)
    clifft.default_hir_pass_manager().run(hir)
    reference = clifft.compute_reference_syndrome(hir)
    mask = [1] * len(reference["detectors"]) if case["postselect"] else []
    program = clifft.lower(
        hir,
        postselection_mask=mask,
        expected_detectors=reference["detectors"],
        expected_observables=reference["observables"],
    )
    clifft.default_bytecode_pass_manager().run(program)
    clifft.sample_survivors(
        program, 1, seed=case["seed"] - 1, keep_records=False
    )

    def sample(shots: int, seed: int) -> dict[str, Any]:
        started = time.perf_counter()
        result = clifft.sample_survivors(
            program, shots, seed=seed, keep_records=False
        )
        return counts(
            int(result.total_shots),
            int(result.discards),
            int(result.logical_errors),
            time.perf_counter() - started,
        )

    return sample, {
        "version": clifft.version(),
        "precision": "FP64",
        "threads": clifft.get_num_threads(),
    }


def symft_sampler(case: dict[str, Any], path: Path) -> tuple[Sample, dict[str, Any]]:
    import symft

    circuit = symft.Circuit(path=path)
    sampler = circuit.compile_counts_sampler(
        batch=True,
        observable=0,
        postselect_detectors=case["postselect"],
        batch_size=0,
        threads=1,
        cuda=False,
    )
    sampler.sample(shots=1, stream_id=case["seed"] - 1)
    info = dict(sampler.info)
    if info["backend"] != "batch" or info["threads"] != 1:
        raise RuntimeError(f"unexpected SymFT sampler: {info}")

    def sample(shots: int, seed: int) -> dict[str, Any]:
        result = sampler.sample(shots=shots, stream_id=seed)
        if result["active_threads"] != 1:
            raise RuntimeError("SymFT used more than one thread")
        return {
            "shots": int(result["shots"]),
            "discarded": int(result["discarded"]),
            "accepted": int(result["accepted"]),
            "logical_errors": int(result["logical_errors"]),
            "sample_s": float(result["timing"]["sample_s"]),
        }

    return sample, {
        "version": symft.__version__,
        "precision": "FP64",
        "simd_backend": symft.simd_backend(),
        "sampler_info": info,
    }


SAMPLERS = {
    "stim": stim_sampler,
    "tsim": tsim_sampler,
    "clifft": clifft_sampler,
    "symft": symft_sampler,
}


def counts(shots: int, discarded: int, errors: int, elapsed: float) -> dict[str, Any]:
    return {
        "shots": shots,
        "discarded": discarded,
        "accepted": shots - discarded,
        "logical_errors": errors,
        "sample_s": elapsed,
    }


def measure(
    sample: Sample, shots: int, seconds: float, repeats: int, seed: int
) -> list[dict[str, Any]]:
    runs = []
    for repeat in range(repeats):
        total = {"shots": 0, "discarded": 0, "accepted": 0, "logical_errors": 0}
        elapsed = 0.0
        calls = 0
        while elapsed < seconds:
            result = sample(shots, seed + repeat * 1_000_000 + calls)
            for key in total:
                total[key] += result[key]
            elapsed += result["sample_s"]
            calls += 1
        runs.append(
            {
                "repeat": repeat + 1,
                **total,
                "sample_s": elapsed,
                "api_calls": calls,
                "shots_per_second": total["shots"] / elapsed,
            }
        )
    return runs


def run_case(
    case: dict[str, Any],
    circuit_dir: Path,
    cpu: int,
    seconds: float,
    repeats: int,
    timeout: int,
) -> dict[str, Any]:
    result = {
        "case_id": case["case_id"],
        "circuit": case["id"],
        "group": case["group"],
        "tool": case["tool"],
        "status": "OK",
        "cpu": cpu,
        "postselect": case["postselect"],
        "input_shots": case["shots"],
        "sample_seconds": seconds,
        "repeats": repeats,
    }
    try:
        started = time.perf_counter()
        with compile_timeout(timeout):
            sample, metadata = SAMPLERS[case["tool"]](case, circuit_dir / case["file"])
        compile_s = time.perf_counter() - started
        shots = case["shots"]
        runs = measure(sample, shots, seconds, repeats, case["seed"] + 10_000)
        return {
            **result,
            **metadata,
            "compile_s": compile_s,
            "shots_per_call": shots,
            "runs": runs,
            "shots_per_second_avg": mean(run["shots_per_second"] for run in runs),
        }
    except CompileTimeout as error:
        return {**result, "status": "COMPILE_TIMEOUT", "error": str(error)}
    except Exception as error:
        return {**result, "status": "ERROR", "error": str(error)}


def report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark results",
        "",
        "| Circuit | Group | Tool | Status | CPU | Input shots | Call shots | Compile (s) | Runs (shots/s) | Average (shots/s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        rates = ", ".join(
            f"{run['shots_per_second']:.6g}" for run in item.get("runs", [])
        )
        average = item.get("shots_per_second_avg")
        lines.append(
            f"| {item['circuit']} | {item['group']} | {item['tool']} "
            f"| {item['status']} | {item['cpu']} | {item['input_shots']} "
            f"| {item.get('shots_per_call', '-')} "
            f"| {item.get('compile_s', math.nan):.6g} | {rates or '-'} "
            f"| {average:.6g} |"
            if average is not None
            else f"| {item['circuit']} | {item['group']} | {item['tool']} "
            f"| {item['status']} | {item['cpu']} | {item['input_shots']} "
            f"| - | - | - | - |"
        )
    lines += [
        "",
        "| Sample seconds per repeat | Repeats |",
        "|---:|---:|",
        f"| {results[0]['sample_seconds']:g} | {results[0]['repeats']} |"
        if results
        else "| - | - |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = arguments()
    config_path, config = load_config(args.config)
    run = config["run"]
    cases = make_cases(config, args.only)
    cpu = args.cpu if args.cpu is not None else run["cpu"]
    seconds = (
        args.seconds
        if args.seconds is not None
        else run.get("sample_seconds", DEFAULT_SAMPLE_SECONDS)
    )
    repeats = args.repeats if args.repeats is not None else run["repeats"]

    if args.list:
        for case in cases:
            print(
                f"{case['case_id']}: shots={case['shots']} "
                f"postselect={case['postselect']}"
            )
        return 0
    if seconds <= 0 or repeats <= 0:
        raise ValueError("seconds and repeats must be positive")

    pin_cpu(cpu)
    circuit_dir = (config_path.parent / run["circuit_dir"]).resolve()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
        result = run_case(
            case,
            circuit_dir,
            cpu,
            seconds,
            repeats,
            run["compile_timeout_seconds"],
        )
        results.append(result)
        print(
            f"  {result['status']} "
            f"{result.get('shots_per_second_avg', math.nan):.6g} shots/s",
            flush=True,
        )

    report_path = (config_path.parent / run["report"]).resolve()
    report(report_path, results)
    print(f"Report: {report_path}")
    return 0 if all(item["status"] == "OK" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
