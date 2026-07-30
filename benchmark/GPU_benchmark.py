#!/usr/bin/env python3
"""Sequential GPU benchmark for Tsim and SymFT."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import re
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean
from typing import Any, Callable


BASE = Path(__file__).resolve().parent
TOOLS = ("tsim", "symft")
SHOT_SECTIONS = {"tsim": "tsim_call", "symft": "symft_call"}

# Disable JAX preallocation so each case releases GPU memory on process exit.
os.environ.update(
    {
        "JAX_ENABLE_X64": "true",
        "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
)

Sample = Callable[[int, int], dict[str, Any]]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="GPU_config.json")
    parser.add_argument("--gpu", type=int, help="physical GPU used by every case")
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
        for tool in TOOLS:
            case = {
                **circuit,
                "tool": tool,
                "case_id": f"{circuit['id']}__{tool}",
                "shots": config["shots"][SHOT_SECTIONS[tool]][circuit["id"]],
                "seed": config["run"]["seed"],
            }
            if tool == "symft":
                case["gpu_options"] = config["symft_gpu"][circuit["id"]]
            if pattern is None or re.search(pattern, case["case_id"]):
                cases.append(case)
    return cases


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


def counts(shots: int, discarded: int, errors: int, elapsed: float) -> dict[str, Any]:
    return {
        "shots": shots,
        "discarded": discarded,
        "accepted": shots - discarded,
        "logical_errors": errors,
        "sample_s": elapsed,
    }


def count_output(
    detectors: Any, observables: Any, postselect: bool
) -> tuple[int, int]:
    """Count materialized output outside the timed region."""
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


def tsim_sampler(case: dict[str, Any], path: Path) -> tuple[Sample, dict[str, Any]]:
    import jax
    import numpy as np
    import tsim

    if not jax.config.jax_enable_x64 or jax.default_backend() != "gpu":
        raise RuntimeError("Tsim requires a CUDA JAX backend with x64 enabled")
    circuit = tsim.Circuit(path.read_text())
    strategy = case["tsim_strategy"]
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

    sample(case["shots"], case["seed"] - 1)  # Fixed-shape JAX warmup.
    return sample, {
        "version": getattr(tsim, "__version__", "unknown"),
        "precision": "JAX x64",
        "device": str(jax.devices()[0]),
        "strategy": strategy,
    }


def symft_sampler(case: dict[str, Any], path: Path) -> tuple[Sample, dict[str, Any]]:
    import symft

    if not symft.cuda_enabled():
        raise RuntimeError("SymFT was not built with CUDA")
    options = case["gpu_options"]
    circuit = symft.Circuit(path=path)
    sampler = circuit.compile_counts_sampler(
        batch=True,
        observable=0,
        postselect_detectors=case["postselect"],
        batch_size=0,
        sample_chunk_shots=options["shots_per_launch"],
        threads=1,
        cuda=True,
        cuda_mode=options["mode"],
        shots_per_launch=options["shots_per_launch"],
        threads_per_block=options["threads_per_block"],
    )
    sampler.sample(shots=1, stream_id=case["seed"] - 1)
    info = dict(sampler.info)
    if info["backend"] != "cuda":
        raise RuntimeError(f"unexpected SymFT sampler: {info}")

    def sample(shots: int, seed: int) -> dict[str, Any]:
        result = sampler.sample(shots=shots, stream_id=seed)
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
        "cuda_backend": symft.active_cuda_backend(),
        "sampler_info": info,
        **options,
    }


SAMPLERS = {"tsim": tsim_sampler, "symft": symft_sampler}


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
    gpu: int,
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
        "gpu": gpu,
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
        runs = measure(sample, case["shots"], seconds, repeats, case["seed"] + 10_000)
        return {
            **result,
            **metadata,
            "compile_s": compile_s,
            "shots_per_call": case["shots"],
            "runs": runs,
            "shots_per_second_avg": mean(run["shots_per_second"] for run in runs),
        }
    except CompileTimeout as error:
        return {**result, "status": "COMPILE_TIMEOUT", "error": str(error)}
    except Exception as error:
        return {**result, "status": "ERROR", "error": str(error)}


def worker(
    output: mp.Queue,
    case: dict[str, Any],
    circuit_dir: Path,
    gpu: int,
    seconds: float,
    repeats: int,
    timeout: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    output.put(run_case(case, circuit_dir, gpu, seconds, repeats, timeout))


def isolated_case(
    case: dict[str, Any],
    circuit_dir: Path,
    gpu: int,
    seconds: float,
    repeats: int,
    timeout: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    output = context.Queue()
    process = context.Process(
        target=worker,
        args=(output, case, circuit_dir, gpu, seconds, repeats, timeout),
    )
    process.start()
    process.join(timeout + seconds * repeats + 600)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            **case,
            "status": "RUN_TIMEOUT",
            "gpu": gpu,
            "error": "case exceeded the total runtime limit",
        }
    try:
        return output.get(timeout=1)
    except queue.Empty:
        return {
            **case,
            "status": "ERROR",
            "gpu": gpu,
            "error": f"worker exited with code {process.exitcode}",
        }


def report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# GPU benchmark results",
        "",
        "| Circuit | Group | Tool | Status | GPU | Input shots | Call shots | Compile (s) | Runs (shots/s) | Average (shots/s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        rates = ", ".join(
            f"{run['shots_per_second']:.6g}" for run in item.get("runs", [])
        )
        average = item.get("shots_per_second_avg")
        lines.append(
            f"| {item['circuit']} | {item['group']} | {item['tool']} "
            f"| {item['status']} | {item['gpu']} | {item['input_shots']} "
            f"| {item.get('shots_per_call', '-')} "
            f"| {item.get('compile_s', math.nan):.6g} | {rates or '-'} "
            f"| {average:.6g} |"
            if average is not None
            else f"| {item['circuit']} | {item['group']} | {item['tool']} "
            f"| {item['status']} | {item['gpu']} | {item['input_shots']} "
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
    gpu = args.gpu if args.gpu is not None else run["gpu"]
    seconds = args.seconds if args.seconds is not None else run["sample_seconds"]
    repeats = args.repeats if args.repeats is not None else run["repeats"]

    if args.list:
        for case in cases:
            print(f"{case['case_id']}: shots={case['shots']} postselect={case['postselect']}")
        return 0
    if seconds <= 0 or repeats <= 0:
        raise ValueError("seconds and repeats must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    circuit_dir = (config_path.parent / run["circuit_dir"]).resolve()
    report_path = (config_path.parent / run["report"]).resolve()
    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']}", flush=True)
        result = isolated_case(
            case,
            circuit_dir,
            gpu,
            seconds,
            repeats,
            run["compile_timeout_seconds"],
        )
        results.append(result)
        report(report_path, results)
        print(
            f"  {result['status']} "
            f"{result.get('shots_per_second_avg', math.nan):.6g} shots/s",
            flush=True,
        )
    print(f"Report: {report_path}")
    return 0 if all(item["status"] == "OK" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
