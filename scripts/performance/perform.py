import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from datetime import datetime

import numpy as np

script_dir_path = os.path.dirname(__file__)
project_dir_path = os.path.join(script_dir_path, "../..")
sys.path.append(project_dir_path)

from scripts.utils.stn import Args, make_cmd

span_time_pattern = re.compile(r"span_time: (\d+(\.\d+)?) s")
avg_speed_pattern = re.compile(r"avg_speed: (\d+(\.\d+)?) shot/s")
project_version_pattern = re.compile(r"version = (.+)")


def get_timestamp() -> str:
    now = datetime.now()
    now = now.astimezone()
    return now.isoformat()


def get_repo_version() -> str:
    pyproject_file_path = os.path.join(project_dir_path, "pyproject.toml")
    with open(pyproject_file_path, "rb") as pyproject_io:
        pyproject_dict = tomllib.load(pyproject_io)
    repo_version = pyproject_dict["project"]["version"]
    return repo_version


def get_repo_revision() -> str:
    cmd = ["git", "rev-parse", "HEAD"]
    repo_revision = subprocess.check_output(cmd, text=True)
    return repo_revision.strip()


def get_device_name() -> str:
    cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]
    device_name = subprocess.check_output(cmd, text=True)
    return device_name.strip()


def main(*,
    exec_file_path: str,
    records_file_path: str,
    circuit_file_path: str,
    experiment_name: str,
    repeats_n: int = 1,
    shots_n: int | Iterable[int] = 1,
    qubits_n: int,
    entries_m: int,
    mem_ints_m: int = 0,
    mem_flts_m: int = 0,
):
    exec_file_path = os.path.abspath(exec_file_path)
    timestamp = get_timestamp()
    repo_version = get_repo_version()
    repo_revision = get_repo_revision()
    device_name = get_device_name()

    with open(circuit_file_path, "rt") as circuit_io:
        circuit_str = circuit_io.read()
    circuit_name = os.path.basename(circuit_file_path)
    circuit_sha256 = hashlib.sha256(circuit_str.encode('utf-8')).hexdigest()

    rng = np.random.default_rng()
    cases_shots_n = shots_n if isinstance(shots_n, Iterable) else (shots_n,)
    for shots_n in cases_shots_n:
        for _ in range(repeats_n):
            seed = int(np.frombuffer(rng.bytes(4), np.uint32)[0])

            cmd = make_cmd(exec_file_path, Args(
                shots_n=shots_n,
                qubits_n=qubits_n,
                entries_m=entries_m,
                mem_ints_m=mem_ints_m,
                mem_flts_m=mem_flts_m,
                seed=seed))

            try:
                process = subprocess.run(
                    cmd,
                    text=True,
                    check=True,
                    input=circuit_str,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as process_error:
                print(process_error.stderr, file=sys.stderr)
                raise

            stderr_str = process.stderr
            span_time_match = span_time_pattern.search(stderr_str)
            span_time = float(span_time_match.group(1))
            avg_speed_match = avg_speed_pattern.search(stderr_str)
            avg_speed = float(avg_speed_match.group(1))

            record = dict(
                timestamp=timestamp,
                repo_version=repo_version,
                repo_revision=repo_revision,
                device_name=device_name,
                circuit_name=circuit_name,
                circuit_sha256=circuit_sha256,
                experiment_name=experiment_name,
                shots_n=shots_n,
                qubits_n=qubits_n,
                entries_m=entries_m,
                mem_ints_m=mem_ints_m,
                mem_flts_m=mem_flts_m,
                seed=seed,
                span_time=span_time,
                avg_speed=avg_speed)
            with open(records_file_path, "at") as records_io:
                records_io.write(json.dumps(record))
                records_io.write("\n")
            for key, value in record.items():
                print(f"{key}={value}")
            print()


if __name__ == '__main__':
    main(
        exec_file_path=os.path.join(project_dir_path, "cmake-build-release/stn_cuda_exec"),
        records_file_path=os.path.join(script_dir_path, "records.jsonl"),
        circuit_file_path=os.path.join(script_dir_path, "circuit_3.txt"),
        experiment_name="performance",
        shots_n=(2 ** n for n in count(8)),
        repeats_n=4,
        qubits_n=42,
        entries_m=2048,
        mem_ints_m=0,
        mem_flts_m=0)
