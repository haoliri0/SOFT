#!/usr/bin/env python3
"""Resumable CPU-only multiprocess sampling, with pinned persistent compiled workers.

The fsynced journal is authoritative. Work is assigned by unique segment index;
unfinished segments are replayed on resume, never added to completed counts.
Preparation and a separate warmup on every worker precede a start barrier.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('shots', 'discarded', 'accepted', 'logical_errors')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as out:
        json.dump(data, out, indent=2)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


def append(out, row):
    out.write(json.dumps(row, separators=(',', ':')) + '\n')
    out.flush()
    os.fsync(out.fileno())


def parse_cpus(raw):
    cpus = []
    for part in raw.split(','):
        ends = [int(v) for v in part.split('-')]
        if len(ends) == 1:
            cpus.append(ends[0])
        elif len(ends) == 2 and ends[0] <= ends[1]:
            cpus.extend(range(ends[0], ends[1] + 1))
        else:
            raise ValueError('invalid CPU range')
    if not cpus or len(cpus) != len(set(cpus)) or min(cpus) < 0:
        raise ValueError('empty/duplicate/negative CPUs')
    return cpus


def expected_shots(index, config):
    assert 0 <= index < math.ceil(config['target_shots'] / config['segment_shots'])
    return min(config['segment_shots'], config['target_shots'] - index * config['segment_shots'])


def validate_row(row, config, seen):
    index = row['index']
    assert index not in seen, 'duplicate segment'
    assert row['stream_id'] == config['stream_start'] + index, 'wrong stream'
    assert row['shots'] == expected_shots(index, config), 'wrong shot count'
    assert all(isinstance(row[k], int) and row[k] >= 0 for k in FIELDS)
    assert row['shots'] == row['discarded'] + row['accepted'], 'count conservation'
    assert row['logical_errors'] <= row['accepted'], 'invalid error count'
    assert row['cpu'] in config['cpus'] and row['actual_cpu'] == row['cpu'], 'affinity mismatch'
    assert row['finish_monotonic_s'] > row['start_monotonic_s']
    assert row['call_wall_s'] > 0 and row['process_cpu_s'] >= 0
    assert abs(row['finish_monotonic_s'] - row['start_monotonic_s'] - row['call_wall_s']) < 1e-7
    seen.add(index)


def read_journal(path, config):
    rows, seen = [], set()
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)  # fail closed on partial/corrupt records
            validate_row(row, config, seen)
            rows.append(row)
    return rows, seen


def summary(rows, config):
    counts = {k: sum(r[k] for r in rows) for k in FIELDS}
    counts.update(target_shots=config['target_shots'], completed_segments=len(rows),
                  complete=counts['shots'] == config['target_shots'], updated_at=now())
    counts['discard_rate'] = counts['discarded'] / counts['shots'] if counts['shots'] else None
    counts['error_rate_after_postselection'] = counts['logical_errors'] / counts['accepted'] if counts['accepted'] else None
    # Durations are calculated separately per invocation; monotonic clocks need
    # not be comparable across reboot. A completed one-session run uses one span.
    durations = []
    for session in {r['session'] for r in rows}:
        group = [r for r in rows if r['session'] == session]
        durations.append(max(r['finish_monotonic_s'] for r in group) -
                         min(r['start_monotonic_s'] for r in group))
    counts['sampling_wall_s'] = sum(durations)
    counts['attempted_shots_per_s'] = counts['shots'] / sum(durations) if durations else None
    counts['worker_process_cpu_s'] = sum(r['process_cpu_s'] for r in rows)
    counts['sampling_process_cpu_equivalents'] = counts['worker_process_cpu_s'] / sum(durations) if durations else None
    return counts


def cpu_ticks(cpus):
    wanted = {f'cpu{c}' for c in cpus}
    return {parts[0]: list(map(int, parts[1:9])) for line in Path('/proc/stat').read_text().splitlines()
            if (parts := line.split()) and parts[0] in wanted}


def process_stats(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'cpu_s': (int(stat[11]) + int(stat[12])) / os.sysconf('SC_CLK_TCK'),
                'rss_bytes': int(stat[21]) * os.sysconf('SC_PAGE_SIZE'),
                'affinity': sorted(os.sched_getaffinity(pid)), 'state': stat[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def main():
    if not __debug__:
        raise RuntimeError("Do not use python -O: journal integrity checks must remain enabled")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shots', type=int, required=True)
    ap.add_argument('--segment-shots', type=int, default=5_000_000)
    ap.add_argument('--warmup-shots', type=int, default=500_000)
    ap.add_argument('--cpus', required=True)
    ap.add_argument('--stream-start', type=int, default=50_000_000)
    ap.add_argument('--warmup-stream-start', type=int, default=49_000_000)
    ap.add_argument('--label', required=True)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--worker', type=Path, default=ROOT / 'build-cpu/cpp/symft_cpu_stream_worker')
    ap.add_argument('--output-root', type=Path, default=ROOT / 'results')
    ap.add_argument('--static-library', type=Path,
                    help='Optional static library path, for build provenance only')
    args = ap.parse_args()
    cpus = parse_cpus(args.cpus)
    assert set(cpus) <= os.sched_getaffinity(0), 'CPUs outside allowed mask'
    assert args.shots > 0 and args.segment_shots > 0 and args.warmup_shots > 0
    assert args.label and all(c.isalnum() or c in '_-' for c in args.label)
    segments = math.ceil(args.shots / args.segment_shots)
    assert 0 <= args.stream_start < args.stream_start + segments <= 2**64
    assert 0 <= args.warmup_stream_start < args.warmup_stream_start + len(cpus) <= 2**64
    assert set(range(args.warmup_stream_start, args.warmup_stream_start + len(cpus))).isdisjoint(
        range(args.stream_start, args.stream_start + segments))
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root.resolve() / args.label
    output.mkdir(exist_ok=args.resume)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_worker = args.worker.resolve()
        original_circuit = ROOT / 'benchmark/circuit/msc_d5_inject_cultivate_p1e-3.stim'
        worker, circuit = output / 'cpu_stream_worker', output / original_circuit.name
        if not args.resume:
            shutil.copy2(original_worker, worker)
            shutil.copy2(original_circuit, circuit)
            shutil.copy2(Path(__file__), output / Path(__file__).name)
            shutil.copy2(Path(__file__).with_name('cpu_stream_worker.cpp'), output / 'cpu_stream_worker.cpp')
        config = dict(target_shots=args.shots, segment_shots=args.segment_shots,
            warmup_shots=args.warmup_shots, cpus=cpus, stream_start=args.stream_start,
            warmup_stream_start=args.warmup_stream_start, precision='FP64', cpu_backend='compiled',
            cpu_real_gauge=True, cpu_rng='cpu-shot-v1', sample_chunk_shots=2048,
            threads_per_worker=1, observable=0, postselect_detectors=True,
            circuit_sha256=sha256(circuit), worker_sha256=sha256(worker),
            script_sha256=sha256(Path(__file__)))
        if args.resume:
            assert json.loads((output / 'metadata.json').read_text())['config'] == config, 'identity mismatch'
        else:
            topology = {str(c): {field: Path(f'/sys/devices/system/cpu/cpu{c}/topology/{field}').read_text().strip()
                        for field in ('physical_package_id', 'core_id', 'thread_siblings_list')} for c in cpus}
            atomic_json(output / 'metadata.json', dict(config=config, created_at=now(), topology=topology,
                lscpu=subprocess.check_output(['lscpu'], text=True), python=sys.executable,
                static_library_sha256=sha256(args.static_library) if args.static_library else None))
        rows, seen = read_journal(output / 'chunks.jsonl', config)
        state = summary(rows, config)
        if state['complete']:
            atomic_json(output / 'summary.json', state)
            print(json.dumps(state), flush=True)
            return
        session = f'{time.time_ns()}_{os.getpid()}'
        start_wall = time.monotonic()
        processes, pending, selector = {}, {}, selectors.DefaultSelector()
        todo = iter(i for i in range(segments) if i not in seen)
        session_data = dict(session=session, pid=os.getpid(), started_at=now(), initial_shots=state['shots'])
        session_path = output / f'session_{session}.json'
        production_start = None
        error = None
        try:
            with (output / 'chunks.jsonl').open('a') as journal, \
                 (output / 'telemetry.jsonl').open('a') as telemetry, \
                 (output / 'warmup.jsonl').open('a') as warmup_out:
                for cpu in cpus:
                    stderr = (output / f'worker_{session}_cpu{cpu}.stderr').open('w')
                    proc = subprocess.Popen([str(worker), str(circuit), str(cpu)], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
                    stderr.close()
                    processes[cpu] = proc
                    selector.register(proc.stdout, selectors.EVENT_READ, cpu)
                session_data['workers'] = {str(c): p.pid for c, p in processes.items()}
                atomic_json(session_path, session_data)
                ready, warmed = {}, set()
                last_status = last_print = 0.0

                def send(cpu, index, stream, shots, phase):
                    assert cpu not in pending
                    pending[cpu] = dict(index=index, stream_id=stream, shots=shots, phase=phase)
                    proc = processes[cpu]
                    proc.stdin.write(f'{index} {stream} {shots}\n')
                    proc.stdin.flush()

                def dispatch(cpu):
                    index = next(todo, None)
                    if index is not None:
                        send(cpu, index, args.stream_start + index, expected_shots(index, config), 'production')

                while True:
                    for key, _ in selector.select(timeout=1):
                        cpu = key.data
                        line = key.fileobj.readline()
                        if not line:
                            raise RuntimeError(f'worker CPU {cpu} unexpectedly exited ({processes[cpu].poll()})')
                        row = json.loads(line)
                        if row['event'] == 'ready':
                            assert cpu not in ready and row['cpu'] == cpu
                            assert row['cpu_backend'] == 'compiled' and row['cpu_real_gauge'] and row['threads'] == 1
                            assert row['sample_chunk_shots'] == 2048 and row['max_k'] == 10
                            ready[cpu] = row
                            # Warmup streams are separate; no warmup counts enter production.
                            send(cpu, 2**63 + cpus.index(cpu), args.warmup_stream_start + cpus.index(cpu),
                                 args.warmup_shots, 'warmup')
                            continue
                        request = pending.pop(cpu)
                        assert row['event'] == 'result'
                        for k in ('index', 'stream_id', 'shots'):
                            assert row[k] == request[k], f'worker response mismatch: {k}'
                        row.update(cpu=cpu, pid=processes[cpu].pid, session=session, recorded_at=now())
                        if request['phase'] == 'warmup':
                            append(warmup_out, row)
                            warmed.add(cpu)
                            if len(warmed) == len(cpus):
                                session_data.update(ready=ready, prepared_and_warmed_wall_s=time.monotonic() - start_wall,
                                                    production_started_at=now(), production_start_monotonic_s=time.monotonic())
                                production_start = session_data['production_start_monotonic_s']
                                atomic_json(session_path, session_data)
                                print(json.dumps({'event': 'production_started', **session_data}), flush=True)
                                for c in cpus:
                                    dispatch(c)
                        else:
                            validate_row(row, config, seen)
                            append(journal, row)
                            rows.append(row)
                            dispatch(cpu)
                    stamp = time.monotonic()
                    if stamp - last_status >= 10 or (production_start is not None and not pending):
                        state = summary(rows, config)
                        state.update(session=session, live_workers=len(processes), active_segments=len(pending),
                                     session_wall_s=stamp - start_wall,
                                     production_elapsed_s=stamp - production_start if production_start else None)
                        atomic_json(output / 'summary.json', state)
                        append(telemetry, dict(at=now(), monotonic_s=stamp, session=session, shots=state['shots'],
                            active_segments=len(pending), cpu_ticks=cpu_ticks(cpus),
                            processes={str(c): process_stats(p.pid) for c, p in processes.items()}))
                        last_status = stamp
                        if stamp - last_print >= 30 or state['complete']:
                            print(json.dumps({'event': 'progress', **state}), flush=True)
                            last_print = stamp
                    if production_start is not None and not pending:
                        assert state['complete'] and len(seen) == segments
                        break
                    if production_start is None and stamp - start_wall > 600:
                        raise RuntimeError('preparation/warmup exceeded 600 seconds')
                    for cpu, proc in processes.items():
                        if proc.poll() is not None:
                            raise RuntimeError(f'worker CPU {cpu} exited unexpectedly')
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            # Only our child processes are touched, never other machine users.
            for proc in processes.values():
                if proc.poll() is None:
                    if error:
                        proc.terminate()
                    else:
                        proc.stdin.close()
            for proc in processes.values():
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            selector.close()
            session_data.update(finished_at=now(), wall_s=time.monotonic() - start_wall, error=error)
            atomic_json(session_path, session_data)
            state = summary(rows, config)
            state['error'] = error
            atomic_json(output / 'summary.json', state)
            print(json.dumps({'event': 'stopped' if error else 'complete', **state}), flush=True)


if __name__ == '__main__':
    main()
