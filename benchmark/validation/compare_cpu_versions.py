#!/usr/bin/env python3
"""Serial, pinned three-way CPU comparison; preserve inputs, binaries and counts.

Run from a checkout. The legacy binary must be built from the named old commit;
the before binary is the frozen, previously optimized compiled CPU build.
Outputs are deliberately written only to a new directory. Warmups and replays
are identified and must not be pooled into a rare-error measurement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ('sampled_shots', 'discarded', 'accepted', 'logical_errors')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    result = {}
    for distance in (3, 5):
        for name in ('legacy', 'before', 'after'):
            selected = [r for r in rows if not r['warmup'] and
                        r['distance'] == distance and r['variant'] == name]
            if not selected:
                continue
            rates = [float(r['fields']['sample_shots_per_s']) for r in selected]
            result[f'd{distance}_{name}'] = {
                'repeats': len(selected), 'median': statistics.median(rates),
                'min': min(rates), 'max': max(rates),
                **{key: sum(int(r['fields'][key]) for r in selected) for key in COUNTS},
            }
        if f'd{distance}_after' in result:
            current = result[f'd{distance}_after']
            for baseline in ('legacy', 'before'):
                current[f'speedup_vs_{baseline}'] = current['median'] / result[f'd{distance}_{baseline}']['median']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-binary', required=True, type=Path)
    parser.add_argument('--before-binary', required=True, type=Path)
    parser.add_argument('--after-binary', required=True, type=Path)
    parser.add_argument('--legacy-commit', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--export', type=Path, help='write compact public evidence to a NEW JSON file')
    parser.add_argument('--cpu', required=True, type=int)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--d3-shots', type=int, default=12_000_000)
    parser.add_argument('--d5-shots', type=int, default=1_500_000)
    parser.add_argument('--stream-start', type=int, default=261008000)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.d3_shots, args.d5_shots) < 0 or args.stream_start < 1:
        parser.error('positive repeats/stream start and nonnegative shot counts required')
    if args.cpu not in os.sched_getaffinity(0):
        parser.error('requested CPU is not in this process affinity mask')
    if args.export is not None and args.export.exists():
        parser.error('export file already exists')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {}
    for name in ('legacy', 'before', 'after'):
        source = getattr(args, f'{name}_binary').resolve()
        frozen = out / f'{name}.frozen'
        shutil.copy2(source, frozen)
        binaries[name] = {'source': str(source), 'sha256': digest(frozen)}
    sibling_list = Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list').read_text().strip()
    siblings = set()
    for part in sibling_list.split(','):
        lo, _, hi = part.partition('-')
        siblings.update(f'cpu{i}' for i in range(int(lo), int(hi or lo) + 1))

    def ticks():
        return {parts[0]: list(map(int, parts[1:])) for line in Path('/proc/stat').read_text().splitlines()
                if (parts := line.split()) and parts[0] in siblings}

    data = {
        'schema_version': 1, 'date': time.strftime('%Y-%m-%d'), 'release': 'symft_26_10_08',
        'legacy_commit': args.legacy_commit, 'cpu': args.cpu, 'smt_siblings': sibling_list,
        'platform': platform.platform(), 'binaries': binaries, 'runs': [], 'summary': {},
        'timer': 'native sampling-call timer; preparation excluded; attempted shots/s',
        'precision': 'FP64', 'repeats': args.repeats,
        'legacy_rng_note': 'Unmodified old CLI has no stream-id flag and replays stream 0. Timing repeats are not independent statistics.',
        'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          sorted((ROOT / 'cpp/src').rglob('*.cpp')) + sorted((ROOT / 'cpp/src').rglob('*.hpp'))},
    }
    for name, cmd in (('hardware', ['lscpu']), ('compiler', ['c++', '--version']),
                      ('source_diff', ['git', 'diff', 'HEAD']), ('git_status', ['git', 'status', '--short'])):
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
        (out / f'{name}.txt').write_text(proc.stdout)
    for repeat in range(-1, args.repeats):
        order = ['legacy', 'before', 'after']
        if repeat >= 0:
            order = order[repeat % 3:] + order[:repeat % 3]
        for distance, requested in ((3, args.d3_shots), (5, args.d5_shots)):
            if requested == 0:
                continue
            shots = min(requested, 100_000) if repeat < 0 else requested
            circuit = ROOT / f'benchmark/circuit/msc_d{distance}_inject_cultivate_p1e-3.stim'
            paired = {}
            for name in order:
                command = ['taskset', '-c', str(args.cpu), str(out / f'{name}.frozen'),
                           '--circuit', str(circuit), '--shots', str(shots), '--sampler', 'batch',
                           '--threads', '1', '--postselect-detectors']
                if name != 'legacy':
                    command += ['--cpu-backend', 'compiled', '--stream-id', str(args.stream_start + repeat)]
                before = ticks()
                start = time.perf_counter()
                proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
                wall = time.perf_counter() - start
                after = ticks()
                tag = f'd{distance}_{name}_{repeat}'
                (out / f'{tag}.stdout.txt').write_text(proc.stdout)
                (out / f'{tag}.stderr.txt').write_text(proc.stderr)
                if proc.returncode:
                    raise RuntimeError(f'{tag}: {proc.stderr}')
                fields = dict(line.split(maxsplit=1) for line in proc.stdout.splitlines() if len(line.split(maxsplit=1)) == 2)
                counts = tuple(int(fields[key]) for key in COUNTS)
                if counts[0] != shots or counts[0] != counts[1] + counts[2] or not 0 <= counts[3] <= counts[2]:
                    raise ValueError('invalid sampling counters')
                if name != 'legacy':
                    if fields.get('cpu_backend') != 'compiled' or fields.get('cpu_real_gauge') != '1':
                        raise ValueError('compiled real backend did not activate')
                    paired[name] = counts
                data['runs'].append({
                    'distance': distance, 'variant': name, 'repeat': repeat, 'warmup': repeat < 0,
                    'stream_id': 0 if name == 'legacy' else args.stream_start + repeat, 'command': command,
                    'circuit_sha256': digest(circuit), 'process_wall_s': wall, 'fields': fields,
                    'cpu_ticks_delta': {c: [b - a for a, b in zip(before[c], after[c])] for c in before},
                })
                (out / 'results.json').write_text(json.dumps(data, indent=2) + '\n')
                print(tag, fields['sample_shots_per_s'], 'shots/s', flush=True)
            if paired['before'] != paired['after']:
                raise ValueError('compiled before/after same-tape counters differ')
    for name, identity in binaries.items():
        if digest(Path(identity['source'])) != identity['sha256'] or digest(out / f'{name}.frozen') != identity['sha256']:
            raise ValueError('benchmark executable changed during measurement')
    data['summary'] = summarize(data['runs'])
    (out / 'results.json').write_text(json.dumps(data, indent=2) + '\n')
    if args.export is not None:
        curated = {k: v for k, v in data.items() if k not in ('runs', 'binaries')}
        curated['binaries'] = {
            name: {'sha256': identity['sha256'], 'file': f'{name}.frozen'} for name, identity in binaries.items()
        }
        curated['raw_results_sha256'] = digest(out / 'results.json')
        curated['archive_note'] = 'Raw stdout, command lines, build binaries and telemetry remain in the local experiment archive; not bundled.'
        curated['build'] = 'CMake Release, native, LTO; diagnostics off. Compiler and hardware recorded in the raw archive.'
        curated['runs'] = []
        for row in data['runs']:
            if row['warmup']:
                continue
            fields = row['fields']
            curated['runs'].append({
                **{k: row[k] for k in ('distance', 'variant', 'repeat', 'stream_id', 'circuit_sha256', 'cpu_ticks_delta')},
                **{k: int(fields[k]) for k in COUNTS},
                'sample_s': float(fields['sample_s_avg']),
                'attempted_shots_per_s': float(fields['sample_shots_per_s']),
            })
        with args.export.open('x') as target:
            json.dump(curated, target, indent=2)
            target.write('\n')
    print(json.dumps(data['summary'], indent=2))


if __name__ == '__main__':
    main()
