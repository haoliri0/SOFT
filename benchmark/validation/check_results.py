#!/usr/bin/env python3
"""Read-only checks of curated measurements and, optionally, original journals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from sampling_statistics import clopper_pearson, compare, self_test, wilson

ROOT = Path(__file__).resolve().parents[2]
COUNTS = ('shots', 'discarded', 'accepted', 'logical_errors')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label, rel=1e-10, absolute=1e-15):
    require(math.isclose(actual, expected, rel_tol=rel, abs_tol=absolute),
            f'{label}: {actual} != {expected}')


def check_counts(counts):
    require(all(type(counts[k]) is int and counts[k] >= 0 for k in COUNTS), 'invalid integer counts')
    require(counts['shots'] == counts['discarded'] + counts['accepted'], 'count conservation')
    require(counts['logical_errors'] <= counts['accepted'], 'errors exceed accepted')


def check_statistics(run):
    c = run['counts']
    check_counts(c)
    close(run['discard_rate'], c['discarded'] / c['shots'], 'discard denominator')
    close(run['error_rate_after_postselection'], c['logical_errors'] / c['accepted'], 'error denominator')
    for measured, expected in zip(run['discard_rate_wilson_95'], wilson(c['discarded'], c['shots'])):
        close(measured, expected, 'Wilson interval')
    for measured, expected in zip(run['error_rate_after_postselection_clopper_pearson_95'],
                                  clopper_pearson(c['logical_errors'], c['accepted'])):
        close(measured, expected, 'Clopper-Pearson interval', absolute=1e-18)


def check_curated(data):
    require(data['schema_version'] == 1, 'unsupported evidence schema')
    for circuit in data['circuits'].values():
        require(digest(ROOT / circuit['path']) == circuit['sha256'], 'circuit hash mismatch')
    for name in ('cpu_d5_50B', 'gpu_d5_50B', 'gpu_d3_1B'):
        check_statistics(data[name])
    cpu, gpu, d3 = (data[name] for name in ('cpu_d5_50B', 'gpu_d5_50B', 'gpu_d3_1B'))
    require(cpu['counts']['shots'] == gpu['counts']['shots'] == 50_000_000_000, '50B count')
    require(d3['counts']['shots'] == 1_000_000_000, 'd3 1B count')
    close(cpu['attempted_shots_per_s'], cpu['counts']['shots'] / cpu['sampling_wall_s'], 'CPU wall rate')
    close(gpu['attempted_shots_per_s'], gpu['counts']['shots'] / gpu['sample_s'], 'GPU sample rate')
    close(d3['attempted_shots_per_second_reported'],
          d3['counts']['shots'] / d3['sampling_time_seconds_reported'], 'rounded d3 rate', rel=1e-5)
    require(cpu['logical_cpu_count'] == 40 and cpu['physical_core_count'] == 20, 'socket labeling')
    require(len(cpu['config']['cpus']) == len(set(cpu['config']['cpus'])) == 40, 'CPU identity')
    require(cpu['paired_nonzero_error_replays']['added_to_production_counts'] is False, 'replay double-counting')
    comparison = compare(cpu['counts'], gpu['counts'])
    for key in ('discard_two_sided_normal_p', 'logical_errors_two_sided_fisher_p'):
        close(cpu['comparison_to_gpu_50B'][key], comparison[key], key)

    single = data['cpu_single_core']
    require(len(single['runs']) == 42, 'seven repeats x three variants x two distances')
    identities = {(r['distance'], r['variant'], r['stream_id']) for r in single['runs']}
    require(len(identities) == 42, 'duplicate CPU repeat')
    for key, aggregate in single['summary'].items():
        distance, variant = int(key[1]), key[3:]
        rows = [r for r in single['runs'] if r['distance'] == distance and r['variant'] == variant]
        require(len(rows) == 7, 'repeat count')
        require(sorted(r['stream_id'] for r in rows) == single['streams'], 'repeat stream coverage')
        for row in rows:
            check_counts(dict(shots=row['sampled_shots'], **{k: row[k] for k in COUNTS[1:]}))
            close(row['attempted_shots_per_s'], row['sampled_shots'] / row['sample_s'],
                  'rounded CPU repeat rate', rel=1e-5)
        for field in ('sampled_shots', 'discarded', 'accepted', 'logical_errors'):
            require(sum(r[field] for r in rows) == aggregate[field], 'CPU repeat sum: ' + field)
        rates = [r['attempted_shots_per_s'] for r in rows]
        for stat, value in [('median', statistics.median(rates)), ('min', min(rates)), ('max', max(rates))]:
            close(aggregate[stat], value, 'CPU ' + stat)
        close(aggregate['median_speedup_vs_legacy'],
              aggregate['median'] / single['summary'][f'd{distance}_legacy']['median'], 'paired speedup')
    for distance in (3, 5):
        variants = {v: sorted((r for r in single['runs'] if r['distance'] == distance and r['variant'] == v),
                              key=lambda r: r['stream_id']) for v in ('compiled', 'compiled_lto')}
        for a, b in zip(variants['compiled'], variants['compiled_lto']):
            require(all(a[k] == b[k] for k in ('sampled_shots', 'discarded', 'accepted', 'logical_errors')),
                    'LTO paired replay mismatch')
    baseline = data['gpu_d5_preoptimization']
    close(statistics.median(baseline['attempted_shots_per_s']),
          baseline['median_attempted_shots_per_s'], 'GPU baseline median')
    print('PASS curated counts, 42 CPU repeats, rates, intervals, comparisons, and fixture hashes')


def check_archive(data, archive):
    for source in data['source_archive']['files']:
        require(digest(archive / source['path']) == source['sha256'], 'source hash: ' + source['path'])
    for name, directory in [('cpu_d5_50B', 'cpu_socket40_d5_50B_20260928'),
                            ('gpu_d5_50B', 'longrun_50B_phase2_fp64')]:
        run = data[name]
        journal = archive / 'results' / directory / 'chunks.jsonl'
        require(digest(journal) == run['journal_sha256'], 'journal hash')
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        require(len({r['index'] for r in rows}) == len(rows), 'duplicate segment index')
        require(len({r['stream_id'] for r in rows}) == len(rows), 'duplicate stream')
        require(sorted(r['index'] for r in rows) == list(range(len(rows))), 'missing segment index')
        counts = {k: 0 for k in COUNTS}
        for row in rows:
            require(row['stream_id'] == run['config']['stream_start'] + row['index'], 'stream schedule')
            c = row['result'] if name.startswith('gpu') else row
            check_counts(c)
            for key in COUNTS:
                counts[key] += c[key]
        require(counts == run['counts'], 'raw journal sum mismatch')
        if name.startswith('cpu'):
            require(len(rows) == 10_000, 'CPU segment count')
            for row in rows:
                require(row['shots'] == 5_000_000, 'CPU segment shots')
                require(row['cpu'] in run['config']['cpus'] and row['actual_cpu'] == row['cpu'], 'worker affinity')
            require(len({r['session'] for r in rows}) == 1, 'CPU session count')
            span = max(r['finish_monotonic_s'] for r in rows) - min(r['start_monotonic_s'] for r in rows)
            close(run['sampling_wall_s'], span, 'raw CPU sampling span')
        else:
            require(len(rows) == 100, 'GPU segment count')
            close(run['sample_s'], math.fsum(r['result']['timing']['sample_s'] for r in rows), 'raw GPU timing sum')
    original = json.loads((archive / 'results/cpu_final_paired_20260928/results.json').read_text())
    rows = [r for r in original['runs'] if not r['warmup']]
    for exported, actual in zip(data['cpu_single_core']['runs'], rows):
        require(exported['distance'] == actual['distance'] and exported['variant'] == actual['variant'], 'CPU source identity')
        for field in ('sampled_shots', 'discarded', 'accepted', 'logical_errors'):
            require(exported[field] == int(actual['fields'][field]), 'CPU source count')
        close(exported['attempted_shots_per_s'], float(actual['fields']['sample_shots_per_s']), 'CPU source rate')
    d3 = data['gpu_d3_1B']
    raw_path = archive / 'results/d3_1B_phase2_fp64_20260923/results.json'
    require(digest(raw_path) == d3['source_results_sha256'], 'd3 raw result hash')
    d3rows = json.loads(raw_path.read_text())['records']
    require(len(d3rows) == 1, 'd3 is one run')
    require({k: d3rows[0][k] for k in COUNTS} == d3['counts'], 'd3 raw counts')
    print('PASS original source hashes, 50B journals, stream uniqueness, timing spans, and exported counters')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root', type=Path)
    args = parser.parse_args()
    self_test()
    data = json.loads((ROOT / 'benchmark/results/optimization_202609.json').read_text())
    check_curated(data)
    if args.experiment_root:
        check_archive(data, args.experiment_root)


if __name__ == '__main__':
    main()
