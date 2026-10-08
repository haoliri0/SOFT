#!/usr/bin/env python3
"""Validate the dated CPU comparison without rerunning a benchmark."""
import argparse
import json
import math
from pathlib import Path
import statistics

from check_results import ROOT, check_counts, close, digest, require


def check(data, check_current_sources=False):
    require(data['schema_version'] == 1 and data['release'] == 'symft_26_10_08', 'release/schema identity')
    require(data['date'] == '2026-10-08' and data['precision'] == 'FP64', 'date/precision')
    require(data['legacy_commit'] == 'e86c6a92525650744ce0dc2a65126e1ab951882d', 'old SymFT commit')
    runs = data['runs']
    require(data['repeats'] == 7 and len(runs) == 42, 'seven repeats, two distances, three versions')
    identities = {(r['distance'], r['variant'], r['repeat']) for r in runs}
    require(len(identities) == 42, 'duplicate benchmark repeat')
    for row in runs:
        distance = row['distance']
        require(distance in (3, 5), 'unsupported benchmark fixture')
        circuit = ROOT / f'benchmark/circuit/msc_d{distance}_inject_cultivate_p1e-3.stim'
        require(digest(circuit) == row['circuit_sha256'], 'fixture hash')
        check_counts({'shots': row['sampled_shots'], **{k: row[k] for k in ('discarded', 'accepted', 'logical_errors')}})
        require(row['sampled_shots'] == (12_000_000 if distance == 3 else 1_500_000), 'shots per repeat')
        require(math.isfinite(row['sample_s']) and row['sample_s'] > 0, 'positive sample timer')
        close(row['attempted_shots_per_s'], row['sampled_shots'] / row['sample_s'], 'rounded CLI throughput', rel=1e-5)
        require(row['stream_id'] == (0 if row['variant'] == 'legacy' else 261008100 + row['repeat']), 'stream schedule')
    for distance in (3, 5):
        for variant in ('legacy', 'before', 'after'):
            rows = [r for r in runs if r['distance'] == distance and r['variant'] == variant]
            require(sorted(r['repeat'] for r in rows) == list(range(7)), 'repeat coverage')
            rates = [r['attempted_shots_per_s'] for r in rows]
            actual = data['summary'][f'd{distance}_{variant}']
            require(actual['repeats'] == 7, 'summary repeat count')
            for field, value in [('median', statistics.median(rates)), ('min', min(rates)), ('max', max(rates))]:
                close(actual[field], value, field)
            for key in ('sampled_shots', 'discarded', 'accepted', 'logical_errors'):
                require(sum(r[key] for r in rows) == actual[key], 'counter sum')
        current = data['summary'][f'd{distance}_after']
        for baseline in ('legacy', 'before'):
            close(current[f'speedup_vs_{baseline}'], current['median'] / data['summary'][f'd{distance}_{baseline}']['median'], 'speedup')
        for repeat in range(7):
            paired = [r for r in runs if r['distance'] == distance and r['repeat'] == repeat and r['variant'] != 'legacy']
            require(len(paired) == 2, 'paired repeat missing')
            require(all(paired[0][k] == paired[1][k] for k in ('stream_id', 'sampled_shots', 'discarded', 'accepted', 'logical_errors')),
                    'compiled same-tape counters')
    if check_current_sources:
        for name, expected in data['source_sha256'].items():
            require(digest(ROOT / name) == expected, 'current source differs from measured build: ' + name)
    print('PASS October CPU: 42 runs, old/new ratios, fixture hashes, and paired counters')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-current-sources', action='store_true')
    args = parser.parse_args()
    check(json.loads((ROOT / 'benchmark/results/cpu_20261008.json').read_text()), args.check_current_sources)
