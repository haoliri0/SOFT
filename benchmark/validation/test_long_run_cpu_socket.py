#!/usr/bin/env python3
"""Unit checks plus optional Linux worker integration; no archived runs required."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from long_run_cpu_socket import expected_shots, parse_cpus, read_journal, sha256, summary, validate_row


def config():
    return dict(target_shots=100257, segment_shots=50000, stream_start=48000000, cpus=[0, 1])


def row(index=0):
    shots = expected_shots(index, config())
    return dict(index=index, stream_id=48000000 + index, shots=shots,
                discarded=shots - 100, accepted=100, logical_errors=1,
                cpu=0, actual_cpu=0, session='test', start_monotonic_s=10.0 + index,
                finish_monotonic_s=11.0 + index, call_wall_s=1.0, process_cpu_s=0.99)


class HarnessUnitTests(unittest.TestCase):
    def test_cpu_ranges(self):
        self.assertEqual(parse_cpus('0-2,40,43-44'), [0, 1, 2, 40, 43, 44])
        for value in ('0,0', '-1', '3-1', '', '1-2-3'):
            with self.assertRaises(ValueError):
                parse_cpus(value)

    def test_tail_and_out_of_order_completion(self):
        rows = [row(i) for i in (2, 0, 1)]
        seen = set()
        for item in rows:
            validate_row(item, config(), seen)
        self.assertEqual(seen, {0, 1, 2})
        self.assertEqual(rows[0]['shots'], 257)
        self.assertEqual(summary(rows, config())['shots'], 100257)

    def test_duplicate_and_invalid_rows(self):
        item = row()
        with self.assertRaisesRegex(AssertionError, 'duplicate'):
            validate_row(item, config(), {0})
        for field in ('shots', 'discarded', 'accepted', 'stream_id', 'actual_cpu'):
            bad = copy.deepcopy(item)
            bad[field] += 1
            with self.assertRaises(AssertionError):
                validate_row(bad, config(), set())
        bad = dict(item, logical_errors=101)
        with self.assertRaises(AssertionError):
            validate_row(bad, config(), set())

    def test_empty_summary(self):
        result = summary([], config())
        self.assertIsNone(result['discard_rate'])
        self.assertIsNone(result['error_rate_after_postselection'])
        self.assertFalse(result['complete'])

    def test_corrupt_journal_rejected(self):
        with tempfile.TemporaryDirectory(prefix='soft-journal-test-') as tmp:
            path = Path(tmp) / 'chunks.jsonl'
            path.write_text('{"index":')
            with self.assertRaises(json.JSONDecodeError):
                read_journal(path, config())


@unittest.skipUnless(os.environ.get('SYMFT_TEST_WORKER'), 'set SYMFT_TEST_WORKER for actual-worker checks')
class WorkerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='soft-worker-test-')
        cls.addClassCleanup(cls.directory.cleanup)
        cpus = sorted(os.sched_getaffinity(0))[:2]
        cls.command = [sys.executable, str(Path(__file__).with_name('long_run_cpu_socket.py')),
                       '--worker', str(Path(os.environ['SYMFT_TEST_WORKER']).resolve()),
                       '--output-root', cls.directory.name, '--label', 'smoke',
                       '--cpus', ','.join(map(str, cpus)), '--shots', '100257',
                       '--segment-shots', '50000', '--warmup-shots', '2049',
                       '--stream-start', '48000000', '--warmup-stream-start', '47900000']
        result = subprocess.run(cls.command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.output = Path(cls.directory.name) / 'smoke'
        cls.config = json.loads((cls.output / 'metadata.json').read_text())['config']
        cls.rows, _ = read_journal(cls.output / 'chunks.jsonl', cls.config)
        cls.command.append('--resume')

    def test_same_stream_reference_and_tail(self):
        item = next(r for r in self.rows if r['index'] == 0)
        self.assertEqual([item[k] for k in ('shots', 'discarded', 'accepted', 'logical_errors')],
                         [50000, 42926, 7074, 0])
        self.assertEqual(next(r['shots'] for r in self.rows if r['index'] == 2), 257)

    def test_completed_resume_does_not_resample(self):
        before = sha256(self.output / 'chunks.jsonl')
        result = subprocess.run(self.command, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)['complete'])
        self.assertEqual(before, sha256(self.output / 'chunks.jsonl'))

    def test_exclusive_lock(self):
        with (self.output / 'run.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(self.command, capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('BlockingIOError', result.stderr)

    def test_identity_mismatch(self):
        command = self.command.copy()
        command[command.index('--shots') + 1] = '100258'
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('identity mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
