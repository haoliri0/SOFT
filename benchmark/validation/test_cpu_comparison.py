"""Regression checks for the dated performance evidence validator."""
import copy
import json
import unittest

from check_cpu_comparison import ROOT, check


class CpuComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'benchmark/results/cpu_20261008.json').read_text())

    def test_bundled_comparison(self):
        check(self.data)

    def test_duplicate_repeat_rejected(self):
        bad = copy.deepcopy(self.data)
        bad['runs'][1] = bad['runs'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            check(bad)

    def test_wrong_speedup_rejected(self):
        bad = copy.deepcopy(self.data)
        bad['summary']['d5_after']['speedup_vs_legacy'] *= 2
        with self.assertRaisesRegex(ValueError, 'speedup'):
            check(bad)

    def test_counter_change_rejected(self):
        bad = copy.deepcopy(self.data)
        bad['runs'][0]['discarded'] += 1
        with self.assertRaisesRegex(ValueError, 'conservation'):
            check(bad)


if __name__ == '__main__':
    unittest.main()
