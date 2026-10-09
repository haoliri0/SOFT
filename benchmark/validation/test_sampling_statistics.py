"""Independent formula checks for the archived rare-event uncertainty."""
import json
import math
from pathlib import Path
import unittest

from sampling_statistics import clopper_pearson, fisher_errors

EVIDENCE = Path(__file__).resolve().parents[1] / 'results/optimization_202609.json'


def direct_log_binomial_cdf(k, n, p):
    # Independent finite sum of log-combination terms, not the recurrence or
    # inversion used by sampling_statistics. Small observed k bounds the cost.
    terms = []
    for j in range(k + 1):
        log_combination = math.fsum(math.log(n - i) - math.log(i + 1) for i in range(j))
        terms.append(math.exp(log_combination + j * math.log(p) + (n - j) * math.log1p(-p)))
    return math.fsum(terms)


class StatisticsTests(unittest.TestCase):
    def test_archived_interval_endpoints_with_independent_sum(self):
        data = json.loads(EVIDENCE.read_text())
        for name in ('cpu_d5_50B', 'gpu_d5_50B', 'gpu_d3_1B'):
            run = data[name]
            k, n = run['counts']['logical_errors'], run['counts']['accepted']
            low, high = run['error_rate_after_postselection_clopper_pearson_95']
            with self.subTest(name=name):
                self.assertAlmostEqual(direct_log_binomial_cdf(k - 1, n, low), 0.975, delta=1e-10)
                self.assertAlmostEqual(direct_log_binomial_cdf(k, n, high), 0.025, delta=1e-10)

    def test_zero_error_upper_bound(self):
        for n in (1000, 7_197_794_221):
            low, high = clopper_pearson(0, n)
            self.assertEqual(low, 0)
            expected = -math.expm1(math.log(0.025) / n)
            self.assertTrue(math.isclose(high, expected, rel_tol=1e-12))

    def test_fisher_with_exact_integer_combinations(self):
        data = json.loads(EVIDENCE.read_text())
        a, b = data['cpu_d5_50B']['counts'], data['gpu_d5_50B']['counts']
        total_errors = a['logical_errors'] + b['logical_errors']
        weights = [math.comb(a['accepted'], k) * math.comb(b['accepted'], total_errors - k)
                   for k in range(total_errors + 1)]
        observed = weights[a['logical_errors']]
        exact = sum(w for w in weights if w <= observed) / sum(weights)
        got = fisher_errors(a['logical_errors'], a['accepted'], b['logical_errors'], b['accepted'])
        self.assertAlmostEqual(got, exact, delta=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
