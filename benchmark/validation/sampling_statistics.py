"""Rare-event binomial helpers preserved from the archived validation analysis.

Scoped to fewer than 1000 observed errors, not a general binomial CDF library.
The self-tests include independent direct combinatorial calculations.
"""
import math

Z95 = 1.959963984540054


def wilson(k, n):
    p = k / n
    denominator = 1 + Z95 * Z95 / n
    midpoint = (p + Z95 * Z95 / (2 * n)) / denominator
    halfwidth = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denominator
    return [max(0, midpoint - halfwidth), min(1, midpoint + halfwidth)]


def binomial_small_tail(k, n, p):
    """P[X <= k], stable for the rare-event regime used in this analysis."""
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    term = math.exp(n * math.log1p(-p))
    terms = [term]
    for j in range(k):
        term *= (n - j) / (j + 1) * p / (1 - p)
        terms.append(term)
    return min(1.0, math.fsum(terms))


def invert_tail(k, n, target):
    low, high = 0.0, min(1.0, (k + 100) / n)
    assert binomial_small_tail(k, n, high) < target
    for _ in range(120):
        mid = (low + high) / 2
        if binomial_small_tail(k, n, mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def clopper_pearson(k, n, alpha=0.05):
    assert 0 <= k <= n and k < 1000, 'helper is scoped to rare errors'
    low = 0.0 if k == 0 else invert_tail(k - 1, n, 1 - alpha / 2)
    high = 1.0 if k == n else invert_tail(k, n, alpha / 2)
    return [low, high]


def log_choose_small_k(n, k):
    return math.fsum(math.log(n - j) for j in range(k)) - math.lgamma(k + 1)


def fisher_errors(e1, n1, e2, n2):
    """Two-sided Fisher exact test on errors/non-errors among accepted shots."""
    errors = e1 + e2
    support = range(max(0, errors - n2), min(errors, n1) + 1)
    logs = {k: log_choose_small_k(n1, k) + log_choose_small_k(n2, errors - k)
            for k in support}
    maximum = max(logs.values())
    weights = {k: math.exp(value - maximum) for k, value in logs.items()}
    threshold = weights[e1] * (1 + 1e-12)
    return min(1.0, math.fsum(w for w in weights.values() if w <= threshold)
               / math.fsum(weights.values()))


def compare(current, baseline):
    n1, n2 = current['shots'], baseline['shots']
    d1, d2 = current['discarded'], baseline['discarded']
    pooled = (d1 + d2) / (n1 + n2)
    z = (d1 / n1 - d2 / n2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return {
        'baseline_counts': baseline,
        'baseline_discard_rate': d2 / n2,
        'baseline_error_rate_after_postselection': baseline['logical_errors'] / baseline['accepted'],
        'discard_difference_percentage_points': (d1 / n1 - d2 / n2) * 100,
        'discard_two_proportion_z': z,
        'discard_two_sided_normal_p': math.erfc(abs(z) / math.sqrt(2)),
        'logical_errors_two_sided_fisher_p': fisher_errors(
            current['logical_errors'], current['accepted'],
            baseline['logical_errors'], baseline['accepted']),
    }


def self_test():
    # Independent direct combinatorial CDF for small sample sizes.
    for n in (10, 30, 50):
        for k in (0, 1, 3, 8):
            for p in (0.001, 0.1, 0.5):
                exact = math.fsum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k+1))
                assert math.isclose(binomial_small_tail(k, n, p), exact, rel_tol=1e-12, abs_tol=1e-14)
    for n in (1000, 3_000_000_000):
        zero = clopper_pearson(0, n)[1]
        assert math.isclose(zero, -math.expm1(math.log(0.025) / n), rel_tol=1e-12)
        for k in (1, 8, 17):
            low, high = clopper_pearson(k, n)
            assert low < k / n < high
            assert abs(binomial_small_tail(k-1, n, low) - 0.975) < 1e-12
            assert abs(binomial_small_tail(k, n, high) - 0.025) < 1e-12
    assert math.isclose(fisher_errors(1, 10, 11, 14), 0.0027594561852200836, rel_tol=1e-12)
    assert fisher_errors(0, 100, 0, 200) == 1
    assert abs(sum(wilson(50, 100)) - 1) < 1e-15
    print('PASS independent statistics self-tests')
