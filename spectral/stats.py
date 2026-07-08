"""
spectral.stats
==============

Non-parametric statistical helpers shared by tests A-D. These were duplicated
across the original notebooks; here they live in one place so every test uses
an identical, verified implementation.

All bootstrap routines accept an explicit ``rng`` (``np.random.default_rng``)
so results are reproducible. The seed *arithmetic* used by each test is kept in
the corresponding analysis module to match the thesis numbers exactly.
"""
from __future__ import annotations

import numpy as np


def finite_array(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def q25_q75_iqr(x):
    x = finite_array(x)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return float(q25), float(q75), float(q75 - q25)


def bootstrap_ci_median(x, B=5000, ci=0.95, rng=None):
    """Percentile bootstrap CI for the median of a 1-D sample."""
    x = finite_array(x)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    n = len(x)
    med = float(np.median(x))
    boot = np.empty(B, dtype=float)
    for b in range(B):
        boot[b] = np.median(x[rng.integers(0, n, size=n)])
    a = (1 - ci) / 2
    return med, float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def bootstrap_ci_mean(x, B=5000, ci=0.95, rng=None):
    """Percentile bootstrap CI for the mean of a 1-D sample.

    Used for rates/proportions by passing a 0/1 indicator array.
    """
    x = finite_array(x)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    n = len(x)
    mean_val = float(np.mean(x))
    boot = np.empty(B, dtype=float)
    for b in range(B):
        boot[b] = np.mean(x[rng.integers(0, n, size=n)])
    a = (1 - ci) / 2
    return mean_val, float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def bootstrap_ci_paired_delta(diffs, B=5000, ci=0.95, rng=None):
    """Percentile bootstrap CI for the median of paired differences."""
    diffs = finite_array(diffs)
    if len(diffs) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    n = len(diffs)
    d_med = float(np.median(diffs))
    boot = np.empty(B, dtype=float)
    for b in range(B):
        boot[b] = np.median(diffs[rng.integers(0, n, size=n)])
    a = (1 - ci) / 2
    return d_med, float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def bootstrap_ci_delta_median(x_ref, x_cmp, B=5000, ci=0.95, rng=None):
    """Bootstrap CI for median(x_cmp) - median(x_ref) (unpaired)."""
    x_ref = finite_array(x_ref)
    x_cmp = finite_array(x_cmp)
    if len(x_ref) == 0 or len(x_cmp) == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng()
    n0, n1 = len(x_ref), len(x_cmp)
    d_med = float(np.median(x_cmp) - np.median(x_ref))
    boot = np.empty(B, dtype=float)
    for b in range(B):
        s0 = x_ref[rng.integers(0, n0, size=n0)]
        s1 = x_cmp[rng.integers(0, n1, size=n1)]
        boot[b] = np.median(s1) - np.median(s0)
    a = (1 - ci) / 2
    return d_med, float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def cliffs_delta(x, y, rng=None, maxN=3000):
    """Cliff's delta effect size between samples x and y.

    Large samples are subsampled to ``maxN`` to keep the O(n*m) count tractable,
    matching the original implementation.
    """
    if rng is None:
        rng = np.random.default_rng()
    x = finite_array(x)
    y = finite_array(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    if len(x) > maxN:
        x = rng.choice(x, size=maxN, replace=False)
    if len(y) > maxN:
        y = rng.choice(y, size=maxN, replace=False)
    gt = lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def interpret_cliffs_delta(d) -> str:
    if not np.isfinite(d):
        return "undefined"
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"
