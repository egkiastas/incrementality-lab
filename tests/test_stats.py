from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from incrementality.stats import (
    bootstrap_binary,
    bootstrap_rows,
    diff_in_means,
    mde,
    moments_from_sums,
    percentile_ci,
    relative_lift,
    required_n,
    smd,
    smd_table,
    srm_test,
    wald_iv,
)


def test_diff_in_means_toy() -> None:
    y = np.array([1, 0, 1, 1, 0, 0, 0, 1], float)
    t = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    est = diff_in_means(y, t)
    assert est.estimate == pytest.approx(0.75 - 0.25)
    # s^2 = p(1-p) * n/(n-1) = 0.1875 * 4/3 = 0.25 in each arm
    assert est.se == pytest.approx(np.sqrt(0.25 / 4 + 0.25 / 4))
    assert est.ci_low < est.estimate < est.ci_high


def test_diff_in_means_matches_welch_ttest() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000) + 0.1 * (t := rng.integers(0, 2, 5000))
    est = diff_in_means(y, t)
    res = sps.ttest_ind(y[t == 1], y[t == 0], equal_var=False)
    assert est.p_value == pytest.approx(res.pvalue, rel=1e-2)


def test_relative_lift_delta_method_matches_simulation() -> None:
    rng = np.random.default_rng(1)
    p1, p0, n1, n0 = 0.06, 0.05, 20_000, 20_000
    sims = rng.binomial(n1, p1, 20_000) / n1 / (rng.binomial(n0, p0, 20_000) / n0) - 1
    y = np.r_[
        np.ones(int(p1 * n1)),
        np.zeros(n1 - int(p1 * n1)),
        np.ones(int(p0 * n0)),
        np.zeros(n0 - int(p0 * n0)),
    ]
    t = np.r_[np.ones(n1), np.zeros(n0)]
    est = relative_lift(y, t)
    assert est.estimate == pytest.approx(0.2)
    assert est.se == pytest.approx(sims.std(), rel=0.05)


def test_binary_bootstrap_equals_row_bootstrap_in_distribution() -> None:
    rng = np.random.default_rng(2)
    t = (rng.random(4000) < 0.7).astype(int)
    y = (rng.random(4000) < 0.1 + 0.05 * t).astype(int)
    d_rows, _l_rows = bootstrap_rows(y, t, 3000, np.random.default_rng(3))
    k1, n1, k0, n0 = y[t == 1].sum(), (t == 1).sum(), y[t == 0].sum(), (t == 0).sum()
    d_bin, _l_bin = bootstrap_binary(k1, n1, k0, n0, 3000, np.random.default_rng(4))
    assert d_rows.mean() == pytest.approx(d_bin.mean(), abs=1e-3)
    assert d_rows.std() == pytest.approx(d_bin.std(), rel=0.07)
    assert sps.ks_2samp(d_rows, d_bin).pvalue > 0.01
    lo, hi = percentile_ci(d_bin)
    assert lo < y[t == 1].mean() - y[t == 0].mean() < hi


def test_srm_detects_mismatch_and_accepts_match() -> None:
    assert srm_test(8500, 1500, 0.85)["p_value"] == pytest.approx(1.0)
    bad = srm_test(8300, 1700, 0.85)
    # chi2 = 200^2/8500 + 200^2/1500
    assert bad["chi2"] == pytest.approx(200**2 / 8500 + 200**2 / 1500)
    assert bad["p_value"] < 1e-6


def test_smd() -> None:
    assert smd(np.array([1.0, 3.0]), np.array([0.0, 2.0])) == pytest.approx(1 / np.sqrt(2))
    X = np.column_stack([np.r_[np.ones(5), np.zeros(5)], np.ones(10)])
    t = np.r_[np.ones(5), np.zeros(5)]
    table = smd_table(X, t)
    assert table[1] == 0.0  # constant feature
    assert np.isinf(table[0])  # perfectly separated feature


def test_wald_iv_one_sided_noncompliance() -> None:
    # 1000 treated: 200 exposed, outcome = 1 for all exposed and 10% of others; control 10%.
    z = np.r_[np.ones(1000), np.zeros(1000)]
    d = np.r_[np.ones(200), np.zeros(800), np.zeros(1000)]
    y = np.r_[np.ones(200), np.ones(80), np.zeros(720), np.ones(100), np.zeros(900)]
    est = wald_iv(y, z, d)
    # ITT_y = 0.28 - 0.10 = 0.18; ITT_d = 0.2  -> 0.9
    assert est.estimate == pytest.approx(0.9)
    assert est.se > 0


def test_mde_and_required_n_are_inverse() -> None:
    s2 = 0.04 * 0.96
    n = 1_000_000
    m = mde(s2, n, 0.85)
    assert required_n(s2, m, 0.85) == pytest.approx(n, rel=1e-9)
    # 85/15 needs 1/(0.85*0.15) / 4 = 1.96x the users of a 50/50 split
    assert required_n(s2, m, 0.85) / required_n(s2, m, 0.5) == pytest.approx(0.25 / (0.85 * 0.15))


def test_moments_from_sums_matches_numpy() -> None:
    y = np.array([1.0, 2.0, 4.0, 7.0])
    mean, var, n = moments_from_sums(4, y.sum(), (y**2).sum())
    assert mean == pytest.approx(y.mean())
    assert var == pytest.approx(y.var(ddof=1))
    assert n == 4
