from __future__ import annotations

import numpy as np
import pytest

from incrementality.sequential import (
    cs_halfwidth,
    linear_looks,
    look_sums,
    msprt_log_lr,
    sequential_path,
    simulate_aa,
)


def test_msprt_lr_is_one_at_zero_variance_ratio_limit() -> None:
    # As tau^2 -> 0 the mixture collapses onto H0: Lambda -> 1
    assert msprt_log_lr(np.array([0.3]), np.array([0.01]), 1e-12)[0] == pytest.approx(0.0, abs=1e-9)


def test_cs_boundary_is_exactly_where_test_rejects() -> None:
    V, tau2, alpha = 1e-4, 4e-4, 0.05
    hw = cs_halfwidth(np.array([V]), tau2, alpha)[0]
    llr = msprt_log_lr(np.array([hw]), np.array([V]), tau2)[0]
    assert llr == pytest.approx(np.log(1 / alpha))


def test_cs_wider_than_fixed_ci() -> None:
    V = np.array([1e-4, 1e-6])
    hw = cs_halfwidth(V, 1e-4, 0.05)
    assert np.all(hw > 1.96 * np.sqrt(V))


def test_look_sums_cumulative() -> None:
    y = np.array([1, 0, 1, 1, 0, 1], float)
    t = np.array([1, 0, 1, 0, 1, 0])
    s = look_sums(y, t, np.array([2, 4, 6]))
    np.testing.assert_allclose(s["n1"], [1, 2, 3])
    np.testing.assert_allclose(s["s1"], [1, 2, 2])
    np.testing.assert_allclose(s["s0"], [0, 1, 2])
    np.testing.assert_allclose(s["n0"], [1, 2, 3])


def test_sequential_path_final_matches_full_sample() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.1).astype(float)
    t = (rng.random(10_000) < 0.85).astype(int)
    path = sequential_path(y, t, np.array([1000, 5000, 10_000]), 0.05, 1e-4)
    assert path["diff"][-1] == pytest.approx(y[t == 1].mean() - y[t == 0].mean())
    assert np.all(path["cs_low"] <= path["ci_low"])


def test_msprt_controls_fpr_under_continuous_peeking() -> None:
    rng = np.random.default_rng(0)
    pool = (rng.random(400_000) < 0.05).astype(np.int8)
    looks = linear_looks(20_000, 50, 500)
    res = simulate_aa(pool, 400, 20_000, 0.85, looks, 0.05, (0.1 * 0.05) ** 2, rng)
    mc_se = np.sqrt(0.05 * 0.95 / 400)
    assert res["fpr_msprt"] <= 0.05 + 2.5 * mc_se, res["fpr_msprt"]
    assert abs(res["fpr_fixed_horizon"] - 0.05) <= 3 * mc_se, res["fpr_fixed_horizon"]
    assert res["fpr_naive_peeking"] > 0.12, res["fpr_naive_peeking"]  # inflation is real


def test_confidence_sequence_covers_truth_uniformly_over_time() -> None:
    """Under a real effect, the CS covers the true effect at *every* look ~>= 95% of runs."""
    rng = np.random.default_rng(1)
    looks = linear_looks(20_000, 40, 500)
    p0, effect = 0.05, 0.01
    covered = []
    for _ in range(200):
        t = (rng.random(20_000) < 0.85).astype(np.int8)
        y = (rng.random(20_000) < p0 + effect * t).astype(np.int8)
        path = sequential_path(y, t, looks, 0.05, (0.1 * p0) ** 2)
        ok = np.isfinite(path["cs_low"])
        covered.append(bool(np.all((path["cs_low"][ok] <= effect) & (path["cs_high"][ok] >= effect))))
    assert np.mean(covered) >= 0.93
