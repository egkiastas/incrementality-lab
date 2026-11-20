from __future__ import annotations

import warnings

import numpy as np
import pytest

from incrementality.uplift import (
    calibration_summary,
    decile_table,
    evaluate,
    qini_auc_score,
    qini_curve,
    uplift_at_k,
    uplift_auc_score,
    uplift_curve,
)


def test_qini_curve_toy_by_hand() -> None:
    # Sorted by score: (y, t) = (1,1), (0,0), (1,1), (0,0)
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 1, 0])
    s = np.array([4.0, 3.0, 2.0, 1.0])
    x, q = qini_curve(y, t, s)
    np.testing.assert_array_equal(x, [0, 1, 2, 3, 4])
    # Q(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k); no control responders -> Y_t(k)
    np.testing.assert_allclose(q, [0, 1, 1, 2, 2])
    x, u = uplift_curve(y, t, s)
    # U(k) = (Y_t/N_t - Y_c/N_c) * k: k=2: (1/1 - 0/1)*2 = 2; k=4: (2/2 - 0)*4 = 4
    np.testing.assert_allclose(u, [0, 1, 2, 3, 4])


def test_ties_are_grouped() -> None:
    y = np.array([1, 0, 1, 0])
    t = np.array([1, 0, 0, 1])
    x, _ = qini_curve(y, t, np.array([1.0, 1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(x, [0, 2, 4])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_scikit_uplift(seed: int) -> None:
    sklift = pytest.importorskip("sklift.metrics")
    rng = np.random.default_rng(seed)
    n = 3000
    t = (rng.random(n) < 0.8).astype(int)
    s = np.round(rng.normal(size=n), 1)  # many ties
    y = (rng.random(n) < 0.1 + 0.08 * t * (s > 0)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert qini_auc_score(y, t, s) == pytest.approx(sklift.qini_auc_score(y, s, t))
        assert uplift_auc_score(y, t, s) == pytest.approx(sklift.uplift_auc_score(y, s, t))
        assert uplift_at_k(y, t, s, 0.3) == pytest.approx(
            sklift.uplift_at_k(y, s, t, strategy="overall", k=0.3)
        )


def test_perfect_score_beats_random() -> None:
    rng = np.random.default_rng(3)
    n = 20_000
    t = (rng.random(n) < 0.5).astype(int)
    tau = rng.uniform(-0.1, 0.3, n)
    y = (rng.random(n) < 0.2 + t * tau).astype(int)
    good = qini_auc_score(y, t, tau)
    rand = qini_auc_score(y, t, rng.random(n))
    assert good > rand + 0.05
    assert abs(rand) < 0.05


def test_evaluate_bootstrap_ci_contains_point() -> None:
    rng = np.random.default_rng(4)
    n = 10_000
    t = (rng.random(n) < 0.85).astype(int)
    tau = rng.uniform(0, 0.2, n)
    y = (rng.random(n) < 0.1 + t * tau).astype(int)
    ev = evaluate(y, t, tau, [0.1, 0.3], 50, np.random.default_rng(0))
    assert ev["qini_ci"][0] <= ev["qini"] <= ev["qini_ci"][1]
    assert ev["uplift_at_k_ci"]["0.10"][0] <= ev["uplift_at_k"]["0.10"] <= ev["uplift_at_k_ci"]["0.10"][1]
    assert len(ev["deciles"]) == 10
    # Top decile by true tau has higher observed uplift than the bottom decile
    assert ev["deciles"][0]["observed"] > ev["deciles"][-1]["observed"]


def test_calibration_slope_one_for_calibrated_predictions() -> None:
    dec = [{"predicted": p, "observed": p, "se": 0.01} for p in np.linspace(0, 0.1, 10)]
    cal = calibration_summary(dec)
    assert cal["slope"] == pytest.approx(1.0)
    assert cal["intercept"] == pytest.approx(0.0, abs=1e-12)


def test_decile_table_sizes() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 1001)
    t = rng.integers(0, 2, 1001)
    rows = decile_table(y, t, rng.random(1001), 10)
    assert sum(r["n"] for r in rows) == 1001


def test_ties_do_not_inherit_storage_order() -> None:
    """Regression: the Criteo file is stored sorted by treatment. A block of tied scores
    must not put all control users of that block into one decile."""
    n = 20_000
    t = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)  # stored sorted by arm
    score = np.zeros(n)  # every prediction tied
    y = np.zeros(n, int)
    y[::7] = 1
    rows = decile_table(y, t, score, 10)
    shares = [r["n_treated"] / r["n"] for r in rows]
    assert min(shares) > 0.4 and max(shares) < 0.6
    ev = evaluate(y, t, score, [0.1], 5, np.random.default_rng(0))
    assert abs(ev["uplift_at_k"]["0.10"]) < 0.05  # no spurious uplift from storage order
