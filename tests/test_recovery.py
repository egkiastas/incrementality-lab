"""Estimators recover known effects on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from incrementality.adjustment import aipw, cross_fitted_predictions, cuped
from incrementality.config import LGBM
from incrementality.stats import Estimate, diff_in_means, relative_lift, wald_iv
from incrementality.synthetic import SyntheticExperiment, generate
from incrementality.uplift import fit_predict, qini_auc_score

SMALL = LGBM | {"n_estimators": 60, "num_leaves": 15, "min_child_samples": 100}
BASE = LGBM | {"n_estimators": 150, "num_leaves": 31, "min_child_samples": 200}
CATE = LGBM | {"n_estimators": 150, "num_leaves": 15, "min_child_samples": 500, "learning_rate": 0.03}


def covers(est: Estimate, truth: float) -> bool:
    return est.ci_low <= truth <= est.ci_high


def test_simple_estimators_cover_truth(big_experiment: SyntheticExperiment) -> None:
    e = big_experiment
    assert covers(diff_in_means(e.visit, e.treatment), e.true_ate_visit)
    assert covers(diff_in_means(e.conversion, e.treatment), e.true_ate_conversion)
    assert covers(relative_lift(e.visit, e.treatment), e.true_ate_visit / e.p0.mean())
    assert covers(wald_iv(e.visit, e.treatment, e.exposure), e.true_late_visit)


def test_adjusted_estimators_cover_truth(big_experiment: SyntheticExperiment) -> None:
    e = big_experiment
    rng = np.random.default_rng(0)
    truth = e.true_ate_visit
    assert covers(cuped(e.visit, e.treatment, e.X)["estimate"], truth)
    cp = cuped(e.visit, e.treatment, cross_fitted_predictions(e.X, e.visit, 2, SMALL, rng))
    assert covers(cp["estimate"], truth)
    assert cp["variance_reduction"] > 0.05
    assert covers(aipw(e.visit, e.treatment, e.X, SMALL, rng, propensity=0.85)["ate"], truth)
    assert covers(aipw(e.visit, e.treatment, e.X, SMALL, rng)["ate"], truth)


def test_difference_in_means_coverage_is_close_to_nominal() -> None:
    hits = []
    for s in range(200):
        e = generate(5_000, seed=1000 + s)
        hits.append(covers(diff_in_means(e.visit, e.treatment), e.true_ate_visit))
    assert 0.90 <= np.mean(hits) <= 0.99


def test_cuped_is_unbiased_and_reduces_variance() -> None:
    est_c, est_d, truth = [], [], []
    for s in range(300):
        e = generate(8_000, seed=5000 + s)
        est_c.append(cuped(e.visit, e.treatment, e.X)["estimate"].estimate)
        est_d.append(diff_in_means(e.visit, e.treatment).estimate)
        truth.append(e.true_ate_visit)
    err = np.array(est_c) - np.array(truth)
    assert abs(err.mean()) < 3 * err.std(ddof=1) / np.sqrt(err.size)
    assert np.var(est_c) < np.var(est_d)


def test_cuped_has_nominal_size_under_the_null() -> None:
    rejections, estimates = [], []
    for s in range(300):
        e = generate(6_000, seed=9000 + s, effect_scale=0.0)
        r = cuped(e.visit, e.treatment, e.X)["estimate"]
        estimates.append(r.estimate)
        rejections.append(r.p_value < 0.05)
    est = np.array(estimates)
    assert abs(est.mean()) < 3 * est.std(ddof=1) / np.sqrt(est.size)
    assert np.mean(rejections) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 300)


def test_cross_fitted_cupac_is_unbiased() -> None:
    params = LGBM | {"n_estimators": 30, "num_leaves": 15, "min_child_samples": 50}
    rng = np.random.default_rng(0)
    errors = []
    for s in range(80):
        e = generate(8_000, seed=7000 + s)
        g = cross_fitted_predictions(e.X, e.visit, 2, params, rng)
        errors.append(cuped(e.visit, e.treatment, g)["estimate"].estimate - e.true_ate_visit)
    err = np.array(errors)
    assert abs(err.mean()) < 3 * err.std(ddof=1) / np.sqrt(err.size)


@pytest.mark.parametrize("name", ["s_learner", "t_learner", "x_learner", "dr_learner", "causal_forest"])
def test_learner_recovers_cate(
    name: str, big_experiment: SyntheticExperiment, holdout_experiment: SyntheticExperiment
) -> None:
    tr, te = big_experiment, holdout_experiment
    pred = fit_predict(name, tr.X, tr.treatment, tr.visit, te.X, BASE, CATE, 0.85, 0, 100, 100)
    assert np.corrcoef(pred, te.cate_visit)[0, 1] > 0.6
    assert qini_auc_score(te.visit, te.treatment, pred) > 0.0


def test_response_model_ranks_uplift_worse(
    big_experiment: SyntheticExperiment, holdout_experiment: SyntheticExperiment
) -> None:
    tr, te = big_experiment, holdout_experiment
    args = (tr.X, tr.treatment, tr.visit, te.X, BASE, CATE, 0.85, 0)
    r_resp = np.corrcoef(fit_predict("response_model", *args), te.cate_visit)[0, 1]
    r_dr = np.corrcoef(fit_predict("dr_learner", *args), te.cate_visit)[0, 1]
    assert r_dr > r_resp + 0.2
