"""Covariate-adjusted ATE estimators: CUPED, CUPAC, AIPW and post-stratification.

CUPED subtracts theta' (Xbar_1 - Xbar_0). For pre-treatment X that term has mean zero under
randomization, so the estimate stays unbiased and its variance drops by R^2(Y ~ X). CUPAC uses
an out-of-fold, treatment-blind model prediction as the single covariate; the cross-fitting
keeps a unit's own outcome out of its covariate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier

from incrementality.stats import Estimate, diff_from_moments, diff_in_means, moments, normal_ci


def _proba(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(X))[:, 1]


def cuped_theta(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    Xc = X - X.mean(axis=0)
    yc = np.asarray(y, np.float64) - float(np.mean(y))
    return np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)


def cuped(y: np.ndarray, t: np.ndarray, X: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Pooled-OLS adjustment on X (1-D or 2-D)."""
    X = np.asarray(X, np.float64)
    if X.ndim == 1:
        X = X[:, None]
    theta = cuped_theta(y, X)
    y_adj = np.asarray(y, np.float64) - (X - X.mean(axis=0)) @ theta
    tb = np.asarray(t).astype(bool)
    est = diff_from_moments(moments(y_adj[tb]), moments(y_adj[~tb]), alpha)
    base = diff_in_means(y, t, alpha)
    return {
        "estimate": est,
        "baseline": base,
        "theta": theta.tolist(),
        "variance_reduction": 1.0 - est.se**2 / base.se**2,
    }


def cross_fitted_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    params: dict[str, Any],
    rng: np.random.Generator,
    train_rows_per_fold: int | None = None,
) -> np.ndarray:
    """Out-of-fold P(y=1 | X). Training rows per fold can be capped; every row gets a prediction."""
    n = X.shape[0]
    folds = rng.integers(0, n_folds, size=n)
    pred = np.empty(n)
    for k in range(n_folds):
        test = folds == k
        train = np.flatnonzero(~test)
        if train_rows_per_fold is not None and train.size > train_rows_per_fold:
            train = np.sort(rng.choice(train, train_rows_per_fold, replace=False))
        model = LGBMClassifier(**params, random_state=int(rng.integers(1 << 31)))
        model.fit(X[train], y[train])
        pred[test] = _proba(model, X[test])
    return pred


def dr_scores(
    X_train: np.ndarray,
    t_train: np.ndarray,
    y_train: np.ndarray,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    seed: int,
    propensity: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AIPW scores (phi1, phi0, e) on (X, t, y) with nuisance models fit on the training rows.

    mean(phi1) estimates E[Y(1)] and mean(phi0) E[Y(0)]. With `propensity=None`, e(x) is
    estimated (clipped to [0.01, 0.99]); otherwise the design value is used.
    """
    tb = np.asarray(t_train).astype(bool)
    m1 = LGBMClassifier(**params, random_state=seed).fit(X_train[tb], y_train[tb])
    m0 = LGBMClassifier(**params, random_state=seed + 1).fit(X_train[~tb], y_train[~tb])
    mu1, mu0 = _proba(m1, X), _proba(m0, X)
    if propensity is None:
        me = LGBMClassifier(**params, random_state=seed + 2).fit(X_train, tb.astype(np.int8))
        e = np.clip(_proba(me, X), 0.01, 0.99)
    else:
        e = np.full(X.shape[0], float(propensity))
    tf = np.asarray(t, np.float64)
    yf = np.asarray(y, np.float64)
    return mu1 + tf * (yf - mu1) / e, mu0 + (1 - tf) * (yf - mu0) / (1 - e), e


def summarize_dr(phi1: np.ndarray, phi0: np.ndarray, e: np.ndarray, alpha: float) -> dict[str, Any]:
    """ATE, adjusted arm means, and relative lift with its influence-function SE."""
    n = phi1.size
    m1, m0 = float(phi1.mean()), float(phi0.mean())
    d = phi1 - phi0
    infl = (phi1 - m1) / m0 - m1 * (phi0 - m0) / m0**2
    return {
        "ate": normal_ci(d.mean(), d.std(ddof=1) / np.sqrt(n), alpha),
        "mean_treated": m1,
        "mean_control": m0,
        "relative_lift": normal_ci(m1 / m0 - 1, infl.std(ddof=1) / np.sqrt(n), alpha),
        "propensity_min": float(e.min()),
        "propensity_max": float(e.max()),
        "propensity_sd": float(e.std()),
    }


def aipw(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    params: dict[str, Any],
    rng: np.random.Generator,
    n_folds: int = 2,
    propensity: float | None = None,
    alpha: float = 0.05,
    train_rows_per_fold: int | None = None,
) -> dict[str, Any]:
    """Cross-fitted AIPW. With an estimated propensity it stays consistent even if assignment
    depends on X; with the design propensity it is an interacted regression adjustment."""
    n = X.shape[0]
    folds = rng.integers(0, n_folds, size=n)
    phi1, phi0, e = np.empty(n), np.empty(n), np.empty(n)
    for k in range(n_folds):
        te = folds == k
        tr = np.flatnonzero(~te)
        if train_rows_per_fold is not None and tr.size > train_rows_per_fold:
            tr = np.sort(rng.choice(tr, train_rows_per_fold, replace=False))
        phi1[te], phi0[te], e[te] = dr_scores(
            X[tr], t[tr], y[tr], X[te], t[te], y[te], params, int(rng.integers(1 << 31)), propensity
        )
    return summarize_dr(phi1, phi0, e, alpha)


def post_stratified(
    y: np.ndarray, t: np.ndarray, g: np.ndarray, n_strata: int = 20, alpha: float = 0.05
) -> tuple[Estimate, list[dict[str, float]]]:
    """Size-weighted within-stratum differences, strata = quantiles of a pre-treatment score."""
    edges = np.quantile(g, np.linspace(0, 1, n_strata + 1))
    b = np.clip(np.searchsorted(edges, g, side="right") - 1, 0, n_strata - 1)
    tb = np.asarray(t).astype(bool)
    yf = np.asarray(y, np.float64)
    est, var, rows = 0.0, 0.0, []
    for k in range(n_strata):
        m = b == k
        y1, y0 = yf[m & tb], yf[m & ~tb]
        w = m.sum() / yf.size
        d = y1.mean() - y0.mean()
        est += w * d
        var += w**2 * (y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
        rows.append(
            {
                "stratum": k + 1,
                "mean_score": float(g[m].mean()),
                "treatment_share": float(tb[m].mean()),
                "control_rate": float(y0.mean()),
                "treated_rate": float(y1.mean()),
                "diff": float(d),
                "n": int(m.sum()),
            }
        )
    return normal_ci(est, np.sqrt(var), alpha), rows


def aa_check(
    y: np.ndarray, g: np.ndarray, share: float, n_sims: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Fake treatment splits of control users: plain and CUPED estimates should both centre on 0."""
    yf = np.asarray(y, np.float64)
    G = np.asarray(g, np.float64)
    y_adj = yf - cuped_theta(yf, G)[0] * (G - G.mean())
    out = {k: np.empty(n_sims) for k in ("est_plain", "p_plain", "est_cuped", "p_cuped")}
    for s in range(n_sims):
        fake = rng.random(yf.size) < share
        for tag, v in (("plain", yf), ("cuped", y_adj)):
            e = diff_from_moments(moments(v[fake]), moments(v[~fake]))
            out[f"est_{tag}"][s] = e.estimate
            out[f"p_{tag}"][s] = e.p_value
    return out
