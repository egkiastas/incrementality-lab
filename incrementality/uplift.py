"""CATE learners and uplift evaluation.

Learners: S, T, X and DR meta-learners on LightGBM, plus EconML's CausalForestDML. All of them use
the known RCT propensity rather than estimating it. Each returns predicted uplift for `X_eval`.

Evaluation follows scikit-uplift's definitions (tests check exact equality) but takes frequency
weights, so a Poisson bootstrap needs only one sort:
  Qini Q(k) = Y_t(k) - Y_c(k) * N_t(k)/N_c(k);  uplift U(k) = (Y_t/N_t - Y_c/N_c)(k) * k;
  normalised score = (area - random) / (perfect - random).
Tied scores form one step of the curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy import stats as sps

Params = dict[str, Any]


def _clf(params: Params, seed: int) -> LGBMClassifier:
    return LGBMClassifier(**params, random_state=seed)


def _reg(params: Params, seed: int) -> LGBMRegressor:
    return LGBMRegressor(**params, random_state=seed)


def _p1(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(model.predict_proba(X))[:, 1], dtype=np.float64)


def s_learner(
    X: np.ndarray, t: np.ndarray, y: np.ndarray, X_eval: np.ndarray, params: Params, seed: int
) -> np.ndarray:
    model = _clf(params, seed).fit(np.column_stack([X, t]), y)
    ones = np.ones((X_eval.shape[0], 1), X_eval.dtype)
    return _p1(model, np.hstack([X_eval, ones])) - _p1(model, np.hstack([X_eval, 0 * ones]))


def _arm_models(
    X: np.ndarray, t: np.ndarray, y: np.ndarray, params: Params, seed: int
) -> tuple[LGBMClassifier, LGBMClassifier]:
    tb = t.astype(bool)
    return _clf(params, seed).fit(X[tb], y[tb]), _clf(params, seed + 1).fit(X[~tb], y[~tb])


def t_learner(
    X: np.ndarray, t: np.ndarray, y: np.ndarray, X_eval: np.ndarray, params: Params, seed: int
) -> np.ndarray:
    m1, m0 = _arm_models(X, t, y, params, seed)
    return _p1(m1, X_eval) - _p1(m0, X_eval)


def x_learner(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    params: Params,
    cate_params: Params,
    propensity: float,
    seed: int,
) -> np.ndarray:
    """Impute individual effects with the other arm's model, regress them on X, and weight the
    control-side model by e (0.85): it borrows the well-estimated treated-arm model."""
    tb = t.astype(bool)
    m1, m0 = _arm_models(X, t, y, params, seed)
    mu0, mu1 = _p1(m0, X), _p1(m1, X)
    yf = y.astype(np.float64)
    tau1 = _reg(cate_params, seed + 2).fit(X[tb], yf[tb] - mu0[tb])
    tau0 = _reg(cate_params, seed + 3).fit(X[~tb], mu1[~tb] - yf[~tb])
    t1 = np.asarray(tau1.predict(X_eval), np.float64)
    t0 = np.asarray(tau0.predict(X_eval), np.float64)
    return propensity * t0 + (1 - propensity) * t1


def dr_learner(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    params: Params,
    cate_params: Params,
    propensity: float,
    seed: int,
    n_folds: int = 2,
) -> np.ndarray:
    """Regress the cross-fitted AIPW pseudo-outcome on X."""
    folds = np.random.default_rng(seed).integers(0, n_folds, size=X.shape[0])
    mu0, mu1 = np.empty(X.shape[0]), np.empty(X.shape[0])
    tb = t.astype(bool)
    for k in range(n_folds):
        tr, te = folds != k, folds == k
        mu1[te] = _p1(_clf(params, seed + 10 * k).fit(X[tr & tb], y[tr & tb]), X[te])
        mu0[te] = _p1(_clf(params, seed + 10 * k + 1).fit(X[tr & ~tb], y[tr & ~tb]), X[te])
    yf, tf, e = y.astype(np.float64), t.astype(np.float64), propensity
    phi = mu1 - mu0 + tf * (yf - mu1) / e - (1 - tf) * (yf - mu0) / (1 - e)
    final = _reg(cate_params, seed + 5).fit(X, phi)
    return np.asarray(final.predict(X_eval), np.float64)


def causal_forest(
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    params: Params,
    seed: int,
    n_estimators: int = 200,
    min_samples_leaf: int = 200,
) -> np.ndarray:
    from econml.dml import CausalForestDML
    from sklearn.dummy import DummyClassifier

    model = CausalForestDML(
        model_y=_clf(params, seed),
        model_t=DummyClassifier(strategy="prior"),
        discrete_outcome=True,
        discrete_treatment=True,
        cv=2,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_samples=0.45,
        honest=True,
        inference=False,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(y.astype(np.int64), t.astype(np.int64), X=X.astype(np.float64))
    return np.asarray(model.effect(X_eval.astype(np.float64)), np.float64).ravel()


def response_model(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, params: Params, seed: int) -> np.ndarray:
    """P(Y=1 | X), ignoring treatment: the usual "likely to respond" targeting score."""
    return _p1(_clf(params, seed).fit(X, y), X_eval)


def fit_predict(
    name: str,
    X: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    X_eval: np.ndarray,
    params: Params,
    cate_params: Params,
    propensity: float,
    seed: int,
    forest_trees: int = 200,
    forest_min_leaf: int = 200,
) -> np.ndarray:
    if name == "s_learner":
        return s_learner(X, t, y, X_eval, params, seed)
    if name == "t_learner":
        return t_learner(X, t, y, X_eval, params, seed)
    if name == "x_learner":
        return x_learner(X, t, y, X_eval, params, cate_params, propensity, seed)
    if name == "dr_learner":
        return dr_learner(X, t, y, X_eval, params, cate_params, propensity, seed)
    if name == "causal_forest":
        return causal_forest(X, t, y, X_eval, params, seed, forest_trees, forest_min_leaf)
    if name == "response_model":
        return response_model(X, y, X_eval, params, seed)
    raise ValueError(f"unknown learner {name!r}")


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
def random_tiebreak_order(score: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices sorting ``score`` descending, with ties broken uniformly at random."""
    tie = np.random.default_rng(seed).random(np.asarray(score).size)
    return np.lexsort((tie, -np.asarray(score, np.float64)))


Sorted = tuple[np.ndarray, np.ndarray, np.ndarray]  # y, t, last index of each tied block


def _prepare(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> Sorted:
    # same ordering as scikit-uplift: stable ascending sort, reversed
    order = np.argsort(np.asarray(score, np.float64), kind="mergesort")[::-1]
    s = np.asarray(score, np.float64)[order]
    thresholds = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]
    return np.asarray(y, np.float64)[order], np.asarray(t, np.float64)[order], thresholds


def _counts(
    y: np.ndarray, t: np.ndarray, w: np.ndarray | None, thr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative (N_all, N_t, Y_t, Y_c) at each threshold, prefixed with a zero point."""
    if w is None:
        w = np.ones_like(y)
    wt = w * t
    n_all = np.cumsum(w)[thr]
    n_t = np.cumsum(wt)[thr]
    y_t = np.cumsum(wt * y)[thr]
    y_c = np.cumsum((w - wt) * y)[thr]
    z = np.zeros(1)
    return (np.r_[z, n_all], np.r_[z, n_t], np.r_[z, y_t], np.r_[z, y_c])


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def _qini_values(n_all: np.ndarray, n_t: np.ndarray, y_t: np.ndarray, y_c: np.ndarray) -> np.ndarray:
    return y_t - y_c * _safe_div(n_t, n_all - n_t)


def _uplift_values(n_all: np.ndarray, n_t: np.ndarray, y_t: np.ndarray, y_c: np.ndarray) -> np.ndarray:
    return (_safe_div(y_t, n_t) - _safe_div(y_c, n_all - n_t)) * n_all


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _group_weights(y: np.ndarray, t: np.ndarray, w: np.ndarray | None) -> dict[str, float]:
    if w is None:
        w = np.ones_like(y)
    return {
        "tr": float(np.sum(w * (y == 1) * (t == 1))),
        "tn": float(np.sum(w * (y == 0) * (t == 1))),
        "cn": float(np.sum(w * (y == 0) * (t == 0))),
        "cr": float(np.sum(w * (y == 1) * (t == 0))),
    }


def _perfect_area(g: dict[str, float], kind: str) -> tuple[float, float, float]:
    """(area, x_end, y_end) of the oracle curve built from 4 aggregated cells (exact)."""
    # rows: (y, t, weight); scores chosen exactly as scikit-uplift does.
    y = np.array([1.0, 0.0, 0.0, 1.0])
    t = np.array([1.0, 1.0, 0.0, 0.0])
    w = np.array([g["tr"], g["tn"], g["cn"], g["cr"]])
    if kind == "qini":
        score = y * t - y * (1 - t)  # [1, 0, 0, -1]
    else:
        summand = y if g["cr"] > g["tn"] else t
        score = 2 * (y == t) + summand
    keep = w > 0
    y, t, w, score = y[keep], t[keep], w[keep], score[keep]
    order = np.argsort(-score, kind="mergesort")
    y, t, w, score = y[order], t[order], w[order], score[order]
    thr = np.r_[np.flatnonzero(np.diff(score)), score.size - 1]
    n_all, n_t, y_t, y_c = _counts(y, t, w, thr)
    vals = _qini_values(n_all, n_t, y_t, y_c) if kind == "qini" else _uplift_values(n_all, n_t, y_t, y_c)
    return _trapz(vals, n_all), float(n_all[-1]), float(vals[-1])


def _normalized_auc(n_all: np.ndarray, vals: np.ndarray, perfect: tuple[float, float, float]) -> float:
    area_p, x_end, y_end = perfect
    base = x_end * y_end / 2
    return (_trapz(vals, n_all) - base) / (area_p - base)


def _scores_from_sorted(srt: Sorted, w: np.ndarray | None) -> tuple[float, float]:
    y, t, thr = srt
    n_all, n_t, y_t, y_c = _counts(y, t, w, thr)
    g = _group_weights(y, t, w)
    qini = _normalized_auc(n_all, _qini_values(n_all, n_t, y_t, y_c), _perfect_area(g, "qini"))
    auuc = _normalized_auc(n_all, _uplift_values(n_all, n_t, y_t, y_c), _perfect_area(g, "uplift"))
    return qini, auuc


def qini_auc_score(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Normalized Qini coefficient (scikit-uplift ``qini_auc_score``, negative_effect=True)."""
    return _scores_from_sorted(_prepare(y, t, score), None)[0]


def uplift_auc_score(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> float:
    """Normalized area under the uplift curve (scikit-uplift ``uplift_auc_score``)."""
    return _scores_from_sorted(_prepare(y, t, score), None)[1]


def qini_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, ts, thr = _prepare(y, t, score)
    n_all, n_t, y_t, y_c = _counts(ys, ts, None, thr)
    return n_all, _qini_values(n_all, n_t, y_t, y_c)


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, ts, thr = _prepare(y, t, score)
    n_all, n_t, y_t, y_c = _counts(ys, ts, None, thr)
    return n_all, _uplift_values(n_all, n_t, y_t, y_c)


def _uplift_at_k_sorted(srt: Sorted, k: float, w: np.ndarray | None) -> float:
    """Observed uplift among the top-k fraction (by sorted position, as scikit-uplift)."""
    m = int(k * srt[0].size)
    y, t = srt[0][:m], srt[1][:m]
    ww = np.ones_like(y) if w is None else w[:m]
    wt = ww * t
    wc = ww - wt
    return float(np.sum(wt * y) / np.sum(wt) - np.sum(wc * y) / np.sum(wc))


def uplift_at_k(y: np.ndarray, t: np.ndarray, score: np.ndarray, k: float) -> float:
    return _uplift_at_k_sorted(_prepare(y, t, score), k, None)


def decile_table(
    y: np.ndarray, t: np.ndarray, score: np.ndarray, n_bins: int = 10, alpha: float = 0.05
) -> list[dict[str, float]]:
    """Observed vs predicted uplift by score bin (bin 1 = highest predicted uplift).

    Ties in the score are broken *randomly* (fixed seed). Breaking them by row position is
    dangerous: the Criteo file is stored in blocks sorted by treatment, so a large block of
    tied predictions would put all its control users into one bin.
    """
    order = random_tiebreak_order(score)
    bins = np.array_split(order, n_bins)
    z = float(sps.norm.ppf(1 - alpha / 2))
    rows = []
    for b, idx in enumerate(bins, start=1):
        tb = t[idx].astype(bool)
        y1, y0 = y[idx][tb].astype(float), y[idx][~tb].astype(float)
        m1, m0 = y1.mean(), y0.mean()
        se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
        rows.append(
            {
                "bin": b,
                "n": int(idx.size),
                "n_treated": int(tb.sum()),
                "n_control": int((~tb).sum()),
                "predicted": float(np.mean(score[idx])),
                "observed": float(m1 - m0),
                "se": float(se),
                "ci_low": float(m1 - m0 - z * se),
                "ci_high": float(m1 - m0 + z * se),
                "treated_rate": float(m1),
                "control_rate": float(m0),
            }
        )
    return rows


def calibration_summary(deciles: list[dict[str, float]]) -> dict[str, float]:
    """Weighted least squares of observed on predicted decile uplift (ideal: slope 1, int 0).

    Robust to degenerate bins: bins with a non-finite SE are dropped, zero SEs are floored at
    the smallest positive SE, and a constant prediction yields slope = NaN instead of an error.
    """
    nan = float("nan")
    p = np.array([d["predicted"] for d in deciles], np.float64)
    o = np.array([d["observed"] for d in deciles], np.float64)
    se = np.array([d["se"] for d in deciles], np.float64)
    ok = np.isfinite(p) & np.isfinite(o) & np.isfinite(se)
    p, o, se = p[ok], o[ok], se[ok]
    keys = [
        "intercept",
        "slope",
        "weighted_mae",
        "spearman_pred_obs",
        "chi2_vs_identity",
        "rmse_resid",
    ]
    if p.size < 3 or not np.any(se > 0):
        return dict.fromkeys(keys, nan)
    se = np.maximum(se, se[se > 0].min())
    w = 1 / se**2
    sw = np.sqrt(w)
    A = np.column_stack([np.ones_like(p), p])
    if np.ptp(p) == 0:
        coef = np.array([np.sum(w * o) / np.sum(w), nan])
        resid = o - coef[0]
    else:
        coef = np.linalg.lstsq(A * sw[:, None], o * sw, rcond=None)[0]
        resid = o - A @ coef
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "weighted_mae": float(np.sum(w * np.abs(o - p)) / np.sum(w)),
        "spearman_pred_obs": float(sps.spearmanr(p, o).statistic) if np.ptp(p) > 0 else nan,
        "chi2_vs_identity": float(np.sum(w * (o - p) ** 2)),
        "rmse_resid": float(np.sqrt(np.mean(resid**2))),
    }


def evaluate(
    y: np.ndarray,
    t: np.ndarray,
    score: np.ndarray,
    budgets: list[float],
    n_boot: int,
    rng: np.random.Generator,
    n_bins: int = 10,
    alpha: float = 0.05,
    curve_points: int = 201,
) -> dict[str, Any]:
    """Point estimates + Poisson-bootstrap percentile CIs for an uplift score on test data.

    The rows are shuffled once (fixed seed) before sorting so that uplift@k does not depend on
    the storage order of tied scores (Qini/AUUC group ties and are unaffected either way).
    """
    perm = np.random.default_rng(12345).permutation(np.asarray(y).size)
    y, t, score = np.asarray(y)[perm], np.asarray(t)[perm], np.asarray(score)[perm]
    srt = _prepare(y, t, score)
    qini, auuc = _scores_from_sorted(srt, None)
    uk = {f"{k:.2f}": _uplift_at_k_sorted(srt, k, None) for k in budgets}
    boot_q = np.empty(n_boot)
    boot_a = np.empty(n_boot)
    boot_k = {key: np.empty(n_boot) for key in uk}
    for b in range(n_boot):
        w = rng.poisson(1.0, size=y.size).astype(np.float64)
        boot_q[b], boot_a[b] = _scores_from_sorted(srt, w)
        for k in budgets:
            boot_k[f"{k:.2f}"][b] = _uplift_at_k_sorted(srt, k, w)

    def ci(a: np.ndarray) -> tuple[float, float]:
        lo, hi = np.quantile(a, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    n_all, n_t, y_t, y_c = _counts(srt[0], srt[1], None, srt[2])
    N = float(n_all[-1])
    grid = np.linspace(0, N, curve_points)
    q = np.interp(grid, n_all, _qini_values(n_all, n_t, y_t, y_c)) / N
    u = np.interp(grid, n_all, _uplift_values(n_all, n_t, y_t, y_c)) / N
    dec = decile_table(y, t, score, n_bins, alpha)
    return {
        "qini": qini,
        "qini_ci": ci(boot_q),
        "auuc": auuc,
        "auuc_ci": ci(boot_a),
        "uplift_at_k": uk,
        "uplift_at_k_ci": {k: ci(v) for k, v in boot_k.items()},
        "deciles": dec,
        "calibration": calibration_summary(dec),
        "curve_x": (grid / N).tolist(),
        "curve_qini": q.tolist(),
        "curve_uplift": u.tolist(),
    }
