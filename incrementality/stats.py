"""Basic estimators and tests. `t` is a 0/1 treatment flag; CIs are two-sided 1 - alpha."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Estimate:
    estimate: float
    se: float
    ci_low: float
    ci_high: float

    @property
    def p_value(self) -> float:
        if not self.se > 0:
            return float("nan")
        return float(2 * sps.norm.sf(abs(self.estimate / self.se)))

    def to_dict(self) -> dict[str, float]:
        return asdict(self) | {"p_value": self.p_value}


def z_crit(alpha: float) -> float:
    return float(sps.norm.ppf(1 - alpha / 2))


def normal_ci(est: float, se: float, alpha: float = 0.05) -> Estimate:
    z = z_crit(alpha)
    return Estimate(float(est), float(se), float(est - z * se), float(est + z * se))


# An arm is summarised as (mean, variance, n).
Moments = tuple[float, float, float]


def moments(y: np.ndarray) -> Moments:
    y = np.asarray(y, np.float64)
    return float(y.mean()), float(y.var(ddof=1)) if y.size > 1 else float("nan"), float(y.size)


def moments_from_sums(n: float, s: float, ss: float) -> Moments:
    mean = s / n
    var = (ss - n * mean**2) / (n - 1) if n > 1 else float("nan")
    return float(mean), float(max(var, 0.0)), float(n)


def diff_from_moments(a1: Moments, a0: Moments, alpha: float = 0.05) -> Estimate:
    """Neyman difference in means: Var = s1^2/n1 + s0^2/n0."""
    (m1, v1, n1), (m0, v0, n0) = a1, a0
    return normal_ci(m1 - m0, np.sqrt(v1 / n1 + v0 / n0), alpha)


def lift_from_moments(a1: Moments, a0: Moments, alpha: float = 0.05) -> Estimate:
    """m1/m0 - 1 with delta-method SE: Var ~ Var(m1)/m0^2 + m1^2 Var(m0)/m0^4."""
    (m1, v1, n1), (m0, v0, n0) = a1, a0
    se = np.sqrt((v1 / n1) / m0**2 + m1**2 * (v0 / n0) / m0**4)
    return normal_ci(m1 / m0 - 1, se, alpha)


def diff_in_means(y: np.ndarray, t: np.ndarray, alpha: float = 0.05) -> Estimate:
    tb = np.asarray(t).astype(bool)
    return diff_from_moments(moments(y[tb]), moments(y[~tb]), alpha)


def relative_lift(y: np.ndarray, t: np.ndarray, alpha: float = 0.05) -> Estimate:
    tb = np.asarray(t).astype(bool)
    return lift_from_moments(moments(y[tb]), moments(y[~tb]), alpha)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bootstrap_rows(
    y: np.ndarray, t: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Within-arm row bootstrap of (difference, relative lift). Reference for the test below."""
    tb = np.asarray(t).astype(bool)
    y1, y0 = np.asarray(y[tb], float), np.asarray(y[~tb], float)
    diffs, lifts = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        m1 = y1[rng.integers(0, y1.size, y1.size)].mean()
        m0 = y0[rng.integers(0, y0.size, y0.size)].mean()
        diffs[b], lifts[b] = m1 - m0, m1 / m0 - 1
    return diffs, lifts


def bootstrap_binary(
    k1: int, n1: int, k0: int, n0: int, n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Same distribution as `bootstrap_rows` for 0/1 outcomes: resampling n rows from an arm
    with k ones gives Binomial(n, k/n) ones. Two draws per replicate instead of 14M."""
    m1 = rng.binomial(n1, k1 / n1, size=n_boot) / n1
    m0 = rng.binomial(n0, k0 / n0, size=n_boot) / n0
    return m1 - m0, m1 / m0 - 1


def srm_test(n_treatment: int, n_control: int, expected_share: float) -> dict[str, float]:
    n = n_treatment + n_control
    exp_t, exp_c = n * expected_share, n * (1 - expected_share)
    chi2 = (n_treatment - exp_t) ** 2 / exp_t + (n_control - exp_c) ** 2 / exp_c
    return {
        "n_treatment": n_treatment,
        "n_control": n_control,
        "expected_share": expected_share,
        "observed_share": n_treatment / n,
        "chi2": float(chi2),
        "p_value": float(sps.chi2.sf(chi2, df=1)),
    }


def smd(x1: np.ndarray, x0: np.ndarray) -> float:
    x1, x0 = np.asarray(x1, np.float64), np.asarray(x0, np.float64)
    pooled = np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)
    diff = x1.mean() - x0.mean()
    if pooled == 0:
        return 0.0 if diff == 0 else float(np.sign(diff) * np.inf)
    return float(diff / pooled)


def smd_table(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    tb = np.asarray(t).astype(bool)
    return np.array([smd(X[tb, j], X[~tb, j]) for j in range(X.shape[1])])


def wald_iv(y: np.ndarray, z: np.ndarray, d: np.ndarray, alpha: float = 0.05) -> Estimate:
    """ITT_y / ITT_d with a delta-method SE. With one-sided noncompliance (control is never
    exposed) and the exclusion restriction, this is the effect of exposure on the exposed."""
    zb = np.asarray(z).astype(bool)
    y = np.asarray(y, np.float64)
    d = np.asarray(d, np.float64)
    y1, y0, d1, d0 = y[zb], y[~zb], d[zb], d[~zb]
    n1, n0 = y1.size, y0.size
    dy, dd = y1.mean() - y0.mean(), d1.mean() - d0.mean()
    beta = dy / dd
    var_dy = y1.var(ddof=1) / n1 + y0.var(ddof=1) / n0
    var_dd = d1.var(ddof=1) / n1 + (d0.var(ddof=1) / n0 if d0.var() > 0 else 0.0)

    def cov(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.cov(a, b, ddof=1)[0, 1]) if a.size > 1 else 0.0

    c = cov(y1, d1) / n1 + cov(y0, d0) / n0
    var_beta = (var_dy - 2 * beta * c + beta**2 * var_dd) / dd**2
    return normal_ci(beta, np.sqrt(max(var_beta, 0.0)), alpha)


def mde(sigma2: float, n_total: float, share: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z = z_crit(alpha) + float(sps.norm.ppf(power))
    return float(z * np.sqrt(sigma2 * (1 / (share * n_total) + 1 / ((1 - share) * n_total))))


def required_n(sigma2: float, effect: float, share: float, alpha: float = 0.05, power: float = 0.8) -> float:
    z = z_crit(alpha) + float(sps.norm.ppf(power))
    return float(z**2 * sigma2 / (effect**2 * share * (1 - share)))
