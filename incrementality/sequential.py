"""Two-sample mixture SPRT (Johari et al., "Always Valid Inference") and its confidence sequence.

With difference theta_hat, variance V and a N(0, tau^2) mixture over the true effect,

    Lambda = sqrt(V / (V + tau^2)) * exp(tau^2 theta_hat^2 / (2 V (V + tau^2))).

Under H0 Lambda is a nonnegative martingale with mean 1, so P(sup Lambda >= 1/alpha) <= alpha
and you can stop the first time it crosses 1/alpha. Variances are plug-in estimates, so the
guarantee is asymptotic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sps


def msprt_log_lr(diff: np.ndarray, V: np.ndarray, tau2: float) -> np.ndarray:
    diff, V = np.asarray(diff, np.float64), np.asarray(V, np.float64)
    return 0.5 * np.log(V / (V + tau2)) + tau2 * diff**2 / (2 * V * (V + tau2))


def cs_halfwidth(V: np.ndarray, tau2: float, alpha: float) -> np.ndarray:
    """Half-width of the always-valid confidence sequence (where log Lambda = log(1/alpha))."""
    V = np.asarray(V, np.float64)
    return np.sqrt(2 * V * (V + tau2) / tau2 * (np.log(1 / alpha) + 0.5 * np.log((V + tau2) / V)))


def look_sums(y: np.ndarray, t: np.ndarray, ends: np.ndarray) -> dict[str, np.ndarray]:
    """Cumulative per-arm counts, sums and sums of squares after each look."""
    y = np.asarray(y, np.float64)
    tf = np.asarray(t).astype(np.float64)
    starts = np.r_[0, ends[:-1]].astype(np.int64)
    yt = y * tf
    yc = y - yt
    out = {
        "n1": np.cumsum(np.add.reduceat(tf, starts)),
        "s1": np.cumsum(np.add.reduceat(yt, starts)),
        "ss1": np.cumsum(np.add.reduceat(yt * y, starts)),
        "s0": np.cumsum(np.add.reduceat(yc, starts)),
        "ss0": np.cumsum(np.add.reduceat(yc * y, starts)),
    }
    out["n0"] = ends.astype(np.float64) - out["n1"]
    return out


def sequential_path(
    y: np.ndarray, t: np.ndarray, ends: np.ndarray, alpha: float, tau2: float
) -> dict[str, np.ndarray]:
    """What an analyst sees at each look: estimate, naive p-value, mSPRT statistic, CS and CI."""
    s = look_sums(y, t, ends)
    n1, n0 = s["n1"], s["n0"]
    with np.errstate(divide="ignore", invalid="ignore"):
        m1, m0 = s["s1"] / n1, s["s0"] / n0
        v1 = (s["ss1"] - n1 * m1**2) / (n1 - 1)
        v0 = (s["ss0"] - n0 * m0**2) / (n0 - 1)
        V = v1 / n1 + v0 / n0
        diff = m1 - m0
        z = diff / np.sqrt(V)
    p = 2 * sps.norm.sf(np.abs(z))
    bad = ~np.isfinite(V) | (V <= 0)  # an arm with no variance yet: no evidence either way
    V = np.where(bad, np.inf, V)
    p = np.where(bad, 1.0, p)
    llr = np.where(bad, 0.0, msprt_log_lr(np.where(bad, 0.0, diff), np.where(bad, 1.0, V), tau2))
    hw = np.where(bad, np.inf, cs_halfwidth(np.where(bad, 1.0, V), tau2, alpha))
    se = np.sqrt(V)
    zc = float(sps.norm.ppf(1 - alpha / 2))
    return {
        "n": ends.astype(np.int64),
        "diff": diff,
        "p_naive": p,
        "log_lr": llr,
        "cs_low": diff - hw,
        "cs_high": diff + hw,
        "ci_low": diff - zc * se,
        "ci_high": diff + zc * se,
    }


def first_look(mask: np.ndarray) -> int:
    hit = np.flatnonzero(mask)
    return int(hit[0]) if hit.size else -1


def linear_looks(n: int, n_looks: int, first: int) -> np.ndarray:
    return np.unique(np.linspace(first, n, n_looks).round().astype(np.int64))


def geometric_looks(n: int, n_looks: int, first: int) -> np.ndarray:
    return np.unique(np.geomspace(first, n, n_looks).round().astype(np.int64))


def _rate_ci(p: float, n: int) -> list[float]:
    se = np.sqrt(p * (1 - p) / n)
    return [max(0.0, p - 1.96 * se), p + 1.96 * se]


def simulate_aa(
    y_pool: np.ndarray,
    n_sims: int,
    users_per_sim: int,
    share: float,
    looks: np.ndarray,
    alpha: float,
    tau2: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """A/A tests on real control users: random arrival order, fake labels, a look at each step."""
    looks = looks[looks <= users_per_sim]
    rej_fixed = np.zeros(n_sims, bool)
    first_naive = np.full(n_sims, -1)
    first_msprt = np.full(n_sims, -1)
    for s in range(n_sims):
        idx = rng.choice(y_pool.size, size=users_per_sim, replace=False)
        fake_t = rng.random(users_per_sim) < share
        path = sequential_path(y_pool[idx], fake_t, looks, alpha, tau2)
        rej_fixed[s] = path["p_naive"][-1] < alpha
        first_naive[s] = first_look(path["p_naive"] < alpha)
        first_msprt[s] = first_look(path["log_lr"] >= np.log(1 / alpha))
    k = np.arange(looks.size)
    out: dict[str, Any] = {
        "n_sims": n_sims,
        "looks": looks.tolist(),
        "fpr_fixed_horizon": float(rej_fixed.mean()),
        "fpr_naive_peeking": float((first_naive >= 0).mean()),
        "fpr_msprt": float((first_msprt >= 0).mean()),
        "fpr_naive_by_look": [float(np.mean((first_naive >= 0) & (first_naive <= j))) for j in k],
        "fpr_msprt_by_look": [float(np.mean((first_msprt >= 0) & (first_msprt <= j))) for j in k],
    }
    for key in ("fpr_fixed_horizon", "fpr_naive_peeking", "fpr_msprt"):
        out[key + "_ci"] = _rate_ci(out[key], n_sims)
    return out


def time_to_decision(
    y: np.ndarray,
    t: np.ndarray,
    looks: np.ndarray,
    alpha: float,
    tau2: float,
    n_orderings: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Replay the experiment in random arrival orders; record when mSPRT (and naive) first reject."""
    full_diff = float(y[t == 1].mean() - y[t == 0].mean())
    users_msprt: list[int | None] = []
    sign_ok: list[bool] = []
    for _ in range(n_orderings):
        perm = rng.permutation(y.size)
        path = sequential_path(y[perm], t[perm], looks, alpha, tau2)
        j = first_look(path["log_lr"] >= np.log(1 / alpha))
        users_msprt.append(int(path["n"][j]) if j >= 0 else None)
        if j >= 0:
            sign_ok.append(bool(np.sign(path["diff"][j]) == np.sign(full_diff)))
    decided = [u for u in users_msprt if u is not None]
    return {
        "n_orderings": n_orderings,
        "full_sample_diff": full_diff,
        "share_decided_msprt": len(decided) / n_orderings,
        "users_to_decision_msprt": users_msprt,
        "median_users_msprt": float(np.median(decided)) if decided else None,
        "p10_users_msprt": float(np.quantile(decided, 0.1)) if decided else None,
        "p90_users_msprt": float(np.quantile(decided, 0.9)) if decided else None,
        "sign_correct_at_decision": float(np.mean(sign_ok)) if sign_ok else None,
    }


def empirical_power(
    p0: float,
    effect: float,
    n_total: int,
    share: float,
    alpha: float,
    n_sims: int,
    rng: np.random.Generator,
) -> float:
    n1 = round(n_total * share)
    n0 = n_total - n1
    m1 = rng.binomial(n1, p0 + effect, size=n_sims) / n1
    m0 = rng.binomial(n0, p0, size=n_sims) / n0
    z = (m1 - m0) / np.sqrt(m1 * (1 - m1) / n1 + m0 * (1 - m0) / n0)
    return float(np.mean(np.abs(z) > sps.norm.ppf(1 - alpha / 2)))
