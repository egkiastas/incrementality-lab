"""Criteo-shaped synthetic experiments with known effects.

12 correlated Gaussian features, T ~ Bernoulli(0.85), exposure only for treated users, and
visits affected only through exposure, so the ITT CATE is e(x) * tau_e(x) (negative for some
users). Conversion happens only after a visit with a probability that ignores T.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass
class SyntheticExperiment:
    X: np.ndarray
    treatment: np.ndarray
    exposure: np.ndarray
    visit: np.ndarray
    conversion: np.ndarray
    p0: np.ndarray
    exposure_prob: np.ndarray
    cate_visit: np.ndarray
    cate_conversion: np.ndarray

    @property
    def true_ate_visit(self) -> float:
        return float(self.cate_visit.mean())

    @property
    def true_ate_conversion(self) -> float:
        return float(self.cate_conversion.mean())

    @property
    def true_late_visit(self) -> float:
        """Effect of exposure on the (would-be) exposed: E[e tau_e] / E[e]."""
        return float(self.cate_visit.mean() / self.exposure_prob.mean())

    def to_polars(self) -> pl.DataFrame:
        cols: dict[str, np.ndarray] = {f"f{j}": self.X[:, j].astype(np.float64) for j in range(12)}
        cols |= {
            "treatment": self.treatment.astype(np.int8),
            "conversion": self.conversion.astype(np.int8),
            "visit": self.visit.astype(np.int8),
            "exposure": self.exposure.astype(np.int8),
        }
        return pl.DataFrame(cols)


def generate(
    n: int,
    seed: int = 0,
    share: float = 0.85,
    exposure_intercept: float = 0.5,
    effect_scale: float = 1.0,
    baseline_intercept: float = -2.0,
) -> SyntheticExperiment:
    """Draw one synthetic experiment of ``n`` users.

    ``effect_scale=0`` gives an exact A/A experiment (no effect for anyone).
    ``exposure_intercept`` controls the exposure rate among the treated (~62% at 0.5,
    ~5% at -3, matching Criteo's 3.6% around -3.3).
    """
    rng = np.random.default_rng(seed)
    d = 12
    # Mildly correlated Gaussian features (AR(1) correlation 0.3 between neighbours).
    cov = 0.3 ** np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    X = rng.multivariate_normal(np.zeros(d), cov, size=n).astype(np.float32)
    f = X.astype(np.float64)

    p0 = _sigmoid(
        baseline_intercept + 0.8 * f[:, 0] + 0.5 * f[:, 1] - 0.4 * f[:, 2] + 0.3 * f[:, 3] * f[:, 4]
    )
    # Effect of exposure (probability scale): strong, non-linear heterogeneity in f2, f3,
    # and a "do-not-disturb" pocket (negative effect) for f7 > 1.
    tau_e = effect_scale * (0.20 * _sigmoid(1.5 * (f[:, 2] + f[:, 3])) - 0.10 * (f[:, 7] > 1.0))
    tau_e = np.clip(tau_e, -p0 + 1e-3, 1 - p0 - 1e-3)
    e = _sigmoid(exposure_intercept + 0.8 * f[:, 5])
    q = _sigmoid(-2.5 + 0.6 * f[:, 1])

    t = (rng.random(n) < share).astype(np.int8)
    expo = ((rng.random(n) < e) & (t == 1)).astype(np.int8)
    p_visit = p0 + expo * tau_e
    visit = (rng.random(n) < p_visit).astype(np.int8)
    conv = ((rng.random(n) < q) & (visit == 1)).astype(np.int8)
    cate_v = e * tau_e
    return SyntheticExperiment(
        X=X,
        treatment=t,
        exposure=expo,
        visit=visit,
        conversion=conv,
        p0=p0,
        exposure_prob=e,
        cate_visit=cate_v,
        cate_conversion=cate_v * q,
    )


def write_fixture(path: Path, n: int = 40_000, seed: int = 7) -> Path:
    """Write a small Criteo-schema Parquet file (used by CI smoke tests; no real data)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exp = generate(n, seed=seed, exposure_intercept=-1.0)
    exp.to_polars().write_parquet(path)
    return path
