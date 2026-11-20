"""Settings for the full run and for the small synthetic fixture used in CI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

LGBM = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass(frozen=True)
class Config:
    seed: int = 20261119
    parquet: Path = Path("data/processed/criteo_uplift_v21.parquet")
    raw_csv: Path = Path("data/raw/criteo-research-uplift-v2.1.csv.gz")
    results_dir: Path = Path("report/results")
    figures_dir: Path = Path("report/figures")
    artifacts_dir: Path = Path("artifacts")
    expected_rows: int = 13_979_592
    features: tuple[str, ...] = tuple(f"f{i}" for i in range(12))
    alpha: float = 0.05
    treatment_share: float = 0.85

    # models
    lgbm: dict[str, Any] = field(default_factory=lambda: dict(LGBM))
    # the CATE stage of X/DR learners fits very noisy pseudo-outcomes: smaller, slower trees
    lgbm_cate: dict[str, Any] = field(
        default_factory=lambda: (
            LGBM | {"n_estimators": 200, "num_leaves": 31, "min_child_samples": 2000, "learning_rate": 0.03}
        )
    )
    lgbm_propensity: dict[str, Any] = field(default_factory=lambda: LGBM | {"n_estimators": 200})

    # 01 health
    health_aa_sims: int = 1000
    propensity_check_rows: int = 2_000_000
    # 02 average effects
    n_bootstrap: int = 2000
    # 03 variance reduction
    cross_fit_folds: int = 2
    train_rows_per_fold: int = 3_000_000
    # 04 sequential
    seq_aa_sims: int = 1000
    seq_users_per_sim: int = 500_000
    seq_looks: int = 100
    first_look: int = 10_000
    tau_rel: float = 0.10
    replay_orderings: int = 100
    replay_looks: int = 120
    mde_grid: tuple[int, ...] = (100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 14_000_000)
    power_sims: int = 2000
    # 05 uplift
    test_frac: float = 0.30
    val_frac: float = 0.10
    train_rows: int = 4_000_000
    forest_rows: int = 500_000
    forest_trees: int = 200
    forest_min_leaf: int = 200
    uplift_bootstrap: int = 200
    learners: tuple[str, ...] = ("s_learner", "t_learner", "x_learner", "dr_learner", "causal_forest")
    # 06 policy
    budgets: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50)
    population: int = 1_000_000


def fixture() -> Config:
    small = LGBM | {"n_estimators": 30, "min_child_samples": 50, "num_leaves": 15}
    return replace(
        Config(),
        seed=7,
        parquet=Path("tests/fixtures/criteo_fixture.parquet"),
        results_dir=Path("build/fixture/results"),
        figures_dir=Path("build/fixture/figures"),
        artifacts_dir=Path("build/fixture/artifacts"),
        expected_rows=40_000,
        lgbm=small,
        lgbm_cate=LGBM | {"n_estimators": 30, "min_child_samples": 100, "num_leaves": 7},
        lgbm_propensity=LGBM | {"n_estimators": 20, "min_child_samples": 50},
        health_aa_sims=50,
        propensity_check_rows=20_000,
        n_bootstrap=200,
        train_rows_per_fold=20_000,
        seq_aa_sims=30,
        seq_users_per_sim=5_000,
        seq_looks=20,
        first_look=500,
        replay_orderings=5,
        replay_looks=20,
        mde_grid=(10_000, 40_000),
        power_sims=200,
        train_rows=20_000,
        forest_rows=5_000,
        forest_trees=20,
        forest_min_leaf=20,
        uplift_bootstrap=10,
    )


def from_args(description: str = "", reuse_flag: bool = False) -> tuple[Config, argparse.Namespace]:
    """Every analysis script takes `--fixture` to run on the small synthetic dataset."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--fixture", action="store_true", help="run on the synthetic fixture")
    if reuse_flag:
        parser.add_argument(
            "--reuse-predictions", action="store_true", help="skip training, re-score saved predictions"
        )
    args = parser.parse_args()
    return (fixture() if args.fixture else Config()), args
