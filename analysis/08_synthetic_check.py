"""Run the same estimators on synthetic data where the true effects are known."""

from typing import Any

import numpy as np

from incrementality import results
from incrementality.adjustment import aipw, cross_fitted_predictions, cuped
from incrementality.config import LGBM, Config, from_args
from incrementality.sequential import linear_looks, simulate_aa
from incrementality.stats import diff_in_means, wald_iv
from incrementality.synthetic import generate
from incrementality.uplift import fit_predict, qini_auc_score

BASE = LGBM | {"n_estimators": 150, "num_leaves": 31, "min_child_samples": 200}
CATE = LGBM | {"n_estimators": 150, "num_leaves": 15, "min_child_samples": 500, "learning_rate": 0.03}


def main(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed + 9)
    tr, te = generate(200_000, seed=11), generate(40_000, seed=12)
    truth = tr.true_ate_visit
    g = cross_fitted_predictions(tr.X, tr.visit, 2, BASE, rng)
    estimates = {
        "diff_in_means": diff_in_means(tr.visit, tr.treatment),
        "cuped_linear": cuped(tr.visit, tr.treatment, tr.X)["estimate"],
        "cupac": cuped(tr.visit, tr.treatment, g)["estimate"],
        "aipw_known_propensity": aipw(tr.visit, tr.treatment, tr.X, BASE, rng, propensity=0.85)["ate"],
    }
    iv = wald_iv(tr.visit, tr.treatment, tr.exposure)
    out: dict[str, Any] = {
        "true_ate": truth,
        "ate": {
            k: v.to_dict() | {"covers_truth": v.ci_low <= truth <= v.ci_high} for k, v in estimates.items()
        },
        "iv": iv.to_dict()
        | {"truth": tr.true_late_visit, "covers_truth": iv.ci_low <= tr.true_late_visit <= iv.ci_high},
    }

    covered_dim, covered_cuped, err_dim, err_cuped = [], [], [], []
    for s in range(300):
        e = generate(8_000, seed=5000 + s)
        d = diff_in_means(e.visit, e.treatment)
        c = cuped(e.visit, e.treatment, e.X)["estimate"]
        covered_dim.append(d.ci_low <= e.true_ate_visit <= d.ci_high)
        covered_cuped.append(c.ci_low <= e.true_ate_visit <= c.ci_high)
        err_dim.append(d.estimate - e.true_ate_visit)
        err_cuped.append(c.estimate - e.true_ate_visit)
    out["replications"] = {
        "n": 300,
        "coverage_diff_in_means": float(np.mean(covered_dim)),
        "coverage_cuped": float(np.mean(covered_cuped)),
        "bias_cuped": float(np.mean(err_cuped)),
        "mc_se_bias_cuped": float(np.std(err_cuped, ddof=1) / np.sqrt(300)),
        "variance_ratio_cuped_vs_dim": float(np.var(err_cuped) / np.var(err_dim)),
    }

    cate = {}
    for name in [*cfg.learners, "response_model"]:
        p = fit_predict(name, tr.X, tr.treatment, tr.visit, te.X, BASE, CATE, 0.85, 0, 100, 100)
        cate[name] = {
            "corr_with_true_cate": float(np.corrcoef(p, te.cate_visit)[0, 1]),
            "qini": qini_auc_score(te.visit, te.treatment, p),
        }
    cate["oracle"] = {
        "corr_with_true_cate": 1.0,
        "qini": qini_auc_score(te.visit, te.treatment, te.cate_visit),
    }
    out["cate"] = cate

    pool = (rng.random(400_000) < 0.05).astype(np.int8)
    aa = simulate_aa(pool, 400, 20_000, 0.85, linear_looks(20_000, 50, 500), 0.05, (0.1 * 0.05) ** 2, rng)
    out["sequential"] = {k: aa[k] for k in ("n_sims", "fpr_fixed_horizon", "fpr_naive_peeking", "fpr_msprt")}
    results.save(cfg.results_dir, "synthetic_check", out, cfg.seed)
    return out


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
