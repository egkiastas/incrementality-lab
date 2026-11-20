"""CUPED / CUPAC, and what adjusting for features does to the estimate.

There is no pre-period outcome in this dataset, so the pre-treatment features f0..f11 play the
CUPED covariate role. On a clean RCT every adjusted estimator should agree with the difference
in means and only narrow the interval. Here they don't, so this step also shows why (treatment
share by predicted-visit ventile) and checks on control-only A/A splits that the adjustment
itself is unbiased.
"""

from typing import Any

import numpy as np
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from incrementality import data, results
from incrementality.adjustment import aa_check, aipw, cross_fitted_predictions, cuped, post_stratified
from incrementality.config import Config, from_args
from incrementality.plotting import BASELINE, INK_2, SERIES, new_figures, save, subtitle

ESTIMATORS = {
    "diff_in_means": "Difference in means",
    "cuped_linear": "CUPED (12 features, linear)",
    "cupac": "CUPAC (cross-fitted GBM)",
    "post_stratified": "Post-stratified on GBM score",
    "aipw_known_propensity": "AIPW, design propensity",
    "aipw_estimated_propensity": "AIPW, estimated propensity",
}


def main(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed + 2)
    ds = data.load(cfg.parquet, cfg.features)
    t = ds.treatment
    share = float(t.mean())
    out: dict[str, Any] = {"n": ds.n}
    for name in ("visit", "conversion"):
        y = ds.outcome(name)
        lin = cuped(y, t, ds.X, cfg.alpha)
        g = cross_fitted_predictions(ds.X, y, cfg.cross_fit_folds, cfg.lgbm, rng, cfg.train_rows_per_fold)
        cp = cuped(y, t, g, cfg.alpha)
        ps, strata = post_stratified(y, t, g, 20, cfg.alpha)
        fit = (cfg.lgbm, rng, cfg.cross_fit_folds)
        known = aipw(
            y, t, ds.X, *fit, propensity=share, alpha=cfg.alpha, train_rows_per_fold=cfg.train_rows_per_fold
        )
        est = aipw(
            y, t, ds.X, *fit, propensity=None, alpha=cfg.alpha, train_rows_per_fold=cfg.train_rows_per_fold
        )
        ctrl = t == 0
        aa = aa_check(y[ctrl], g[ctrl], cfg.treatment_share, 500, rng)

        base = lin["baseline"]
        rows = {
            "diff_in_means": base.to_dict() | {"relative_lift": base.estimate / float(y[ctrl].mean())},
            "cuped_linear": lin["estimate"].to_dict(),
            "cupac": cp["estimate"].to_dict(),
            "post_stratified": ps.to_dict(),
            "aipw_known_propensity": known["ate"].to_dict(),
            "aipw_estimated_propensity": est["ate"].to_dict()
            | {k: v for k, v in est.items() if k not in ("ate", "relative_lift")}
            | {"relative_lift": est["relative_lift"].to_dict()},
        }
        for k, r in rows.items():
            if k != "diff_in_means":
                r["variance_reduction"] = 1 - r["se"] ** 2 / base.se**2
                r["shift_in_se"] = (r["estimate"] - base.estimate) / base.se
        out[name] = rows | {
            "cupac_model_auc": float(roc_auc_score(y, g)),
            "strata": strata,
            "aa_check": {
                "n_sims": 500,
                "mean_estimate_cupac": float(aa["est_cuped"].mean()),
                "mc_se_cupac": float(aa["est_cuped"].std(ddof=1) / np.sqrt(500)),
                "fpr_plain": float((aa["p_plain"] < 0.05).mean()),
                "fpr_cupac": float((aa["p_cuped"] < 0.05).mean()),
                "variance_reduction": float(1 - aa["est_cuped"].var(ddof=1) / aa["est_plain"].var(ddof=1)),
                "ks_uniform_p_cupac": float(sps.kstest(aa["p_cuped"], "uniform").pvalue),
            },
        }
    results.save(cfg.results_dir, "cuped", out, cfg.seed)
    plot(cfg, out)
    return out


def plot(cfg: Config, out: dict[str, Any]) -> None:
    fig, axes = new_figures(2, w=11, h=4.0)
    for ax, name in zip(axes, ("visit", "conversion"), strict=True):
        r = out[name]
        for i, key in enumerate(ESTIMATORS):
            e = r[key]
            x = e["estimate"] * 100
            colour = SERIES[1] if key == "diff_in_means" else SERIES[2]
            ax.errorbar(
                x,
                i,
                xerr=[[x - e["ci_low"] * 100], [e["ci_high"] * 100 - x]],
                fmt="o",
                color=colour,
                ms=7,
                capsize=4,
            )
            vr = e.get("variance_reduction")
            note = f"  {x:.3f}" + (f"   (variance {-vr * 100:+.0f}%)" if vr is not None else "")
            ax.text(e["ci_high"] * 100, i, note, va="center", fontsize=8, color=INK_2)
        ax.set_yticks(range(len(ESTIMATORS)), list(ESTIMATORS.values()))
        ax.invert_yaxis()
        lo = min(r[k]["ci_low"] for k in ESTIMATORS) * 100
        hi = max(r[k]["ci_high"] for k in ESTIMATORS) * 100
        ax.set_xlim(lo - (hi - lo) * 0.1, hi + (hi - lo) * 0.7)
        ax.set_xlabel("ATE (percentage points), 95% CI")
        ax.set_title(f"{name.capitalize()}: adjusting for features moves the estimate", pad=22)
        sh = r["aipw_estimated_propensity"]["shift_in_se"]
        subtitle(
            ax,
            f"{out['n'] / 1e6:.2f}M users; AIPW is {abs(sh):.0f} SEs "
            f"{'below' if sh < 0 else 'above'} the raw difference",
        )
    axes[1].set_yticklabels([])
    fig.subplots_adjust(wspace=0.08)
    save(fig, cfg.figures_dir, "cuped_estimators")

    st = out["visit"]["strata"]
    k = np.array([s["stratum"] for s in st])
    fig, (ax0, ax1) = new_figures(2, w=12.5, h=3.8)
    ax0.plot(k, [s["treatment_share"] * 100 for s in st], color=SERIES[0], marker="o", ms=5)
    ax0.axhline(85, color=BASELINE, ls="--", lw=1.2)
    ax0.text(1, 85, "planned 85%", color=INK_2, fontsize=8.5, va="bottom")
    ax0.set_xticks(k[::2])
    ax0.set_xlabel("ventile of predicted visit probability (treatment-blind GBM)")
    ax0.set_ylabel("share of users in treatment (%)")
    ax0.set_title("Treated users are over-represented among likely visitors", pad=22)
    subtitle(ax0, "under clean randomization this line is flat at 85%")
    ax1.bar(k, [s["control_rate"] * 100 for s in st], color=SERIES[0], width=0.7)
    ax1.set_yscale("log")
    ax1.set_xticks(k[::2])
    ax1.set_xlabel("ventile of predicted visit probability")
    ax1.set_ylabel("control visit rate (%, log)")
    ax1.set_title("...who visit 1,000x more often anyway", pad=22)
    subtitle(ax1, "so a raw treated-vs-control difference partly measures who was treated")
    save(fig, cfg.figures_dir, "cuped_imbalance_mechanism")


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
