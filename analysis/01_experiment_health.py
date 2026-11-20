"""Can the randomization be trusted? SRM in both tails, balance, treatment predictability, A/A."""

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

from incrementality import data, results
from incrementality.config import Config, from_args
from incrementality.plotting import BASELINE, INK_2, SERIES, new_figure, new_figures, save, subtitle
from incrementality.stats import diff_from_moments, moments_from_sums, smd_table, srm_test


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982)."""
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    var = auc * (1 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
    return float(np.sqrt(var / (n_pos * n_neg)))


def aa_pvalues(
    y_control: dict[str, np.ndarray], share: float, n_sims: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    n = next(iter(y_control.values())).size
    out = {k: np.empty(n_sims) for k in y_control}
    yf = {k: v.astype(np.float64) for k, v in y_control.items()}
    for s in range(n_sims):
        fake = rng.random(n) < share
        n1 = int(fake.sum())
        for k, y in yf.items():
            s1 = float(y[fake].sum())
            s0 = float(y.sum()) - s1
            # binary outcome, so the sum of squares equals the sum
            est = diff_from_moments(moments_from_sums(n1, s1, s1), moments_from_sums(n - n1, s0, s0))
            out[k][s] = est.p_value
    return out


def main(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    ds = data.load(cfg.parquet, cfg.features)
    t = ds.treatment.astype(bool)
    n_t, n_c = int(t.sum()), int((~t).sum())
    n, p = n_t + n_c, cfg.treatment_share

    integrity = {
        "n_rows": ds.n,
        "control_exposed": int(ds.exposure[~t].sum()),
        "conversions_without_visit": int(((ds.conversion == 1) & (ds.visit == 0)).sum()),
        "exposure_rate_treated": float(ds.exposure[t].mean()),
        "visit_rate": float(ds.visit.mean()),
        "conversion_rate": float(ds.conversion.mean()),
    }

    srm = srm_test(n_t, n_c, p)
    # The other tail: how likely is a Bernoulli(p) split to land this close to p?
    z_share = abs(srm["observed_share"] - p) / np.sqrt(p * (1 - p) / n)
    srm["prob_share_this_close_under_bernoulli"] = float(2 * sps.norm.cdf(z_share) - 1)

    smd = smd_table(ds.X, ds.treatment)
    smd_se = float(np.sqrt(1 / n_t + 1 / n_c))  # SE of an SMD under randomization
    balance = [
        {
            "feature": f,
            "smd": float(smd[j]),
            "z": float(smd[j] / smd_se),
            "p_value": float(2 * sps.norm.sf(abs(smd[j]) / smd_se)),
        }
        for j, f in enumerate(ds.feature_names)
    ]

    # If assignment were independent of X, no model could beat AUC 0.5.
    parts = data.stratified_split(data.strata(ds), {"a": 0.5, "b": 0.5}, rng)
    take = cfg.propensity_check_rows // 2
    tr = rng.choice(parts["a"], size=min(take, parts["a"].size), replace=False)
    te = rng.choice(parts["b"], size=min(take, parts["b"].size), replace=False)
    clf = LGBMClassifier(**cfg.lgbm_propensity, random_state=cfg.seed).fit(ds.X[tr], ds.treatment[tr])
    auc = float(roc_auc_score(ds.treatment[te], np.asarray(clf.predict_proba(ds.X[te]))[:, 1]))
    n_pos = int(ds.treatment[te].sum())
    se = auc_se(auc, n_pos, te.size - n_pos)
    propensity = {
        "auc": auc,
        "auc_ci": [auc - 1.96 * se, auc + 1.96 * se],
        "p_value_vs_0.5": float(2 * sps.norm.sf(abs(auc - 0.5) / se)),
        "test_rows": int(te.size),
    }

    pv = aa_pvalues({"visit": ds.visit[~t], "conversion": ds.conversion[~t]}, p, cfg.health_aa_sims, rng)
    aa = {}
    for k, v in pv.items():
        fpr = float((v < cfg.alpha).mean())
        half = 1.96 * np.sqrt(fpr * (1 - fpr) / v.size)
        aa[k] = {
            "n_sims": v.size,
            "fpr": fpr,
            "fpr_ci": [max(0.0, fpr - half), fpr + half],
            "ks_uniform_p": float(sps.kstest(v, "uniform").pvalue),
        }

    out = {
        "integrity": integrity,
        "srm": srm,
        "balance": balance,
        "max_abs_smd": float(np.max(np.abs(smd))),
        "smd_se_under_randomization": smd_se,
        "n_features_significant_bonferroni": sum(b["p_value"] < cfg.alpha / len(balance) for b in balance),
        "propensity_check": propensity,
        "aa": aa,
    }
    results.save(cfg.results_dir, "health", out, cfg.seed)
    plot(cfg, balance, pv, ds.n)
    return out


def plot(cfg: Config, balance: list[dict[str, Any]], pv: dict[str, np.ndarray], n_users: int) -> None:
    fig, ax = new_figure(w=6.5, h=4.2)
    feats = [b["feature"] for b in balance][::-1]
    vals = [b["smd"] for b in balance][::-1]
    ax.axvspan(-0.1, 0.1, color="#eef4fc", zorder=0)
    ax.axvline(0, color=BASELINE, lw=1)
    ax.scatter(vals, feats, s=40, color=SERIES[0], zorder=3, edgecolor="white", linewidth=1.5)
    lim = max(0.12, max(abs(v) for v in vals) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("standardized mean difference (treated - control)")
    ax.set_title("Covariate balance: inside the usual band, but not zero", pad=22)
    subtitle(ax, f"{n_users / 1e6:.2f}M users; shaded band = |SMD| < 0.1")
    save(fig, cfg.figures_dir, "health_balance_smd")

    fig, axes = new_figures(2, w=9, h=3.6)
    for ax, (k, v) in zip(axes, pv.items(), strict=True):
        ax.hist(v, bins=20, range=(0, 1), color=SERIES[0], edgecolor="white", linewidth=2)
        ax.axhline(v.size / 20, color=BASELINE, lw=1.2, ls="--")
        ax.set_title(f"A/A p-values - {k}", pad=22)
        subtitle(ax, f"{v.size} random 85/15 splits of control; FPR = {(v < cfg.alpha).mean():.3f}")
        ax.set_xlabel("p-value")
        ax.set_ylabel("simulations")
    axes[0].text(
        0.98,
        v.size / 20,
        "uniform",
        color=INK_2,
        fontsize=8,
        ha="right",
        va="bottom",
        transform=axes[0].get_yaxis_transform(),
    )
    save(fig, cfg.figures_dir, "health_aa_pvalues")


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
