"""Where the visit-uplift learners disagree, and who those users are."""

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps

from incrementality import data, results
from incrementality.config import Config, from_args
from incrementality.plotting import (
    BASELINE,
    LEARNER_COLORS,
    SERIES,
    learner_label,
    new_figure,
    new_figures,
    save,
    subtitle,
)
from incrementality.stats import diff_in_means, smd
from incrementality.uplift import random_tiebreak_order


def main(cfg: Config) -> dict[str, Any]:
    preds = pl.read_parquet(cfg.artifacts_dir / "uplift_predictions.parquet")
    ds = data.load(cfg.parquet, cfg.features)
    split = preds["split"].to_numpy()
    test_rows = preds["row"].to_numpy()[split == 1]
    y, t, X = ds.visit[test_rows], ds.treatment[test_rows], ds.X[test_rows]
    learners = list(cfg.learners)
    P = np.column_stack([preds[f"visit__{n}"].to_numpy()[split == 1] for n in learners])
    resp = preds["visit__response_model"].to_numpy()[split == 1]
    R = np.column_stack([sps.rankdata(P[:, j]) / P.shape[0] for j in range(P.shape[1])])
    disagreement = R.std(axis=1)  # spread of each user's percentile rank across learners

    baseline_decile = np.empty(resp.size, np.int64)  # 0 = highest baseline propensity
    for q, idx in enumerate(np.array_split(random_tiebreak_order(resp), 10)):
        baseline_decile[idx] = q
    by_baseline = []
    for q in range(10):
        m = baseline_decile == q
        e = diff_in_means(y[m], t[m])
        by_baseline.append(
            {
                "decile": q + 1,
                "control_rate": float(y[m & (t == 0)].mean()),
                "observed_uplift": e.estimate,
                "ci_low": e.ci_low,
                "ci_high": e.ci_high,
                "mean_disagreement": float(disagreement[m].mean()),
                **{f"pred_{n}": float(P[m, j].mean()) for j, n in enumerate(learners)},
            }
        )

    high = disagreement >= np.quantile(disagreement, 0.9)
    profile = sorted(
        ((f, smd(X[high, j], X[~high, j])) for j, f in enumerate(ds.feature_names)),
        key=lambda kv: -abs(kv[1]),
    )
    resp_rank = sps.rankdata(resp) / resp.size
    out = {
        "learners": learners,
        "n_test": int(test_rows.size),
        "spearman": np.corrcoef(R, rowvar=False).tolist(),
        "prediction_sd": {n: float(P[:, j].std()) for j, n in enumerate(learners)},
        "rank_corr_with_response_score": {
            n: float(np.corrcoef(R[:, j], resp_rank)[0, 1]) for j, n in enumerate(learners)
        },
        "by_baseline_decile": by_baseline,
        "high_disagreement_users": {
            "share": float(high.mean()),
            "control_rate": float(y[high & (t == 0)].mean()),
            "rest_control_rate": float(y[~high & (t == 0)].mean()),
            "observed_uplift": diff_in_means(y[high], t[high]).to_dict(),
            "top_feature_smd": dict(profile[:5]),
        },
    }
    results.save(cfg.results_dir, "learners", out, cfg.seed)
    plot(cfg, out)
    return out


def plot(cfg: Config, out: dict[str, Any]) -> None:
    names = [learner_label(n) for n in out["learners"]]
    S = np.array(out["spearman"])
    fig, ax = new_figure(w=6.2, h=5.2)
    ax.grid(False)
    im = ax.imshow(S, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{S[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if S[i, j] > 0.6 else "#0b0b0b",
            )
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_title("Do the learners rank users the same way?", pad=22)
    subtitle(ax, f"Spearman correlation of predicted visit uplift, {out['n_test']:,} test users")
    fig.colorbar(im, ax=ax, fraction=0.04)
    save(fig, cfg.figures_dir, "learners_agreement")

    rows = out["by_baseline_decile"]
    d = np.array([r["decile"] for r in rows])
    fig, (ax0, ax1) = new_figures(2, w=11, h=4.0)
    ax0.fill_between(
        d,
        [r["ci_low"] * 100 for r in rows],
        [r["ci_high"] * 100 for r in rows],
        color=BASELINE,
        alpha=0.25,
        lw=0,
    )
    ax0.plot(
        d,
        [r["observed_uplift"] * 100 for r in rows],
        color="#0b0b0b",
        lw=2,
        label="observed (treated - control)",
    )
    for n in out["learners"]:
        ax0.plot(
            d,
            [r[f"pred_{n}"] * 100 for r in rows],
            color=LEARNER_COLORS[n],
            lw=1.4,
            marker="o",
            ms=4,
            label=learner_label(n),
        )
    ax0.set_xticks(d, [f"D{i}" for i in d])
    ax0.set_xlabel("decile of baseline visit propensity (D1 = highest)")
    ax0.set_ylabel("uplift (pp)")
    ax0.set_title("Uplift is concentrated in high-propensity users", pad=22)
    subtitle(ax0, "mean predicted uplift per learner vs observed (95% CI band)")
    ax0.legend(fontsize=7.5)
    ax1.bar(d, [r["mean_disagreement"] for r in rows], color=SERIES[6], width=0.65)
    ax1.set_xticks(d, [f"D{i}" for i in d])
    ax1.set_xlabel("decile of baseline visit propensity (D1 = highest)")
    ax1.set_ylabel("sd of percentile rank across learners")
    ax1.set_title("Rankings disagree most among mid- and high-propensity users", pad=22)
    subtitle(ax1, "per-user disagreement, averaged within decile")
    save(fig, cfg.figures_dir, "learners_by_baseline")


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
