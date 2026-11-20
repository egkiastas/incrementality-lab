"""Whom to target under a budget, evaluated on the held-out test users.

Incremental outcomes are the mean doubly robust (AIPW) score of the targeted users, with the
nuisance models fit on the training split only, so the numbers hold up despite the imbalance
found in step 03. The plain within-group difference (scikit-uplift's uplift@k) is kept next to
it for comparison.
"""

from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps

from incrementality import data, results
from incrementality.adjustment import dr_scores
from incrementality.config import Config, from_args
from incrementality.plotting import BASELINE, INK_2, LEARNER_COLORS, SERIES, new_figures, save, subtitle
from incrementality.stats import diff_in_means, z_crit
from incrementality.uplift import random_tiebreak_order


def mean_ci(x: np.ndarray) -> tuple[float, float, float]:
    m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))
    return m, m - z_crit(0.05) * se, m + z_crit(0.05) * se


def segment(y: np.ndarray, t: np.ndarray, phi: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    m, lo, hi = mean_ci(phi[mask])
    se = (hi - lo) / (2 * z_crit(0.05))
    naive = diff_in_means(y[mask], t[mask])
    return {
        "share_of_users": float(mask.mean()),
        "dr_uplift": m,
        "dr_ci": [lo, hi],
        "dr_p_value": float(2 * sps.norm.sf(abs(m) / se)) if se > 0 else 1.0,
        "naive_uplift": naive.estimate,
        "control_rate": float(y[mask & (t == 0)].mean()),
    }


def top(score: np.ndarray, k: float) -> np.ndarray:
    mask = np.zeros(score.size, bool)
    mask[random_tiebreak_order(score)[: int(k * score.size)]] = True
    return mask


def main(cfg: Config) -> dict[str, Any]:
    uplift = results.load(cfg.results_dir, "uplift")
    preds = pl.read_parquet(cfg.artifacts_dir / "uplift_predictions.parquet")
    ds = data.load(cfg.parquet, cfg.features)
    split = preds["split"].to_numpy()
    test_rows = preds["row"].to_numpy()[split == 1]
    cells = data.strata(ds)
    splits = data.train_val_test_split(cells, cfg.test_frac, cfg.val_frac, cfg.seed + 4)
    rng = np.random.default_rng(cfg.seed + 8)
    nuisance_rows = data.stratified_subsample(splits["train"], cells, cfg.train_rows, rng)
    t, X_test = ds.treatment[test_rows], ds.X[test_rows]
    P = cfg.population
    out: dict[str, Any] = {"population": P, "budgets": list(cfg.budgets)}
    for outcome in ("visit", "conversion"):
        y_all = ds.outcome(outcome)
        y = y_all[test_rows]
        phi1, phi0, _ = dr_scores(
            ds.X[nuisance_rows],
            ds.treatment[nuisance_rows],
            y_all[nuisance_rows],
            X_test,
            t,
            y,
            cfg.lgbm,
            cfg.seed + 9,
        )
        phi = phi1 - phi0
        ate = mean_ci(phi)
        naive_ate = diff_in_means(y, t)
        ev = uplift[outcome]["test"]
        best = uplift[outcome]["best_on_val"]
        scores = {
            k: preds[f"{outcome}__{k}"].to_numpy()[split == 1] for k in [*cfg.learners, "response_model"]
        }
        if outcome == "conversion":
            scores |= {f"visit_model__{k}": preds[f"visit__{k}"].to_numpy()[split == 1] for k in cfg.learners}

        table = []
        for budget in cfg.budgets:
            key = f"{budget:.2f}"
            row: dict[str, Any] = {"budget": budget}
            for s in ["random", *scores]:
                if s == "random":
                    m, lo, hi = ate
                    naive = naive_ate.estimate
                else:
                    m, lo, hi = mean_ci(phi[top(scores[s], budget)])
                    naive = ev[s]["uplift_at_k"][key]
                row[s] = {
                    "incremental": budget * P * m,
                    "ci": [budget * P * lo, budget * P * hi],
                    "share_of_total_incremental": budget * m / ate[0],
                    "incremental_naive": budget * P * naive,
                }
            row["lift_vs_random"] = row[best]["incremental"] / row["random"]["incremental"]
            row["lift_vs_response"] = row[best]["incremental"] / row["response_model"]["incremental"]
            table.append(row)

        resp, best_score = scores["response_model"], scores[best]
        decile = np.empty(resp.size, np.int64)
        for d, idx in enumerate(np.array_split(random_tiebreak_order(resp), 10)):
            decile[idx] = d
        top_resp, top_upl = top(resp, 0.1), top(best_score, 0.1)

        # "Do not disturb": thresholds fixed on validation predictions, tested on test.
        val_col = (
            f"visit__{best.split('__')[1]}" if best.startswith("visit_model__") else f"{outcome}__{best}"
        )
        val_scores = preds[val_col].to_numpy()[split == 0]
        candidates = {"predicted_negative": best_score < 0}
        for q in (0.05, 0.10):
            candidates[f"bottom_{int(q * 100)}pct"] = best_score <= float(np.quantile(val_scores, q))
        dnd = {k: segment(y, t, phi, m) for k, m in candidates.items() if m.sum() > 1000}

        out[outcome] = {
            "best_model": best,
            "test_ate_dr": {"estimate": ate[0], "ci": [ate[1], ate[2]]},
            "test_ate_naive": naive_ate.to_dict(),
            "budget_table": table,
            "response_model": {
                "deciles": [segment(y, t, phi, decile == d) | {"decile": d + 1} for d in range(10)],
                "spearman_with_uplift_score": float(
                    sps.spearmanr(resp[:1_000_000], best_score[:1_000_000]).statistic
                ),
                "top_decile_overlap": float((top_resp & top_upl).sum() / top_resp.sum()),
                "top_response_decile": segment(y, t, phi, top_resp),
                "top_uplift_decile": segment(y, t, phi, top_upl),
            },
            "do_not_disturb": dnd,
        }
    results.save(cfg.results_dir, "policy", out, cfg.seed)
    plot(cfg, out)
    return out


def plot(cfg: Config, out: dict[str, Any]) -> None:
    P = out["population"]
    fig, axes = new_figures(2, w=11, h=4.2)
    for ax, outcome in zip(axes, ("visit", "conversion"), strict=True):
        r = out[outcome]
        strategies = [
            (r["best_model"], "Uplift model (best on val)", SERIES[2]),
            ("response_model", "Response model (ignores T)", LEARNER_COLORS["response_model"]),
            ("random", "Random targeting", BASELINE),
        ]
        x = np.arange(len(r["budget_table"]))
        for i, (key, label, colour) in enumerate(strategies):
            v = np.array([row[key]["incremental"] for row in r["budget_table"]])
            lo = np.array([row[key]["ci"][0] for row in r["budget_table"]])
            hi = np.array([row[key]["ci"][1] for row in r["budget_table"]])
            ax.bar(x + (i - 1) * 0.26, v, width=0.23, color=colour, label=label)
            ax.errorbar(
                x + (i - 1) * 0.26, v, yerr=[v - lo, hi - v], fmt="none", color="#0b0b0b", lw=1, capsize=2.5
            )
        everyone = r["test_ate_dr"]["estimate"] * P
        ax.axhline(everyone, color=INK_2, lw=0.8, ls=":")
        ax.text(x[-1] + 0.45, everyone, "treat everyone", color=INK_2, fontsize=8, ha="right", va="bottom")
        ax.set_xticks(x, [f"{row['budget']:.0%}" for row in r["budget_table"]])
        ax.set_xlabel("share of users we can target (budget)")
        ax.set_ylabel(f"incremental {outcome}s per {P:,} users")
        ax.set_title(f"Incremental {outcome}s captured under a budget", pad=22)
        subtitle(ax, "test set, doubly robust estimate, 95% CI")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.17), ncol=3, fontsize=9)
    save(fig, cfg.figures_dir, "policy_budget_curve")

    diag = out["visit"]["response_model"]
    deciles = diag["deciles"]
    b = np.array([d["decile"] for d in deciles])
    fig, (ax0, ax1) = new_figures(2, w=11, h=3.8)
    ax0.bar(b, [d["control_rate"] * 100 for d in deciles], color=LEARNER_COLORS["response_model"], width=0.65)
    ax0.set_xticks(b, [f"D{i}" for i in b])
    ax0.set_xlabel("decile of response-model score (D1 = most likely to visit)")
    ax0.set_ylabel("visit rate WITHOUT ads (control, %)")
    ax0.set_title("A response model finds people who visit anyway", pad=22)
    subtitle(ax0, "control-group visit rate per decile, test set")
    bars = [
        ("Uplift model\ntop 10%", diag["top_uplift_decile"], SERIES[2]),
        ("Response model\ntop 10%", diag["top_response_decile"], LEARNER_COLORS["response_model"]),
    ]
    for i, (_, d, colour) in enumerate(bars):
        anyway, caused = d["control_rate"] * 1000, d["dr_uplift"] * 1000
        ax1.barh(i, anyway, color=BASELINE, height=0.55, label="would visit anyway" if i == 0 else None)
        ax1.barh(
            i, caused, left=anyway, color=colour, height=0.55, label="caused by the ad" if i == 0 else None
        )
        ax1.text(anyway + caused, i, f"  +{caused:.0f} caused", va="center", fontsize=9, color=INK_2)
    ax1.set_yticks(range(2), [label for label, _, _ in bars])
    ax1.set_xlim(0, max(d["control_rate"] + d["dr_uplift"] for _, d, _ in bars) * 1300)
    ax1.set_xlabel("visits per 1,000 targeted users")
    ax1.set_title("More visits, fewer of them caused by the ad", pad=22)
    subtitle(ax1, "top 10% of test users by each score; doubly robust uplift")
    ax1.legend(loc="lower right", fontsize=8.5)
    save(fig, cfg.figures_dir, "policy_response_model_diagnosis")


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
