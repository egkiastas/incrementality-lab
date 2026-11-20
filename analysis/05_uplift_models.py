"""Heterogeneous effects: S/T/X/DR learners and a causal forest, evaluated on held-out users.

One stratified split of every user: 60% train / 10% validation / 30% test. Models train on a
stratified subsample of the training part (to fit a 16 GB laptop), the best learner is picked on
validation Qini, and everything reported is on the test set. Conversions are rare, so the
visit-trained models are also scored against conversions.
"""

import time
from typing import Any

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from incrementality import data, results
from incrementality.config import Config, from_args
from incrementality.plotting import (
    BASELINE,
    INK_2,
    LEARNER_COLORS,
    learner_label,
    new_figure,
    save,
    subtitle,
)
from incrementality.uplift import evaluate, fit_predict, qini_auc_score

OUTCOMES = ("visit", "conversion")


def main(cfg: Config, reuse_predictions: bool = False) -> dict[str, Any]:
    ds = data.load(cfg.parquet, cfg.features)
    cells = data.strata(ds)
    splits = data.train_val_test_split(cells, cfg.test_frac, cfg.val_frac, cfg.seed + 4)
    rng = np.random.default_rng(cfg.seed + 5)
    train_idx = data.stratified_subsample(splits["train"], cells, cfg.train_rows, rng)
    forest_idx = data.stratified_subsample(train_idx, cells, cfg.forest_rows, rng)
    propensity = float(ds.treatment[train_idx].mean())
    eval_idx = np.r_[splits["val"], splits["test"]]
    pred_path = cfg.artifacts_dir / "uplift_predictions.parquet"
    models = [*cfg.learners, "response_model"]

    if reuse_predictions and pred_path.exists():
        saved = pl.read_parquet(pred_path)
        assert np.array_equal(saved["row"].to_numpy(), eval_idx), "saved predictions don't match split"
        preds = {c: saved[c].to_numpy() for c in saved.columns if "__" in c}
        fit_seconds = results.load(cfg.results_dir, "uplift").get("fit_seconds", {})
    else:
        preds, fit_seconds = {}, {}
        for outcome in OUTCOMES:
            y = ds.outcome(outcome)
            for i, name in enumerate(models):
                idx = forest_idx if name == "causal_forest" else train_idx
                start = time.perf_counter()
                preds[f"{outcome}__{name}"] = fit_predict(
                    name,
                    ds.X[idx],
                    ds.treatment[idx],
                    y[idx],
                    ds.X[eval_idx],
                    cfg.lgbm,
                    cfg.lgbm_cate,
                    propensity,
                    cfg.seed + 100 * i,
                    cfg.forest_trees,
                    cfg.forest_min_leaf,
                ).astype(np.float32)
                fit_seconds[f"{outcome}__{name}"] = time.perf_counter() - start
                print(f"{outcome} {name}: {fit_seconds[f'{outcome}__{name}']:.0f}s")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        split_flag = np.r_[np.zeros(splits["val"].size, np.int8), np.ones(splits["test"].size, np.int8)]
        pl.DataFrame({"row": eval_idx, "split": split_flag, **preds}).write_parquet(pred_path)

    nv = splits["val"].size
    val_rows, test_rows = eval_idx[:nv], eval_idx[nv:]
    out: dict[str, Any] = {
        "split_sizes": {k: int(v.size) for k, v in splits.items()},
        "train_rows": int(train_idx.size),
        "forest_rows": int(forest_idx.size),
        "fit_seconds": fit_seconds,
    }
    for outcome in OUTCOMES:
        y = ds.outcome(outcome)
        yv, tv = y[val_rows], ds.treatment[val_rows]
        yt, tt = y[test_rows], ds.treatment[test_rows]
        candidates = {name: f"{outcome}__{name}" for name in models}
        if outcome == "conversion":
            candidates |= {f"visit_model__{n}": f"visit__{n}" for n in cfg.learners}
        scores = {name: preds[col][nv:] for name, col in candidates.items()}
        scores["random"] = np.random.default_rng(cfg.seed + 6).random(test_rows.size)
        val_qini = {name: qini_auc_score(yv, tv, preds[col][:nv]) for name, col in candidates.items()}
        names = list(scores)
        evals = Parallel(n_jobs=min(4, len(names)))(
            delayed(evaluate)(
                yt,
                tt,
                scores[n],
                list(cfg.budgets),
                cfg.uplift_bootstrap,
                np.random.default_rng(cfg.seed + 7 + i),
            )
            for i, n in enumerate(names)
        )
        out[outcome] = {
            "test_ate": float(yt[tt == 1].mean() - yt[tt == 0].mean()),
            "val_qini": val_qini,
            "best_on_val": max((k for k in val_qini if k != "response_model"), key=val_qini.__getitem__),
            "test": dict(zip(names, evals, strict=True)),
            "prediction_sd": {n: float(np.std(s)) for n, s in scores.items() if n != "random"},
        }
    results.save(cfg.results_dir, "uplift", out, cfg.seed)
    plot(cfg, out)
    return out


def plot(cfg: Config, out: dict[str, Any]) -> None:
    for outcome in OUTCOMES:
        r = out[outcome]
        ev, best = r["test"], r["best_on_val"]
        fig, ax = new_figure(w=7.5, h=4.6)
        for name in [*cfg.learners, "response_model", "random"]:
            e = ev[name]
            ax.plot(
                np.array(e["curve_x"]) * 100,
                np.array(e["curve_qini"]) * 1000,
                color=LEARNER_COLORS[name],
                ls="--" if name in ("random", "response_model") else "-",
                lw=2.4 if name == best else 1.6,
                label=f"{learner_label(name)}  Qini={e['qini']:.3f}",
            )
        ax.set_xlabel("% of users targeted (ranked by score)")
        ax.set_ylabel("incremental outcomes per 1,000 users")
        ax.set_title(f"Qini curves on the test set - {outcome}", pad=22)
        subtitle(
            ax, f"{out['split_sizes']['test']:,} held-out users; best on validation: {learner_label(best)}"
        )
        ax.legend(loc="lower right", fontsize=8)
        save(fig, cfg.figures_dir, f"uplift_qini_{outcome}")

        dec = ev[best]["deciles"]
        b = np.array([d["bin"] for d in dec])
        obs = np.array([d["observed"] for d in dec]) * 100
        lo = np.array([d["ci_low"] for d in dec]) * 100
        hi = np.array([d["ci_high"] for d in dec]) * 100
        fig, ax = new_figure(w=7.5, h=4.0)
        ax.bar(
            b,
            obs,
            color=LEARNER_COLORS.get(best.split("__")[-1], "#2a78d6"),
            width=0.65,
            label="observed uplift (test)",
        )
        ax.errorbar(b, obs, yerr=[obs - lo, hi - obs], fmt="none", color="#0b0b0b", capsize=3, lw=1)
        ax.scatter(
            b,
            [d["predicted"] * 100 for d in dec],
            color="#0b0b0b",
            marker="D",
            s=28,
            zorder=4,
            label="mean predicted uplift",
        )
        ax.axhline(r["test_ate"] * 100, color=BASELINE, ls="--", lw=1.2)
        ax.text(10.4, r["test_ate"] * 100, "ATE", color=INK_2, fontsize=8.5, va="center")
        ax.set_xticks(b, [f"D{i}" for i in b])
        ax.set_xlabel("decile of predicted uplift (D1 = highest)")
        ax.set_ylabel("uplift (pp)")
        ax.set_title(f"Uplift by decile - {outcome} ({learner_label(best)})", pad=22)
        subtitle(ax, "bars: treated minus control rate within decile, 95% CI")
        ax.legend(loc="upper right")
        save(fig, cfg.figures_dir, f"uplift_deciles_{outcome}")

    ev = out["visit"]["test"]
    fig, ax = new_figure(w=6.2, h=5.0)
    every = []
    for name in cfg.learners:
        p = np.array([x["predicted"] for x in ev[name]["deciles"]]) * 100
        o = np.array([x["observed"] for x in ev[name]["deciles"]]) * 100
        every += [*p, *o]
        ax.plot(
            p,
            o,
            marker="o",
            ms=5,
            lw=1.2,
            color=LEARNER_COLORS[name],
            label=f"{learner_label(name)} (slope {ev[name]['calibration']['slope']:.2f})",
        )
    lim = (min(every) - 0.2, max(every) + 0.2)
    ax.plot(lim, lim, color=BASELINE, ls="--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("mean predicted uplift in decile (pp)")
    ax.set_ylabel("observed uplift in decile (pp)")
    ax.set_title("Uplift calibration - visit", pad=22)
    subtitle(ax, "dashed = perfect calibration; slope from WLS of observed on predicted")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, cfg.figures_dir, "uplift_calibration_visit")


if __name__ == "__main__":
    cfg, args = from_args(__doc__ or "", reuse_flag=True)
    main(cfg, reuse_predictions=args.reuse_predictions)
