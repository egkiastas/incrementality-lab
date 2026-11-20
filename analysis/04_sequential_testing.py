"""Sequential testing and power.

1. A/A tests on real control users with data arriving over time: how often does a t-test at every
   look declare a winner, and how often does the mSPRT?
2. The real experiment replayed in random arrival orders: when can the mSPRT call it?
3. Minimum detectable effect for the 85/15 split as run vs a 50/50 design.
"""

from typing import Any

import numpy as np

from incrementality import data, results
from incrementality.config import Config, from_args
from incrementality.plotting import BASELINE, INK_2, SERIES, new_figure, new_figures, save, subtitle
from incrementality.sequential import (
    empirical_power,
    geometric_looks,
    linear_looks,
    sequential_path,
    simulate_aa,
    time_to_decision,
)
from incrementality.stats import mde, required_n

POWER = 0.8


def main(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed + 3)
    ds = data.load(cfg.parquet, cfg.features, with_features=False)
    t = ds.treatment
    ctrl = t == 0
    share = float(t.mean())
    cupac = results.load(cfg.results_dir, "cuped")
    users_per_sim = min(cfg.seq_users_per_sim, int(ctrl.sum()))
    aa_looks = linear_looks(users_per_sim, cfg.seq_looks, cfg.first_look)
    replay_looks = geometric_looks(ds.n, cfg.replay_looks, cfg.first_look)
    out: dict[str, Any] = {"n_users": ds.n, "tau_rel": cfg.tau_rel, "replay_orderings": cfg.replay_orderings}
    paths = {}
    for name in ("visit", "conversion"):
        y = ds.outcome(name)
        p0 = float(y[ctrl].mean())
        tau2 = (cfg.tau_rel * p0) ** 2
        aa = simulate_aa(
            y[ctrl], cfg.seq_aa_sims, users_per_sim, cfg.treatment_share, aa_looks, cfg.alpha, tau2, rng
        )
        replay = time_to_decision(y, t, replay_looks, cfg.alpha, tau2, cfg.replay_orderings, rng)
        perm = np.random.default_rng(cfg.seed).permutation(ds.n)  # one replay for the figure
        paths[name] = sequential_path(y[perm], t[perm], replay_looks, cfg.alpha, tau2)

        sigma2 = p0 * (1 - p0)
        vr = cupac[name]["cupac"]["variance_reduction"]
        mde_rows = []
        for n_users in cfg.mde_grid:
            row = {"n_users": n_users}
            for label, s in (("85_15", share), ("50_50", 0.5)):
                m = mde(sigma2, n_users, s, cfg.alpha, POWER)
                row[f"mde_rel_{label}"] = m / p0
                row[f"mde_rel_{label}_cupac"] = m * np.sqrt(1 - vr) / p0
            mde_rows.append(row)
        effect = replay["full_sample_diff"]
        m_1m = mde(sigma2, 1_000_000, share, cfg.alpha, POWER)
        out[name] = {
            "control_rate": p0,
            "aa": aa,
            "replay": replay,
            "fixed_horizon_users_80pct_power": {
                label: required_n(sigma2, effect, s, cfg.alpha, POWER)
                for label, s in (("85_15", share), ("50_50", 0.5))
            },
            "variance_ratio_85_15_vs_50_50": 0.25 / (share * (1 - share)),
            "mde": mde_rows,
            "mde_check_1m_users": {
                "analytic_mde": m_1m,
                "simulated_power": empirical_power(
                    p0, m_1m, 1_000_000, share, cfg.alpha, cfg.power_sims, rng
                ),
                "n_sims": cfg.power_sims,
            },
        }
    results.save(cfg.results_dir, "sequential", out, cfg.seed)
    plot(cfg, out, paths)
    return out


def plot(cfg: Config, out: dict[str, Any], paths: dict[str, dict[str, np.ndarray]]) -> None:
    alpha = cfg.alpha
    fig, axes = new_figures(2, w=10, h=3.8)
    for ax, name in zip(axes, ("visit", "conversion"), strict=True):
        aa = out[name]["aa"]
        looks = np.arange(1, len(aa["looks"]) + 1)
        ax.plot(
            looks,
            np.array(aa["fpr_naive_by_look"]) * 100,
            color=SERIES[7],
            label="naive t-test at every look",
        )
        ax.plot(looks, np.array(aa["fpr_msprt_by_look"]) * 100, color=SERIES[0], label="mSPRT (always valid)")
        ax.axhline(alpha * 100, color=BASELINE, ls="--", lw=1.2)
        ax.text(
            looks[-1], alpha * 100, f"alpha = {alpha:.0%}", ha="right", va="bottom", fontsize=8.5, color=INK_2
        )
        ax.text(
            looks[-1],
            aa["fpr_naive_peeking"] * 100,
            f"{aa['fpr_naive_peeking']:.1%}",
            ha="right",
            va="bottom",
            fontsize=9,
            color=INK_2,
        )
        ax.set_xlabel("number of looks (peeks) so far")
        ax.set_ylabel("false-positive rate (%)")
        ax.set_ylim(0, max(aa["fpr_naive_peeking"] * 100 * 1.15, 10))
        ax.set_title(f"A/A ({name}): peeking manufactures winners", pad=22)
        subtitle(ax, f"{aa['n_sims']} A/A tests on real control users, {aa['looks'][-1]:,} users each")
    axes[0].legend(loc="upper left")
    save(fig, cfg.figures_dir, "sequential_peeking_fpr")

    fig, axes = new_figures(2, w=10, h=3.8)
    for ax, name in zip(axes, ("visit", "conversion"), strict=True):
        p = paths[name]
        n, ok = p["n"], np.isfinite(p["cs_low"])
        full = out[name]["replay"]["full_sample_diff"] * 100
        ax.fill_between(
            n[ok],
            p["cs_low"][ok] * 100,
            p["cs_high"][ok] * 100,
            color=SERIES[0],
            alpha=0.18,
            lw=0,
            label="95% confidence sequence (always valid)",
        )
        ax.plot(
            n[ok],
            p["ci_low"][ok] * 100,
            color=SERIES[1],
            lw=1.2,
            ls="--",
            label="95% fixed-horizon CI (only valid once)",
        )
        ax.plot(n[ok], p["ci_high"][ok] * 100, color=SERIES[1], lw=1.2, ls="--")
        ax.plot(n[ok], p["diff"][ok] * 100, color=SERIES[0], lw=1.6, label="running estimate")
        ax.axhline(0, color=BASELINE, lw=1)
        ax.axhline(full, color=INK_2, lw=0.8, ls=":")
        decided = out[name]["replay"]["median_users_msprt"]
        if decided:
            ax.axvline(decided, color=SERIES[2], lw=1.2)
            ax.text(decided, full * 2.2, "  median mSPRT decision", color=INK_2, fontsize=8.5, va="bottom")
        ax.set_xscale("log")
        ax.set_ylim(-abs(full) * 3, abs(full) * 6)
        ax.set_xlabel("users observed (log scale)")
        ax.set_ylabel("ATE (pp)")
        ax.set_title(f"{name.capitalize()}: the experiment replayed in arrival order", pad=22)
        subtitle(ax, f"dotted line = full-sample estimate ({full:.3f} pp)")
    axes[0].legend(loc="upper right", fontsize=8)
    save(fig, cfg.figures_dir, "sequential_confidence_sequence")

    fig, ax = new_figure(w=7, h=3.6)
    for i, name in enumerate(("visit", "conversion")):
        all_users = out[name]["replay"]["users_to_decision_msprt"]
        xs = np.sort([u for u in all_users if u])
        if xs.size == 0:
            continue
        ax.step(
            xs, np.arange(1, xs.size + 1) / len(all_users) * 100, where="post", color=SERIES[i], label=name
        )
        n_fixed = out[name]["fixed_horizon_users_80pct_power"]["85_15"]
        ax.axvline(n_fixed, color=SERIES[i], ls=":", lw=1.2)
        ax.text(n_fixed, 4 + 10 * i, f" fixed-horizon N\n (80% power), {name}", color=INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("users observed when mSPRT stops (log scale)")
    ax.set_ylabel("% of replays decided")
    ax.set_title("Time to decision on the real effect", pad=22)
    subtitle(ax, f"{out['replay_orderings']} random arrival orders of all {out['n_users']:,} users")
    ax.legend(loc="lower right")
    save(fig, cfg.figures_dir, "sequential_time_to_decision")

    fig, axes = new_figures(2, w=10, h=3.8)
    for ax, name in zip(axes, ("visit", "conversion"), strict=True):
        rows = out[name]["mde"]
        n_users = [r["n_users"] for r in rows]
        for key, colour, label, style in (
            ("mde_rel_85_15", SERIES[1], "85/15 (as run)", "-"),
            ("mde_rel_50_50", SERIES[0], "50/50", "-"),
            ("mde_rel_85_15_cupac", SERIES[2], "85/15 + CUPAC", "--"),
        ):
            ax.plot(
                n_users, [r[key] * 100 for r in rows], color=colour, marker="o", ms=5, ls=style, label=label
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("total users in experiment (log)")
        ax.set_ylabel("relative MDE (%, log)")
        ax.set_title(f"{name.capitalize()}: minimum detectable lift", pad=22)
        subtitle(ax, "alpha = 5%, power = 80%, two-sided")
    axes[0].legend()
    save(fig, cfg.figures_dir, "sequential_power_mde")


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
