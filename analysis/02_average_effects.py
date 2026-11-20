"""Average effects: difference in means (delta-method and bootstrap CIs), relative lift, and the
effect of actually seeing an ad (Wald/IV), next to the naive exposed-vs-unexposed comparison."""

from typing import Any

import numpy as np

from incrementality import data, results
from incrementality.config import Config, from_args
from incrementality.plotting import BASELINE, INK_2, SERIES, new_figure, new_figures, save, subtitle
from incrementality.stats import (
    bootstrap_binary,
    diff_from_moments,
    diff_in_means,
    lift_from_moments,
    moments_from_sums,
    percentile_ci,
    smd_table,
    wald_iv,
)


def complier_untreated_mean(y: np.ndarray, z: np.ndarray, d: np.ndarray) -> float:
    """E[Y(0) | would be exposed], from E[Y|Z=0] = pi E[Y(0)|C] + (1-pi) E[Y|Z=1,D=0]."""
    zb, db = z.astype(bool), d.astype(bool)
    pi = db[zb].mean()
    return float((y[~zb].mean() - (1 - pi) * y[zb & ~db].mean()) / pi)


def main(cfg: Config) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed + 1)
    ds = data.load(cfg.parquet, cfg.features)
    t = ds.treatment.astype(bool)
    exposed = ds.exposure.astype(bool)
    out: dict[str, Any] = {"n_treatment": int(t.sum()), "n_control": int((~t).sum())}
    for name in ("visit", "conversion"):
        y = ds.outcome(name)
        k1, n1, k0, n0 = int(y[t].sum()), int(t.sum()), int(y[~t].sum()), int((~t).sum())
        a1, a0 = moments_from_sums(n1, k1, k1), moments_from_sums(n0, k0, k0)
        dim = diff_from_moments(a1, a0, cfg.alpha)
        boot_diff, boot_lift = bootstrap_binary(k1, n1, k0, n0, cfg.n_bootstrap, rng)
        iv = wald_iv(y, ds.treatment, ds.exposure, cfg.alpha)
        naive = float(y[exposed].mean() - y[~exposed].mean())
        out[name] = {
            "rate_treated": a1[0],
            "rate_control": a0[0],
            "diff_in_means": dim.to_dict(),
            "diff_bootstrap_ci": percentile_ci(boot_diff, cfg.alpha),
            "relative_lift": lift_from_moments(a1, a0, cfg.alpha).to_dict(),
            "relative_lift_bootstrap_ci": percentile_ci(boot_lift, cfg.alpha),
            "incremental_per_million": dim.estimate * 1e6,
            "iv_effect_on_exposed": iv.to_dict(),
            "naive_exposed_vs_unexposed": naive,
            "naive_over_iv": naive / iv.estimate,
            "untreated_rate_of_exposed_users": complier_untreated_mean(y, ds.treatment, ds.exposure),
        }
    first_stage = diff_in_means(ds.exposure, ds.treatment)
    exposure_smd = smd_table(ds.X[t], ds.exposure[t])  # exposed vs unexposed, treated only
    arm_smd = smd_table(ds.X, ds.treatment)
    out["exposure"] = {
        "rate_treated": float(ds.exposure[t].mean()),
        "rate_control": float(ds.exposure[~t].mean()),
        "first_stage_f": float((first_stage.estimate / first_stage.se) ** 2),
        "smd_exposed_vs_unexposed": dict(zip(ds.feature_names, exposure_smd.tolist(), strict=True)),
        "smd_treated_vs_control": dict(zip(ds.feature_names, arm_smd.tolist(), strict=True)),
        "max_abs_smd_exposed_vs_unexposed": float(np.max(np.abs(exposure_smd))),
    }
    results.save(cfg.results_dir, "ate", out, cfg.seed)
    plot(cfg, out)
    return out


def plot(cfg: Config, out: dict[str, Any]) -> None:
    fig, axes = new_figures(2, w=10, h=3.6)
    for ax, name in zip(axes, ("visit", "conversion"), strict=True):
        r = out[name]
        bars = [
            ("ITT (assigned)", r["diff_in_means"]["estimate"], SERIES[0]),
            ("IV: effect on exposed", r["iv_effect_on_exposed"]["estimate"], SERIES[2]),
            ("Naive exposed vs not", r["naive_exposed_vs_unexposed"], SERIES[7]),
        ]
        for i, (_, v, c) in enumerate(bars):
            ax.barh(i, v * 100, color=c, height=0.6)
            ax.text(v * 100 * 1.08, i, f"  {v * 100:.2f} pp", va="center", color=INK_2, fontsize=9)
        iv = r["iv_effect_on_exposed"]
        err = [[(iv["estimate"] - iv["ci_low"]) * 100], [(iv["ci_high"] - iv["estimate"]) * 100]]
        ax.errorbar(iv["estimate"] * 100, 1, xerr=err, color="#0b0b0b", capsize=4, lw=1.2)
        ax.set_yticks(range(3), [b[0] for b in bars])
        ax.invert_yaxis()
        ax.set_xlim(0, max(b[1] for b in bars) * 140)
        ax.set_xlabel("effect on outcome rate (percentage points)")
        ax.set_title(f"{name.capitalize()}: three answers to 'what did the ad do?'", pad=22)
        subtitle(ax, "only ITT is guaranteed by randomization; IV needs the exclusion restriction")
    axes[1].set_yticklabels([])
    fig.subplots_adjust(wspace=0.08)
    save(fig, cfg.figures_dir, "ate_exposure_estimands")

    e = out["exposure"]
    feats = list(e["smd_exposed_vs_unexposed"])
    yy = np.arange(len(feats))
    fig, ax = new_figure(w=6.8, h=4.4)
    for vals, colour, label in (
        (e["smd_treated_vs_control"], SERIES[0], "treated vs control (randomized)"),
        (e["smd_exposed_vs_unexposed"], SERIES[7], "exposed vs unexposed (treated)"),
    ):
        ax.scatter(
            list(vals.values()),
            yy,
            color=colour,
            s=36,
            label=label,
            zorder=3,
            edgecolor="white",
            linewidth=1.2,
        )
    ax.axvline(0, color=BASELINE, lw=1)
    ax.axvspan(-0.1, 0.1, color="#eef4fc", zorder=0)
    ax.set_yticks(yy, feats)
    ax.invert_yaxis()
    ax.set_xlabel("standardized mean difference")
    ax.set_title("Exposure is a selected, not a randomized, group", pad=22)
    subtitle(ax, "shaded band = |SMD| < 0.1")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, cfg.figures_dir, "ate_exposure_selection")


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
