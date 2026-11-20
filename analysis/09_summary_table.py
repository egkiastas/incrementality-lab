"""Collect the headline numbers from report/results/*.json into report/results/summary.md.

The README table is copied from this file, so every number in it traces back to a run.
"""

from incrementality import results
from incrementality.config import Config, from_args


def pp(x: float, d: int = 3) -> str:
    return f"{x * 100:.{d}f} pp"


def ci(lo: float, hi: float, d: int = 3) -> str:
    return f"[{lo * 100:.{d}f}, {hi * 100:.{d}f}]"


def main(cfg: Config) -> str:
    r = {n: results.load(cfg.results_dir, n) for n in ("ate", "cuped", "sequential", "uplift", "policy")}
    ate, cup, seq, up, pol = r["ate"], r["cuped"], r["sequential"], r["uplift"], r["policy"]
    rows = []
    for o in ("visit", "conversion"):
        naive = ate[o]["diff_in_means"]
        dr = cup[o]["aipw_estimated_propensity"]
        lift = dr["relative_lift"]
        rows.append(
            (
                f"{o.capitalize()} effect",
                f"{pp(naive['estimate'])} {ci(naive['ci_low'], naive['ci_high'])}",
                f"{pp(dr['estimate'])} {ci(dr['ci_low'], dr['ci_high'])} (AIPW)",
            )
        )
        rows.append(
            (
                f"{o.capitalize()} relative lift",
                f"{ate[o]['relative_lift']['estimate']:.1%}",
                f"{lift['estimate']:.1%} {ci(lift['ci_low'], lift['ci_high'], 1)}",
            )
        )
    iv = ate["visit"]["iv_effect_on_exposed"]
    rows.append(
        (
            "Effect of seeing an ad (visits)",
            f"{pp(ate['visit']['naive_exposed_vs_unexposed'], 1)} exposed vs not",
            f"{pp(iv['estimate'], 1)} {ci(iv['ci_low'], iv['ci_high'], 1)} (IV)",
        )
    )
    vr = cup["visit"]["cupac"]["variance_reduction"]
    rows.append(
        (
            "Variance of the visit ATE",
            "difference in means",
            f"-{vr:.0%} with CUPAC (= {1 / (1 - vr):.2f}x users)",
        )
    )
    aa = seq["visit"]["aa"]
    rows.append(
        (
            "A/A false positives, 100 peeks",
            f"{aa['fpr_naive_peeking']:.1%} (t-test)",
            f"{aa['fpr_msprt']:.1%} (mSPRT)",
        )
    )
    rep = seq["visit"]["replay"]
    rows.append(
        (
            "Users to call the visit effect",
            f"{seq['visit']['fixed_horizon_users_80pct_power']['85_15']:,.0f} (fixed horizon, effect known)",
            f"{rep['median_users_msprt']:,.0f} median (mSPRT)",
        )
    )
    best = up["visit"]["best_on_val"]
    ev = up["visit"]["test"]
    rows.append(
        (
            "Test-set Qini, visits",
            f"response {ev['response_model']['qini']:.3f}, random {ev['random']['qini']:.3f}",
            f"{best} {ev[best]['qini']:.3f} [{ev[best]['qini_ci'][0]:.3f}, {ev[best]['qini_ci'][1]:.3f}]",
        )
    )
    for row in pol["visit"]["budget_table"]:
        b = row[pol["visit"]["best_model"]]
        rand, resp = row["random"]["incremental"], row["response_model"]["incremental"]
        share = b["share_of_total_incremental"]
        rows.append(
            (
                f"Extra visits per 1M users, {row['budget']:.0%} budget",
                f"random {rand:,.0f}, response {resp:,.0f}",
                f"{b['incremental']:,.0f} [{b['ci'][0]:,.0f}, {b['ci'][1]:,.0f}] ({share:.0%} of all)",
            )
        )
    table = "| | Naive / baseline | This analysis |\n|---|---|---|\n"
    table += "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows) + "\n"
    (cfg.results_dir / "summary.md").write_text(table)
    print(table)
    return table


if __name__ == "__main__":
    main(from_args(__doc__ or "")[0])
