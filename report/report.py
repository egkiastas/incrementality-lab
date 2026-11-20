# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # What did the ads do, how soon could we tell, and who should see them?
# ### Notes on the Criteo Uplift v2.1 experiment
#
# Vangelis Gkiastas · [github.com/egkiastas/incrementality-lab](https://github.com/egkiastas/incrementality-lab)
#
# *All numbers below are read from `report/results/*.json`, written by the scripts in `analysis/`.*

# %%
import json
from pathlib import Path

from IPython.display import Image, Markdown, display

R, F = Path("results"), Path("figures")
H, A, C, S, U, P, L, V = (
    json.loads((R / f"{n}.json").read_text())
    for n in ["health", "ate", "cuped", "sequential", "uplift", "policy", "learners", "synthetic_check"]
)
LABELS = {
    "s_learner": "S-learner",
    "t_learner": "T-learner",
    "x_learner": "X-learner",
    "dr_learner": "DR-learner",
    "causal_forest": "Causal forest",
    "response_model": "Response model",
    "random": "Random",
    "oracle": "Oracle (true CATE)",
}


def lab(k):
    return (
        LABELS[k.split("__")[1]] + " (visit-trained)" if k.startswith("visit_model__") else LABELS.get(k, k)
    )


def md(s):
    display(Markdown(s))


def fig(name, width=900):
    display(Image(filename=str(F / f"{name}.png"), width=width))


def pp(x, d=3):
    return f"{x * 100:.{d}f} pp"


def pc(x, d=1):
    return f"{x * 100:.{d}f}%"


def ci(lo, hi, d=3, scale=100):
    return f"[{lo * scale:.{d}f}, {hi * scale:.{d}f}]"


av, ac = A["visit"], A["conversion"]
cv, cc = C["visit"], C["conversion"]
dr_v, dr_c = cv["aipw_estimated_propensity"], cc["aipw_estimated_propensity"]
pv = P["visit"]
best_v = pv["best_model"]
b10, b50 = pv["budget_table"][0], pv["budget_table"][-1]

# %%
md(f"""
## Summary for the product team

**The ads work.** Site visits go up by about {pc(dr_v["relative_lift"]["estimate"], 0)}
(+{dr_v["estimate"] * 100:.2f} visits per 100 users) and purchases by about
{pc(dr_c["relative_lift"]["estimate"], 0)} (+{dr_c["estimate"] * 100:.2f} per 100 users).

**The usual dashboard number is too high.** A straight treated-vs-held-out comparison says
+{pc(av["relative_lift"]["estimate"], 0)} on visits. Part of that comes from how the dataset was put
together, not from the ads (section 3). Report the adjusted figure.

**Most of the value sits with a small group.** Showing ads to the 10% of users the uplift model ranks
highest delivers {pc(b10[best_v]["share_of_total_incremental"], 0)} of the extra visits that showing them
to everyone would: {b10[best_v]["incremental"]:,.0f} per million users, against
{b10["response_model"]["incremental"]:,.0f} if we pick "people likely to visit" and
{b10["random"]["incremental"]:,.0f} at random. Above roughly 20% of users, the simple
"likely to visit" ranking does just as well.

**Nobody is put off by the ads,** but half of the users barely respond. The lower-ranked half produces
only about {pc(1 - b50[best_v]["share_of_total_incremental"], 0)} of the extra visits, which makes it the
obvious place to save budget.

**Purchases** are too rare (about 3 per 1,000 users) to target better than the simple model does.

**On running tests:** checking a normal A/B test after every batch of users and stopping when it looks
good produced a false win in {pc(S["visit"]["aa"]["fpr_naive_peeking"], 0)} of no-effect tests. The
always-valid test stayed at {pc(S["visit"]["aa"]["fpr_msprt"], 0)}.

**Recommendations:**
1. Keep the channel and concentrate spend on the top 10-20% by predicted uplift.
2. Stop spending on the lowest-ranked half.
3. Report the adjusted lift.
4. Use always-valid monitoring on live tests.
5. Consider a 50/50 split next time: it needs about half the users of 85/15 for the same certainty.

The public data was altered on purpose to hide the original campaign size, so the exact percentages
won't transfer. The direction and the targeting advice should.
""")

# %%
md(f"""
## 1. Data

Criteo Uplift Prediction Dataset v2.1 (Diemert et al., 2021): {H["integrity"]["n_rows"]:,} users,
{H["srm"]["n_treatment"]:,} of them eligible to see ads. The outcomes are a visit to the
advertiser's site ({pc(H["integrity"]["visit_rate"], 2)}) and a purchase
({pc(H["integrity"]["conversion_rate"], 3)}). Only {pc(H["integrity"]["exposure_rate_treated"], 1)} of
eligible users actually saw an ad, and control users never did. Twelve anonymised features were
logged before assignment.

Two details from Criteo's paper end up mattering. The file pools several tests, each re-sampled to 85%
treated, and negatives were down-sampled non-uniformly "so that the original incrementality level
cannot be deduced".
""")

# %%
s, p = H["srm"], H["propensity_check"]
md(f"""
## 2. Is the randomization clean?

Every standard check passes. The sample-ratio test gives p = {s["p_value"]:.3f}, the largest
standardized mean difference is {H["max_abs_smd"]:.3f}, and 1,000 A/A splits of the control group
produce false positives at {pc(H["aa"]["visit"]["fpr"])} (visits) and {pc(H["aa"]["conversion"]["fpr"])}
(conversions).

At this sample size, though, passing a threshold isn't the same as being random. The SE of an SMD
under randomization is {H["smd_se_under_randomization"]:.5f}, and
{H["n_features_significant_bonferroni"]} of 12 features are significantly imbalanced after Bonferroni.
A LightGBM classifier predicts treatment with AUC {p["auc"]:.4f} {ci(*p["auc_ci"], d=4, scale=1)}. The
treated share is also suspiciously exact: a genuine Bernoulli(0.85) assignment lands this close to 85%
with probability {s["prob_share_this_close_under_bernoulli"]:.4f}.
""")
fig("health_balance_smd", 650)
fig("health_aa_pvalues", 850)

# %%
rows = []
for k, name in [
    ("diff_in_means", "Difference in means"),
    ("cuped_linear", "CUPED (12 features)"),
    ("cupac", "CUPAC (cross-fitted GBM)"),
    ("post_stratified", "Post-stratified (20 strata)"),
    ("aipw_known_propensity", "AIPW, design propensity"),
    ("aipw_estimated_propensity", "AIPW, estimated propensity"),
]:
    cells = []
    for o in ("visit", "conversion"):
        e = C[o][k]
        shift = f" ({e['shift_in_se']:+.0f} SE)" if "shift_in_se" in e else ""
        cells.append(f"{e['estimate'] * 100:.3f} {ci(e['ci_low'], e['ci_high'])}{shift}")
    rows.append(f"| {name} | " + " | ".join(cells) + " |")
st = cv["strata"]
md(
    f"""
## 3. The average effect

The plain difference in means puts the visit effect at {pp(av["diff_in_means"]["estimate"])}
{ci(av["diff_in_means"]["ci_low"], av["diff_in_means"]["ci_high"])}
({pc(av["relative_lift"]["estimate"])} relative) and the conversion effect at
{pp(ac["diff_in_means"]["estimate"])} ({pc(ac["relative_lift"]["estimate"])}). The exact binomial
bootstrap gives {ci(*av["diff_bootstrap_ci"])} for visits, the same as the delta method to the third
decimal.

On a clean RCT, adjusting for pre-treatment features should leave the estimate where it is and only
narrow the interval. Here it moves it a long way:

| Estimator | Visits, pp [95% CI] (shift) | Conversions, pp [95% CI] (shift) |
|---|---|---|
"""
    + "\n".join(rows)
    + f"""

The figure below shows why. Sorting users by a treatment-blind prediction of visiting, the treated share
rises from {pc(st[0]["treatment_share"], 1)} in the lowest ventile to {pc(st[-1]["treatment_share"], 1)}
in the highest, and those are the users with a {pc(st[-1]["control_rate"], 1)} visit rate even without
ads. Part of the raw difference is just measuring who ended up treated. Down-sampling negatives and
then forcing each sub-test back to 85% would produce this.

The adjustment itself is not the problem. On 500 fake 85/15 splits of control users, the mean CUPAC
estimate was {cv["aa_check"]["mean_estimate_cupac"] * 100:.4f} pp (MC SE
{cv["aa_check"]["mc_se_cupac"] * 100:.4f}) with a {pc(cv["aa_check"]["fpr_cupac"])} false-positive rate.

I use AIPW with an estimated propensity as the headline: **{pp(dr_v["estimate"])}
{ci(dr_v["ci_low"], dr_v["ci_high"])}** for visits ({pc(dr_v["relative_lift"]["estimate"])}) and
{pp(dr_c["estimate"], 4)} for conversions ({pc(dr_c["relative_lift"]["estimate"])}). It is consistent
if either the outcome or the propensity model is right. The estimators still disagree among themselves,
so the honest answer for visits is roughly 0.63-0.75 pp. These are effects in the re-sampled file.
"""
)
fig("cuped_imbalance_mechanism")
fig("cuped_estimators")

# %%
iv_v, iv_c, ex = av["iv_effect_on_exposed"], ac["iv_effect_on_exposed"], A["exposure"]
md(f"""
### What does actually seeing an ad do?

Because control users can never be exposed, assignment is an instrument for exposure (first-stage
F = {ex["first_stage_f"]:,.0f}). Under the exclusion restriction, the Wald ratio gives the effect on
users who were exposed: **{pp(iv_v["estimate"], 1)} {ci(iv_v["ci_low"], iv_v["ci_high"], 1)}** on
visits and {pp(iv_c["estimate"], 2)} on conversions. Comparing exposed with unexposed users instead
gives {av["naive_over_iv"]:.2f}x and {ac["naive_over_iv"]:.2f}x those numbers, because exposed users
are a selected group: their features differ by up to {ex["max_abs_smd_exposed_vs_unexposed"]:.2f} SDs,
and they would have visited at {pc(av["untreated_rate_of_exposed_users"], 1)} with no ad at all, against
{pc(av["rate_control"], 1)} for an average control user. The IV number inherits the imbalance above
(it is ITT divided by the exposure rate) and only describes the kind of user who gets exposed.
""")
fig("ate_exposure_estimands")
fig("ate_exposure_selection", 650)

# %%
md(f"""
## 4. Variance reduction

There is no pre-period outcome, so the features stand in for the CUPED covariate. CUPAC replaces
them with a single out-of-fold, treatment-blind LightGBM prediction (AUC {cv["cupac_model_auc"]:.3f}
for visits). On visits CUPED removes {pc(cv["cuped_linear"]["variance_reduction"], 0)} of the variance
and CUPAC {pc(cv["cupac"]["variance_reduction"], 0)}, which is like running the plain test with
{1 / (1 - cv["cupac"]["variance_reduction"]):.2f}x the users. On conversions the gains are
{pc(cc["cuped_linear"]["variance_reduction"], 0)} and {pc(cc["cupac"]["variance_reduction"], 0)}.
""")

# %%
sv, sc = S["visit"], S["conversion"]
md(f"""
## 5. How soon can we tell?

To simulate data arriving over time, each of 1,000 A/A experiments takes 500,000 control users in
random order, gives them fake 85/15 labels and looks at the data 100 times. A t-test at every look
declares a winner in {pc(sv["aa"]["fpr_naive_peeking"])} of runs for visits and
{pc(sc["aa"]["fpr_naive_peeking"])} for conversions. A single test at the end is at
{pc(sv["aa"]["fpr_fixed_horizon"])} and {pc(sc["aa"]["fpr_fixed_horizon"])}. The mixture SPRT
(τ = 10% of the baseline rate, decided before looking) stays at {pc(sv["aa"]["fpr_msprt"])} and
{pc(sc["aa"]["fpr_msprt"])}.

Replaying the real experiment in 100 random arrival orders, the mSPRT called the visit effect after a
median of {sv["replay"]["median_users_msprt"]:,.0f} users (P10-P90 {sv["replay"]["p10_users_msprt"]:,.0f}
to {sv["replay"]["p90_users_msprt"]:,.0f}) and the conversion effect after
{sc["replay"]["median_users_msprt"]:,.0f}, always with the right sign. A fixed-horizon test with 80%
power, sized as if you already knew the effect, would need
{sv["fixed_horizon_users_80pct_power"]["85_15"]:,.0f} and {sc["fixed_horizon_users_80pct_power"]["85_15"]:,.0f}.
That 1.5-2.5x is the price of being allowed to look whenever you like.

The 85/15 split costs too: for the same precision it needs {sv["variance_ratio_85_15_vs_50_50"]:.2f}x
the users of a 50/50 split. Simulating the power formula at its own MDE gave
{pc(sv["mde_check_1m_users"]["simulated_power"])} power against the 80% target.
""")
fig("sequential_peeking_fpr")
fig("sequential_confidence_sequence")
fig("sequential_time_to_decision", 700)
fig("sequential_power_mde")

# %%
md(f"""
## 6. Who responds?

I split all users once, stratified on treatment and visit: 60% train, 10% validation and 30% test
({U["split_sizes"]["test"]:,} users). The meta-learners use LightGBM and the known propensity, trained on
{U["train_rows"]:,} users; the causal forest got {U["forest_rows"]:,} because it is far slower. The
winner is picked on validation Qini, and everything below is on the test set with 200-replicate Poisson
bootstrap CIs.
""")
for o in ("visit", "conversion"):
    r = U[o]
    rows = []
    for k, e in r["test"].items():
        if k == "random":
            continue
        mark = "**" if k == r["best_on_val"] else ""
        rows.append(
            f"| {mark}{lab(k)}{mark} | {e['qini']:.4f} {ci(*e['qini_ci'], d=4, scale=1)} | "
            f"{e['uplift_at_k']['0.10'] * 100:.2f} | {e['calibration']['slope']:.2f} |"
        )
    md(
        f"""
**{o.capitalize()}** (random ranking: Qini {r["test"]["random"]["qini"]:.4f})

| Model | Qini [95% CI] | Uplift in top 10% (pp) | Calibration slope |
|---|---|---|---|
"""
        + "\n".join(rows)
    )
fig("uplift_qini_visit", 720)
fig("uplift_deciles_visit", 720)
fig("uplift_calibration_visit", 600)
md("""
Conversions are harder. The base rate is 0.29% and the effect about 0.1 pp, so each user carries
roughly a tenth of the signal of a visit. I didn't re-weight classes, because the CATE is a difference
of two probabilities and re-weighting distorts both. Besides training conversion models directly, I
scored the visit-trained models against conversions, since a dense related outcome can rank better than
the rare one. Validation picked a visit-trained model, but none of them separates from the response
model.
""")
fig("uplift_qini_conversion", 720)
fig("uplift_deciles_conversion", 720)

# %%
d = pv["response_model"]
rows = []
for o in ("visit", "conversion"):
    r = P[o]
    for row in r["budget_table"]:
        cell = {
            k: f"{row[k]['incremental']:,.0f} [{row[k]['ci'][0]:,.0f}, {row[k]['ci'][1]:,.0f}]"
            for k in (r["best_model"], "response_model", "random")
        }
        rows.append(
            f"| {o} | {row['budget']:.0%} | {cell[r['best_model']]} | {cell['response_model']} | "
            f"{cell['random']} | {row[r['best_model']]['share_of_total_incremental']:.0%} |"
        )
neg = pv["do_not_disturb"]["predicted_negative"]
md(
    f"""
## 7. Targeting under a budget

Incremental outcomes per {P["population"]:,} users, estimated on the test set as the mean doubly robust
score of the targeted users (nuisance models fit on the training split), so the numbers hold up despite
the imbalance in section 3.

| Outcome | Budget | Uplift model | Response model | Random | Share of all extra |
|---|---|---|---|---|---|
"""
    + "\n".join(rows)
    + f"""

A response model loses only when the budget is tight. Its top 10% would visit at
{pc(d["top_response_decile"]["control_rate"], 1)} without any ad, against
{pc(d["top_uplift_decile"]["control_rate"], 1)} for the uplift model's top 10%. From 20% upwards the two
are indistinguishable: uplift in this data sits mostly with high-propensity users, so propensity alone is
a decent proxy (rank correlation {d["spearman_with_uplift_score"]:.2f}, {pc(d["top_decile_overlap"], 0)} of
the top deciles shared). I expected a bigger win for the uplift model than this.

I also looked for users the ads hurt, with thresholds fixed on validation. The
{pc(neg["share_of_users"], 1)} predicted to be harmed show a positive effect on test
({pp(neg["dr_uplift"], 2)} {ci(*neg["dr_ci"], d=2)}), so the negative predictions are noise. The useful
segment is the opposite one: the half of users with the lowest predicted uplift, who produce about
{pc(1 - b50[best_v]["share_of_total_incremental"], 0)} of the extra visits.
"""
)
fig("policy_budget_curve")
fig("policy_response_model_diagnosis")

# %%
hd = L["high_disagreement_users"]
dec = U["visit"]["test"][U["visit"]["best_on_val"]]["deciles"]
inc = [x["observed"] * x["n"] for x in dec]
md(f"""
## 8. Where things disagree or mislead

All five learners agree that uplift tracks baseline propensity, but they order users differently within
it (prediction SD in pp: {", ".join(f"{lab(k)} {v * 100:.2f}" for k, v in L["prediction_sd"].items())}).
The T-learner is the odd one out. Its control model sees only 15% of the data, so its predictions are the
most spread out and its calibration slope the flattest. The X-learner was built for exactly this
imbalance, and the S-learner did better than I expected, probably because the effect scales with the
baseline. DR and the causal forest over-predict a little where the true effect is near zero. The 10% of
users with the most disagreement have a {pc(hd["control_rate"], 1)} baseline (against
{pc(hd["rest_control_rate"], 1)}) and a distinct feature profile.

Things that looked fine and weren't:
1. **The randomization checks.** They pass, yet the raw ATE ends up
   {pc(cv["diff_in_means"]["estimate"] / dr_v["estimate"] - 1, 0)} too high for visits and
   {pc(cc["diff_in_means"]["estimate"] / dr_c["estimate"] - 1, 0)} for conversions.
2. **The average lift.** The top decile of predicted uplift accounts for {pc(inc[0] / sum(inc), 0)} of
   incremental visits and the bottom half for {pc(sum(inc[5:]) / sum(inc), 0)}.
3. **Exposed users.** They visit far more, mostly because of who they are.
4. **Relative lift on conversions.** +{pc(ac["relative_lift"]["estimate"], 0)} sounds big, but it is
   {pp(ac["diff_in_means"]["estimate"])}, about {ac["incremental_per_million"]:,.0f} extra conversions
   per million users.
5. **Storage order.** The CSV is sorted by treatment, and tied model scores inherited that order in
   decile tables until ties were broken at random.
""")
fig("learners_agreement", 620)
fig("learners_by_baseline")

# %%
rows = "\n".join(
    f"| {lab(k)} | {d['corr_with_true_cate']:.3f} | {d['qini']:.3f} |" for k, d in V["cate"].items()
)
rep = V["replications"]
md(
    f"""
## 9. Checking the methods on synthetic data

The same code runs on synthetic data shaped like Criteo's (12 features, 85/15 assignment, exposure only
for the treated, heterogeneous effects including a negative pocket), where the answers are known. Every
ATE estimator's CI covered the true value ({pp(V["true_ate"])}). Over {rep["n"]} replications the CIs
covered {pc(rep["coverage_diff_in_means"])} (difference in means) and {pc(rep["coverage_cuped"])}
(CUPED) of the time, slightly under 95%. The CUPED bias was {rep["bias_cuped"] * 100:.3f} pp (MC SE
{rep["mc_se_bias_cuped"] * 100:.3f}). On synthetic A/A data the mSPRT's false-positive rate was
{pc(V["sequential"]["fpr_msprt"])}, against {pc(V["sequential"]["fpr_naive_peeking"])} for peeking.

| Learner | Correlation with true CATE | Qini |
|---|---|---|
"""
    + rows
)

# %% [markdown]
# ## 10. Limitations
#
# - The public file was re-sampled on the outcome on purpose, so absolute effects describe this file,
#   not Criteo's campaigns. AIPW fixes the covariate-dependent assignment that causes, but no adjustment
#   undoes selection on the outcome.
# - Without the sub-test identifier I can't analyse each test separately, which is what I'd really want.
# - The causal forest trained on 8x less data than the other learners.
# - Conversion uplift is close to the noise floor.
# - The mSPRT relies on a normal approximation that is rough at the first few conversion looks.
#
# ## References
#
# Diemert et al. (2021), *A Large Scale Benchmark for Individual Treatment Effect Prediction and Uplift
# Modeling*, arXiv:2111.10106 · Deng et al. (2013), *CUPED*, WSDM · Johari et al. (2022), *Always Valid
# Inference*, Operations Research · Howard et al. (2021), *Time-uniform confidence sequences*, AoS ·
# Künzel et al. (2019), *Metalearners*, PNAS · Kennedy (2023), *Doubly robust CATE estimation*, EJS ·
# Wager & Athey (2018), *Causal forests*, JASA · Radcliffe (2007), *Using control groups to target on
# predicted lift* · Imbens & Angrist (1994), *LATE*, Econometrica.
