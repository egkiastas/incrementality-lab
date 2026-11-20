# Design notes

These are the decisions someone reviewing this would probably ask about.

## Where the data comes from

The download link on Criteo's page and the S3 bucket behind `sklift.datasets.fetch_criteo` both
returned 404 when I started. `analysis/00_download_data.py` still tries the official link first,
then falls back to the GitHub release mirror that the libuplift project points to. The file is only
accepted if its SHA-256 matches libuplift's published hash and its MD5 matches the one hard-coded in
scikit-uplift 0.5.1. Two unrelated projects agreeing on the bytes is good enough for me. The row
count is checked again after converting to Parquet. If both links die, drop the `.csv.gz` into
`data/raw/` and the checksums still run.

## The covariate imbalance

The usual checks pass (SRM p = 0.999, every SMD < 0.1). But with 14M rows the standard error of an
SMD is about 0.00075, so the largest one (0.049) is ~65 SEs out, all 12 features fail a Bonferroni
test, and a LightGBM classifier predicts treatment with AUC 0.507. The treated share is also too
exact: 85.00001%, which a coin flip lands that close to about once in a thousand runs.

What made it matter was sorting users by predicted visit probability. The treated share climbs from
about 84.9% to 86.5% exactly where visit rates are highest, which inflates the raw difference in means.
Criteo's paper says each sub-test was re-sampled to 85% treated and negatives were down-sampled
non-uniformly to hide the true effect size, and that combination would produce this.

I left the data alone. The headline ATE is cross-fitted AIPW with an *estimated* propensity, since it
is consistent if either the outcome models or the propensity model is right. The plain difference
and the other adjusted estimators are reported next to it. They don't fully agree (0.63–0.75 pp), so
I give the range. On fake 85/15 splits of control users the adjustment centres on zero, so the shift
comes from the split, not from the method.

## Using the known propensity

This is an RCT, so the X-learner, DR-learner and causal forest all use the design value (0.85)
instead of an estimated propensity. The causal forest gets `DummyClassifier(strategy="prior")` as its
treatment model. The cost is that control pseudo-outcomes in the DR-learner get multiplied by
1/0.15 ≈ 6.7, which is why its final stage is heavily regularised. The one place I do estimate e(x)
is the AIPW check above, because there the question is whether assignment really was independent of X.

## Sequential testing

The test is a two-sample mixture SPRT (normal mixture, plug-in variances) following Johari et al.,
*Always Valid Inference*. It rejects when the likelihood ratio passes 1/α, and inverting the same
boundary gives the confidence sequence. The mixing scale τ is 10% of the control rate, fixed up
front. The first look is at 10,000 users. I didn't use group-sequential designs because they need the
looks planned in advance, and Bonferroni across looks is far too conservative. With discrete looks the
mSPRT ends up conservative. A larger τ would have stopped sooner on conversions, where the effect is
~50% relative.

## Ties and the file's sort order

In the first full run one DR-learner decile had no control users at all. LightGBM regressors are
piecewise constant, so about 10% of test users shared one predicted value. The Criteo CSV is also
stored roughly sorted by treatment. A stable sort broke ties by row order, and the deciles inherited
the file's treatment pattern. Qini and AUUC were fine, because they treat tied scores as one step.
Decile tables, uplift@k and top-k targeting were not. Every ranking now breaks ties at random with a
fixed seed, and a regression test stores data sorted by arm with all scores tied and checks that the
deciles stay balanced.

## Smaller choices

- For a 0/1 outcome, resampling rows from an arm with k ones gives Binomial(n, k/n) ones, so the
  2,000-replicate bootstrap is two binomial draws per replicate instead of 14M row draws.
- CUPAC's model is cross-fitted and never sees T, so a unit's own outcome can't leak into its
  "pre-treatment" covariate.
- Models train on a 4M-user stratified subsample (500k for the causal forest) to fit in 16 GB. Every
  estimate and evaluation uses all rows of its split.
- The best learner is picked on validation Qini and reported on the test set. Budget numbers use
  doubly robust scores with nuisance models fit on the training split, so they hold up despite the
  imbalance.
