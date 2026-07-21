# Golden Calibration — finding features that survive being chosen

**Status: Development Candidate, not v1.** The twenty tickers this was built on are a *development
panel*: their outer-fold results were used to decide the next methodology change, so the numbers
below measure how the procedure was built, not how well it performs. The name `Golden Calibration
v1` is earned only after the certification run in §8 — a fresh panel, the contract frozen, no
threshold touched.

The contract is [`config/feature_discovery_contract.json`](../config/feature_discovery_contract.json).
This document describes it. Where the two disagree, the JSON is the contract.

---

## 1. The problem

Given a per-asset model and a pool of candidate features, decide which features — if any — earn a
place. The naive procedure is: score every candidate on cross-validation folds, take the best, keep
it if its gain clears a threshold. Measured on twenty S&P 500 tables, that procedure reports a mean
inner gain of **+0.1378** and delivers **−0.0337** on data it did not choose on. Four failures
compound to produce that gap, and each needs its own control.

**The model may be unable to use features at all.** In 63% of draws from the sealed hyper-parameter
space the booster had *no split nodes* — a constant predictor. Comparing feature subsets under a
constant compares nothing, and a third of a published 498-asset study was trained that way.

**The operating point may not transfer.** Predictions concentrate near the base rate and their
spread narrows as the training set grows (one table: max 0.443 → 0.422 → 0.412 → 0.407). An
absolute probability threshold fixed on smaller folds can sit above everything a larger model
produces: 32 of 80 folds traded nothing.

**Importance inside a model is not utility.** The feature SHAP ranks first improves a capable model
in 22% of folds; picking at random gives ~24%.

**The maximum of many noisy estimates is biased upward.** This is the dominant term. Selecting the
best of 45 candidates and then scoring that best on the same folds inflates the estimate by
construction, and no amount of care downstream removes it.

---

## 2. The ladder

Five questions, in order. Each has a statistic, a threshold fixed before the data is seen, a
negative control, a stop condition and a cost. A stage that fails ends the ladder — later stages
cannot repair an earlier one, they can only inherit its error.

### Rung 1 — Can the model learn at all?

`gamma` and `min_child_weight` are thresholds in units of summed **hessian**: a row contributes
`w_i · p(1−p)`. Sample weights shrink the effective size — 9629 rows carrying 524.7 of weight give a
fold hessian near 50–100 where unweighted data would give ~1400. Absolute ranges calibrated for the
larger scale forbid nearly every split at the smaller one. Measured at hessian 75: `gamma = 1.0`
leaves 29 split nodes, `gamma = 5.0` leaves **one**. `min_child_weight` across its whole range only
moves splits 1038 → 389 and never kills the model, so `gamma` is the culprit.

**Fix:** express both relative to the fold's own hessian. The same `g_rel` then means the same thing
on every asset — measured across four assets and two fold sizes with hessians from 47.8 to 101.3,
`g_rel = 0.002` gives 480–919 split nodes everywhere. This is what makes the space *calibrate itself
from each asset's data* instead of being imposed on all of them.

| | dead trials | median split nodes | median pred_std |
|---|---|---|---|
| absolute space | **380/600 (63%)** | 0 | 0.0000 |
| hessian-relative | **48/600 (8%)** | 72 | 0.0161 |

**Viability floor: `split_nodes ≥ 20`, `pred_std ≥ 0.005`.** Read off the register, not chosen by
taste: median log-growth is −1.4629 at zero splits (0% of trials positive), −0.6222 at 1–20 (22%),
**+0.1789 at 21–200 (59%)**. The sign of the objective flips exactly where predictions stop varying.

*Negative control:* the floor rejects 88% of draws from the old space and 26% from the new one. A
gate that never rejects validates nothing.

*The floor decides admissibility, never rank.* It says whether a configuration **tested** its
features. Ranking stays marginal log-growth and fold-win fraction. A gate that picked winners would
be tuning dressed as hygiene.

### Rung 2 — Does the operating point transfer?

Make the threshold relative to the prediction distribution: *fire the top `q` of signals*, where the
cut is that quantile of the model's predictions **on its own training rows**. Causal — the cut
exists before the window is traded — and it needs nothing the procedure may not see.

The source matters and was measured, not assumed. On one table's first outer fold the inner-OOF
distribution puts q85 at 0.457 while the outer fold wants 0.512, because models trained on less data
predict wider; the model's own training distribution puts it at **0.516**.

| | zero-trade folds | fired-share range |
|---|---|---|
| absolute threshold | 32/80 | 1.000 |
| quantile | **0/80** | **0.563** |

**This fixed the mechanism and changed nothing else** — outer win rate 16/33 → 15/33, median
−0.0171 → +0.0000. Worth stating plainly: a repair that works can still leave the result untouched,
and that is information, not disappointment.

### Rung 3 — Does the feature improve a model that can learn?

Per asset and per outer-train window: one shared event set, identical inner folds, **HPO on core
only**, then freeze. Every configuration — core, core + one feature, core + one whole family, greedy
subset — sees the same events, folds, seed, operating-point contract and execution costs. Only the
features vary, which is the only way a difference is attributable to them.

*Why HPO on core:* tuning on the full candidate superset co-adapts parameters to features the
baseline lacks, and inflates every gain measured against it.

Results over 3600 single-feature configurations, all viable:

- marginal gain median **−0.0649**, positive in **24%**, median fold-win **0.250**
- **every one of the 12 families has a negative median gain as a block** (−0.054 to −0.133), yet the
  best *member* of several is positive (momentum_return +0.0511, price_distance +0.0501, macd
  +0.0245) — a family is a taxonomy, not an input block
- greedy subsets are small: median **one** feature
- SHAP rank versus marginal utility: **ρ = −0.074**; the SHAP top-1 helps in 22% of folds against
  ~24% at random

**Adding a feature to a capable core usually hurts.** SHAP leaves the selection path here and stays
a diagnostic column.

### Rung 4 — Does the choice survive data that did not choose it?

Split the inner folds by role. Discovery folds rank the candidates; a rotating confirmation fold
measures the pick, having played no part in choosing it. Each inner fold serves once as untouched
validation. The operating point comes from discovery and is applied to confirmation as a fixed
level — choosing it on the confirmation fold reintroduces the same leak one rung down.

Two arms, both declared before the run so that neither is a repair fitted to earlier results:

- **flat** — 45 candidates → single maximum → confirmation
- **hierarchical** — 12 families → strongest family → its representative → confirmation, with the
  **family** as the stability unit, so a recurring OHLCV relationship counts even when a different
  variant of it wins each time

| | provisional acceptances | discovery → confirmation |
|---|---|---|
| naive procedure | 66/80 | — |
| flat | **14/80** | +0.0419 → **+0.0000** |
| hierarchical | **11/80** | +0.0239 → **−0.0003** |

**Cross-fitting cuts acceptances fivefold.** One table shows the mechanism bare: a feature chosen in
*all four* rotations, with a positive discovery gain every time, confirmed at
`0.0000 / −0.0299 / −0.0221 / −0.0778`. Stability of choice turned out to be stability of noise —
a distinction the whole earlier procedure could not see.

### Rung 5 — Is the edge bigger than the maximum a search produces by itself?

**Specified, not yet executed.** The contract defines it; these results do not include it.

> **H₀:** optional features carry no incremental information over the frozen core, and the observed
> maximum gain is what searching a pool of candidates produces by itself.

The null shifts the **whole optional-feature block** within the discovery fold, in blocks long
enough to respect label dependence. Core, labels and economic outcome stay intact: permuting
`Y_outcome` alone would leave the economic scoring reading unpermuted trade outcomes, so the null
would not match the statistic being tested. Block shifting preserves each feature's autocorrelation
and the correlations among them, and destroys only their temporal alignment to the outcome.

Each permutation reproduces **the entire act of choosing** — for flat, the maximum over all 45; for
hierarchical, the full `family → representative` path; and in both, the operating-point selection,
because the chosen `q` is candidate-dependent in 60% of configurations and a null that froze it
would test a different procedure than the one that ran.

`M = 50`, `α = 0.10`, `b = #{null ≥ real}`. Pass requires `b ≤ 4` after the full fifty.
**Futility bound:** stop at `b = 5` — even if every remaining permutation fell short,
`p ≥ 6/51 = 0.1176` and the candidate cannot pass. Early-stopped tests report `rejected_early`, the
permutations performed, the exceedances and a *lower bound* on the p-value; `(1+b)/(1+n)` is not a
fixed-budget p-value for a test that stopped early.

---

## 3. The smallest sufficient step

The recurring design question is not "what is the best possible control" but "what is the smallest
step that already settles the question". Four instances, each of which saved more than it cost:

| instead of | do | why it is enough |
|---|---|---|
| tuning the model space per asset | scale two parameters by the fold's hessian | one number then means one thing everywhere — measured across hessians 47.8–101.3 |
| holding out a separate validation set | rotate the confirmation fold | every fold serves as untouched validation once, at no extra data cost |
| running all 50 permutations | stop at the futility bound | the verdict is already determined; the remaining permutations cannot change it |
| running the null everywhere | run it only for provisional survivors | the null can only reject an acceptance, never rescue a rejection — 42 rotations instead of 320, 8.2 core-hours instead of 62 |

---

## 4. Negative results

They carry most of the information here, so they are stated rather than buried.

- **SHAP does not predict marginal utility.** ρ = −0.074 over 80 folds; top-1 helps in 22% against
  ~24% at random. SHAP describes a model that already exists; it does not identify what would
  improve one.
- **A whole family hurts even when its best member helps.** All 12 families negative as blocks;
  several positive at their best member. Redundancy inside a family costs more than the family adds.
- **Greedy buys nothing over its own best single feature.** Median outer delta −0.0337 for greedy,
  −0.0164 for the best single, 29/66 wins against 30/66. The damage is done when a winner is picked,
  not when features are combined.
- **Optimism gap +0.1639**, positive in 59 of 66 folds — the textbook fingerprint of selection on
  the evaluation data.
- **Selection is unstable.** Median Jaccard between outer folds **0.000**; 108 of 114 pairs share no
  feature at all.
- **63% of draws from the sealed space produced a constant predictor**, and 26% of the published
  sealed models have zero split nodes.

---

## 5. Cost model

Measured per ticker, before the runtime fix, at three workers with uncapped thread pools:

| stage | core-seconds / ticker |
|---|---|
| viability register | ~90 |
| feature utility register | ~400 |
| outer evaluation | ~19 |
| cross-fitting, both arms | ~590 |

**Runtime configuration is part of the cost model.** XGBoost pinned to one thread while OpenBLAS and
OpenMP opened one per core meant three workers ran twelve threads on four cores. Capping the pools:
12.33 s → 8.15 s (**1.51×**), with output identical to the bit. Worker count, eight tables:

| jobs | wall | core-seconds |
|---|---|---|
| 1 | 63.90 s | 63.88 |
| 2 | 33.71 s | 65.73 |
| 3 | 24.62 s | 65.96 |
| **4** | **19.72 s** | 67.88 |

Wall falls monotonically while core-seconds rise 6%: four workers are Pareto-optimal on four cores
once the pools are capped. Re-measure the per-stage costs under this configuration before budgeting
a new panel.

---

## 6. Transfer to LSTM

**Derived protocol, not validated.** No LSTM result below has been produced; this is what the ladder
becomes when the model changes, and it must be run before it is believed.

What stays: the viability gate, discovery/confirmation rotation, multiplicity control, the max-null,
the empty-subset policy, cost-aware stopping, and the untouched outer validation.

What changes:

- **daily bars, not hourly**, with their own event set and windows — the two models do not share a
  panel, so their registers are separate and their results are not pooled
- **attribution by occlusion, not TreeSHAP** — the `shap` package pulls `numba`, which pins
  `numpy < 2.5` against this repository's `2.5.0`. Occlusion introduces its own methodological
  variable: *what replaces the removed channel*. Substituting the training mean leaves the channel
  present but constant, and correlated channels share the credit; occluding single time steps
  additionally puts a hole in the sequence that the market never produces, so occlude **blocks**
- **viability means something else** — an LSTM has no split nodes. The analogue is prediction spread
  plus a floor on how much the output moves when an input channel is occluded; the threshold has to
  be read off an LSTM register the way rung 1's was read off the XGB one
- **refit cost is far higher**, so the futility bound and the survivors-only null matter more, not
  less

---

## 7. Runbook

For a new panel of assets. Every step writes a `run_manifest.json` recording environment, worker and
thread configuration, wall and core seconds, and artifact hashes.

| # | step | command | stop condition |
|---|---|---|---|
| 1 | profile the tables | `scripts/profile_tables.py` | — |
| 2 | choose the panel by a data-only rule | `scripts/select_sample.py` | rule must not read a result or the OOS window |
| 3 | viability register | `scripts/model_viability.py --jobs 4` | if no space yields learnable models, fix the space before anything else |
| 4 | feature utility register | `scripts/feature_utility.py --jobs 4` | a window with no viable model contributes no feature evidence |
| 5 | outer evaluation | `scripts/nested_outer.py --jobs 4` | — |
| 6 | cross-fitted selection | `scripts/crossfit_selection.py --jobs 4` | **zero provisional acceptances ends the run** — the null cannot rescue anything |
| 7 | max-null, survivors only | *(runner pending)* | `b = 5` ends that candidate |

Verify at every step: `make verify` green, sealed pipeline untouched, zero OOS reads.

**An empty subset is a complete and correct answer.** The procedure does not pick the best available
feature; it picks only features that survived being chosen and judged by different data.

---

## 8. Certification

The current panel cannot certify the method, because its outer results informed the method's design.
Certification requires:

1. freeze this contract and the code;
2. select a fresh panel by a declared rule;
3. change no threshold, no null, no HPO space, no selection logic;
4. run the whole ladder;
5. report accuracy, cost and the number of empty subsets.

Two controls belong to certification, not to development:

- **Negative control** — not "one synthetic noise feature must be rejected in a given run", but a
  series of independent synthetic-null experiments yielding an empirical false-acceptance rate with
  a confidence interval, compared against the declared level.
- **Positive control** — a synthetic feature with a controlled dependence on the target, in a
  separate benchmark, measuring the minimum detectable effect size, the probability of recovery, the
  cost of detection, and how power degrades with the number of candidates.

A methodology that always rejects is as useless as one that never rejects. Only after both controls
does this become **Golden Calibration v1**.
