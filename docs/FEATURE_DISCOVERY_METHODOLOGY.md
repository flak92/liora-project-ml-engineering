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

## 2. The ladder — a Rung 0–9 DAG

Ten questions, in order, as a directed acyclic graph: each node has inputs, a statistic, a threshold
fixed before the data is seen, a negative control, a stop condition, a cost, an artifact, and one
admissible next step. A node that fails ends the walk — later nodes cannot repair an earlier one,
they can only inherit its error. Rungs 0–5 are built and run; Rung 6 is built; Rungs 7 and 9 are
specified and frozen; Rung 8 aggregates what the earlier rungs wrote.

| Rung | Question | Status |
|---|---|---|
| 0 | Is the problem frozen — data, labels, splits, the OOS boundary, the hashes? | built |
| 1 | Can the model learn at all? | built |
| 2 | Does the operating point transfer? | built |
| 3 | Does a feature improve a model that can learn? | built |
| 4 | Does the choice survive data that did not choose it? | built |
| 5 | Is the edge bigger than the maximum a search produces by itself? | built |
| 6 | How much more is a survivor worth under its own tuned model? | built |
| 7 | Do survivors combine — interactions and sequences? | specified |
| 8 | Which OHLCV families travel across assets? | built (aggregation) |
| 9 | Does the whole method hold on a fresh panel? | specified |

**Four classes of variable.** Every configurable in the contract is tagged with the class that
governs when it may move. *Structural* (interval, label horizon, triple-barrier, costs, embargo, the
OOS ban) — changing one changes the problem, so it is frozen at Rung 0. *Calibratable* (`gamma`,
`min_child_weight`, the q-grid, block length, permutation count, HPO budget) — chosen from
data-fitted ranges, never imposed as one absolute number across assets. *Searched* (features,
families, interactions, subsets) — the objects the ladder is looking for. *Methodology-governing*
(the viability floor, confirmation rotations, the test level, the early-stop rule, the empty-subset
acceptance) — the numbers that decide how much proof is enough. The classes are what let the
contract say not just *what* a value is but *when it is allowed to change and what goes stale when
it does*.

### Rung 0 — Is the problem frozen?

Before any measurement, the identity of the problem is fixed and hashed: the bar store and its
sha256, the sample and its sha256, the label horizon and embargo, the Train/OOS boundary with
`oos_reads = 0`, the seeds, the thread-pool caps. Every run writes a `run_manifest.json` recording
the executing commit, the contract's hash as it stood, whether the tree was dirty, the environment
and the measured wall and core seconds. This is why the contract can carry a null commit SHA: a file
cannot hold the hash of the commit that contains it, so the commit that actually ran is recorded at
runtime instead. Without this rung two runs are not comparable; with it, a number can always be
traced to the exact code and data that produced it.

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
| hierarchical | **12/80** | +0.0239 → **−0.0003** |

**The selection rule, and why it had to change.** A fold's verdict is
`T = max(median_j − 0.004)` over units that are *eligible* — picked in at least two rotations, with
a majority of positive confirmation deltas — and `T = 0` when nothing is eligible. `T > 0` is
exactly the old acceptance, since the median threshold and the complexity charge are the same
0.004, so no threshold moved.

What moved is the tie-break. The rule used to examine only the modal pick. On NOW's outer fold 0
two families were each chosen twice and the tie fell to insertion order, handing the verdict to
`price_distance` (positive in 1 of 2, median −0.0070) while `volume` sat there having won both of
its rotations with a median of +0.0298. The fold was rejected on an ordering accident. Counter's
ordering is not reproducible across Python versions, and — the reason this is mandatory rather than
merely tidier — **a permutation cannot honestly reproduce "whichever unit Counter returned
first"**. Replaying the corrected rule over the stored rotations changed exactly one decision.

**Cross-fitting cuts acceptances fivefold.** One table shows the mechanism bare: a feature chosen in
*all four* rotations, with a positive discovery gain every time, confirmed at
`0.0000 / −0.0299 / −0.0221 / −0.0778`. Stability of choice turned out to be stability of noise —
a distinction the whole earlier procedure could not see.

### Rung 5 — Is the edge bigger than the maximum a search produces by itself?

**Specified, not yet executed.** The contract defines it; these results do not include it.

> **H₀:** optional features carry no incremental information over the frozen core, and the observed
> maximum gain is what searching a pool of candidates produces by itself.

The null shifts the **whole optional-feature block** within the discovery fold, in blocks long
enough to respect label dependence — the rule is `L_block = max(H, L_dependency)`, where
`L_dependency` is the first lag after which `|ACF|` stays inside `±1.96/√n` for `K = 5` consecutive
lags. Dependence is measured in **bar time**, not over consecutive event rows: events cluster and
their spacing varies, so a row-wise autocorrelation would measure event density as much as
dependence. Blocks are then mapped onto events through `t0`. A fold must hold at least four blocks,
so `L` is clamped at a quarter of the fold and the rotation flagged if the rule asks for more.

*The rule was frozen before any null was run, and the value measured under it.* On the panel,
`L_dependency` came out 24 bars on NVDA, 14 on MTD, 7 on GWW and 9 on AZO — **the floor binds
everywhere, so `L_block = 24`**. That is the expected outcome rather than a lucky one: the label is
built over 24 bars, so two events closer than the horizon share outcome bars whatever the ACF says.
Verdicts are re-checked at `2L`; a candidate whose verdict depends on the block size is reported
`block_sensitive` and does not pass.

Core, labels and economic outcome stay intact: permuting
`Y_outcome` alone would leave the economic scoring reading unpermuted trade outcomes, so the null
would not match the statistic being tested. Block shifting preserves each feature's autocorrelation
and the correlations among them, and destroys only their temporal alignment to the outcome.

Each permutation reproduces **the entire act of choosing** — for flat, the maximum over all 45; for
hierarchical, the full `family → representative` path; and in both, the operating-point selection,
because the chosen `q` is candidate-dependent in 60% of configurations and a null that froze it
would test a different procedure than the one that ran.

#### The unit is the outer fold, and the permutation belongs to outer-train

A per-rotation null answers a smaller question than the one that matters: it controls "45
candidates, one maximum, one rotation". The procedure being defended is larger — four rotations,
recurrence of the pick, a majority of positive confirmations, a median clearing the complexity
charge — and a null that never reproduces those clauses cannot control the multiplicity they
introduce. So one permutation produces one null realisation of the **whole outer-train optional
matrix**, and the four rotations are merely different *views* of that same realisation. The
statistic is `T`, computed by the same function that produced the real verdict.

This also dissolves a problem with no other solution. Purged walk-forward inner folds **overlap**: a
row belongs to several at once, so "permute within a fold" is not well defined and the natural-
sounding requirement *no block crosses a fold boundary* is unsatisfiable. We do not protect the
boundary of each overlapping fold. We protect the boundary that exists — outer-train against
outer-validation — and purge and embargo are applied afterwards, when the folds are cut out of the
already-permuted object. The confirmation fold is permuted along with the rest: H₀ says the
features are uninformative *everywhere*, and leaving confirmation intact would test the value of
selection rather than the absence of information.

Measured consequence: the procedure-level null sits far higher than the rotation-level one. On
ADBE's outer fold 2 a permutation produced `T = 0.1048` against a real `T = 0.0822` within the
first few draws. **The rotation-level test was too easy**, and the aggregation is what reveals it.
Conversely, a permutation whose best single rotation reached +0.056 scored `T = 0` because no unit
recurred with a majority — the same aggregation cutting the other way.

**What this test does and does not cover.** *Max-null controls candidate-search, operating-point and
rotation-aggregation multiplicity within one outer fold; cross-fold and cross-asset evidence is
controlled by the stability and confirmation contract.* It says nothing about the 16 outer folds and
12 tickers the test is run across. One max-null does not control the multiplicity of the project and
must not be described as if it did.

**A consequence of choosing which folds to test.** The 16 folds are tested *because* the real
procedure accepted something there. The per-fold p-value survives that unharmed — the permutation
distribution is built from that fold's own data, so the conditional argument does not depend on how
the fold was chosen. What does **not** survive is reading the count of passes as a calibrated
family-wise statement. The folds were not sampled; they were selected for having accepted. Report
per-fold verdicts; never "x of 16 passed, expected y under the null".

#### Three nulls, differing only in how the features are made uninformative

Everything downstream is byte-identical between them, which is what makes them comparable.

| | mechanism | what it controls | status of the evidence |
|---|---|---|---|
| **A1** | global block permutation across outer-train | `optional ↔ outcome` alignment | **marginal**, not the conditional H₀ |
| **A2** | blocks permuted only within `G = 4` chronological macro-segments of equal *bar* span | the same, with the local regime held | sensitivity check on A1 survivors |
| **B** | cross-fitted `g`: permute the residual `optional − g(core)`, rebuild as `g(core) + residual_perm` | keeps the dependence on core | **sensitivity check, not an exact CRT** |

A1 is honest about its own limitation: it destroys `optional ↔ core` as well as
`optional ↔ outcome`, so it is a *marginal alignment* null, not the
`X_optional ⊥ Y | X_core` that the hypothesis names. Breaking the coupling to core can actively
hurt a model — changed competition for splits, `colsample_bytree`, gain dilution, changed available
interactions — which would shift the whole null distribution left and make it too easy to beat.
That is what B exists to detect, and why "A passed, B rejected" has its own row in the
interpretation table rather than being treated as noise.

B's honesty runs the other way: `P(X_optional | X_core)` is *estimated*, not known, so its
correctness depends on the quality of `g`. Its alpha is chosen by held-out reconstruction MSE of the
features alone — no label, no outcome, no trading result enters the nuisance model — and rebuilding
as `g(core) + residual_perm` does not reproduce each column's original marginal, a drift that is
measured and recorded rather than hidden.

**A2's segments are cut by bar span, not row count.** An equal-row split would make a quiet stretch
span years and a busy one span weeks, which is the opposite of holding the local distribution fixed.

`M = 50`, `α = 0.10`, `b = #{null ≥ real}`. Pass requires `b ≤ 4` after the full fifty
(`p = 5/51 = 0.0980 ≤ α`). **Futility bound:** stop at `b = 5` — even if every remaining permutation
fell short, `p ≥ 6/51 = 0.1176` and the candidate cannot pass. Early-stopped tests report
`rejected_early`, the permutations performed, the exceedances and a *lower bound* on the p-value;
`(1+b)/(1+n)` is not a fixed-budget p-value for a test that stopped early.

The futility bound is **deterministic, not statistical**: `b` never decreases, so once it reaches
five it is still ≥ 5 at fifty, and the early verdict and the full-budget verdict agree by
construction. Checking that they agree therefore proves nothing. What is worth checking on real
data — and is checked — is that the number reported at the stop genuinely *bounds* the one a full
run produces.

#### Two controls, because failing in either direction is invisible from the results

A methodology that always rejects is exactly as useless as one that never rejects, and neither
failure shows up in a run that produced few survivors.

The **negative control** costs nothing, because every permutation already *is* an independent
synthetic-null experiment: under it the optional features carry no information by construction, so
`T_null > 0` means the acceptance contract accepted something when there was nothing there. The rate
of that across all permutations is the type-I error of the four-rotation rule *on its own* — the
multiplicity the max-null exists to remove — reported with a Wilson interval, because the normal
approximation is wrong exactly where these rates live.

The **positive control** plants a column of known strength,
`z = a·standardised(y) + √(1−a²)·noise`, replacing an existing candidate's column so the name and
family machinery cannot tell it apart. It reports the minimum detectable effect at power 0.8, the
power at each strength, the cost, and how power decays as the candidate pool grows from 5 to 45.
The plant is deliberately unrealistic. It is a ruler, not a feature.

#### Known property, recorded rather than repaired

The quantile arm calls `select_operating_point` with `min_oof_trades` only, leaving
`min_active_folds = 0` and `max_fold_trade_share = 1.0` at their defaults, so an operating point
whose trades all come from a single fold is admissible. This cannot bias the permutation test — the
real run and every permutation take the identical code path — so it is a property of the *procedure
being defended*, not of the test defending it. It is recorded here because it matters when the
methodology is carried to a new panel, and left alone because changing it now would alter the
procedure mid-test.

### Rung 6 — How much more is a survivor worth under its own tuned model?

Everything up to here freezes the hyper-parameters on core, deliberately: tuning on the superset
co-adapts the parameters to features the baseline lacks and inflates every gain. That is the right
discipline for *screening* — cheaply eliminating the many — but it undersells the few that survive,
because it never asks what a confirmed feature is worth once the model is allowed to tune *with* it.

Rung 6 asks exactly that, and only of survivors. For each unit that cleared Rung 4 and Rung 5, it
tunes core alone at a budget `B` and core+survivor at the *same* budget `B`, both on discovery folds
only, then compares the two on a confirmation fold neither tuning ever saw. Equal budget is the
fairness constraint: a survivor must earn its keep against a core that had the same chance to
improve, not against a core frozen at last rung's parameters.

The danger is that tuning is itself a search, so Rung 6 could manufacture an improvement the way any
search manufactures a maximum. Two guards prevent it. The comparison fold is untouched by the
tuning, so the reported gain is out-of-sample for the hyper-parameter search. And Rung 6 **can only
demote, never promote**: it may find that a Rung-5 survivor adds nothing once core is retuned (and
strike it), but it cannot resurrect anything Rung 5 rejected — if it could, the screening was
pointless. This splits the method cleanly into a FAST SCREENING phase (frozen core, all 45
candidates, cross-fitting, max-null) and a MAXIMUM EXTRACTION phase (survivors only,
candidate-specific tuning), spending the full per-feature cost only where the evidence already
points.

### Rung 7 — Do survivors combine? *(specified, not executed)*

A feature can be weak alone and valuable in combination — with a volatility regime, a trend
direction, another feature. Searching all pairs of 45 candidates is `O(N²)` = 990 evaluations and
re-imports every winner's-curse problem the ladder just controlled. Restricting the search to
survivors makes it `O(S²)`: with the handful that survive Rung 5, the pair space is single digits.
The same subset evaluator and family machinery the earlier rungs use carry over unchanged; the
interaction gate is the confirmation/null discipline of Rungs 4–5 applied to pairs. It is specified
and frozen, to be built once there are at least two survivors to combine.

### Rung 8 — Which OHLCV families travel across assets?

The per-ticker ladder says what works for one asset. Transfer asks whether a surviving family is a
universal edge, a regime-conditional one, or one asset's artefact — answerable only across the
panel. Rung 8 aggregates the per-ticker verdicts into a ticker×family matrix, on the family as the
stability unit, and classifies each: **universal** (confirmed on a majority of the tickers where it
was even a candidate), **conditional** (more than one, not a majority), **asset-specific** (exactly
one). The output doubles as a transfer prior — the order in which to search families on a new asset —
which narrows effort without ever skipping the ladder: a new asset still earns its own confirmation.

---

## 2b. Objective and stopping

**The objective function `J` is a report, not a gate.** The vision writes it as
`J = U_outer − λ₁·optimism − λ₂·instability − λ₃·complexity − λ₄·cost`, and that is exactly how it is
used — to *rank* the survivors and book the compute, never to decide admissibility. Admissibility is
the gates and the max-null, which are hard constraints; a feature with a poor `J` that nonetheless
passed every gate is still selected, and a feature with a fine `J` that failed a gate is still
rejected. Writing `J` as a soft selection criterion — letting high outer utility buy back a large
optimism gap — would reopen precisely the winner's-curse door Rungs 4 and 5 exist to shut. So `J`
orders what the gates have already admitted, and nothing more.

**The stopping rule is: what is the smallest experiment that could still change the verdict?** At
every point the method prefers the least compute that reaches the same decision. The futility bound
is one instance — a null stops at `b = 5` because no remaining permutation can change it. The general
form is an automaton that reads the artifacts each rung wrote and names the next admissible
experiment: viability → utility → cross-fit → null → survivor HPO → transfer, halting with an empty
subset the moment the evidence cannot support one. The automaton advises; it never acts, and it
never auto-recalibrates. A state like "the model cannot learn — widen the space" does not loop back
into an automatic retune, because widening the space is co-adaptation to the panel and a decision
that mints a new contract version. Those states stop and ask for human authorization.

## 2c. The Compiler output

The end of the method, for one asset, is a single resolved record — assembled from the artifacts,
never recomputed:

```json
{ "ticker": "AZO", "model": "xgb", "status": "resolved",
  "selection_mode": "hierarchical", "selected_features": ["momentum_return"],
  "rejected_features": 44, "confirmation_win_rate": 1.0,
  "max_null_p": [{"unit": "momentum_return", "p_mc": 0.058824}],
  "outer_delta": 0.058349, "compute_seconds": 540.8,
  "stop_reason": "survived the procedure-level max-null" }
```

or, just as valid, the empty resolution:

```json
{ "ticker": "ADBE", "status": "resolved_empty",
  "stop_reason": "no unit exceeded the procedure-level max-null" }
```

Both are answers. The method does not pick the best available feature; it reports only what survived
being chosen and judged by different data — and reports plainly when that set is empty.

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
| 7 | procedure-level null A1 | `scripts/procedure_null.py --null a1 --jobs 4` | `b = 5` ends that fold and arm |
| 8 | grouped-null sensitivity | `scripts/procedure_null.py --null a2 --survivors-from …a1.json` | no survivors is a complete result, not a failure |
| 9 | conditional-null sensitivity | `scripts/procedure_null.py --null b --survivors-from …a1.json` | as above |
| 10 | controls | `scripts/null_controls.py --negative --positive` | a ladder that never finds the plant is not a ladder |
| 11 | summary | `scripts/rung5_summary.py` | a human reads this and writes the verdict |

Steps 7–11 run unattended under `make loop-start`, which puts the supervisor inside a tmux session
so the tmux *server* is the daemon and the terminal can be closed. The chain is resumable at the
permutation: every finished unit is a ledger line, so a machine that dies at hour four resumes at
hour four rather than starting again. `make loop-status` shows the session, the lock, the control
channel, per-stage progress and the ledger's hash-chain integrity; `make loop-stop` sets a
cooperative halt that lands between units, never mid-unit.

The chain **never makes a methodological decision**. It computes, checks invariants, and records.
Its gates test only things that must hold regardless of which way the verdict went — permutation
uniqueness, block displacement, verdict/exceedance consistency, the p-value bound, manifest
completeness. A gate that fired on "too few survivors" would be the chain forming an opinion.

Verify at every step: `make verify` green, sealed pipeline untouched, zero OOS reads. And
`make loop-selftest`, which proves the orchestration itself — that a child forked without `9>&-`
really does keep the lock alive (the negative control, without which the test proves nothing), that
an artifact survives SIGKILL mid-write, that the watchdog actually restarts a dead chain, and that
the gates fail closed on a missing or truncated input.

**An empty subset is a complete and correct answer.** The procedure does not pick the best available
feature; it picks only features that survived being chosen and judged by different data.

---

## 8. Rung 9 — Certification on a fresh panel

The terminal rung, and the only one that measures the method rather than builds it. The current
panel cannot certify anything, because its outer results informed the method's design.
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
