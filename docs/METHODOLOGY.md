# METHODOLOGY

How this research was run, what it claims, and — just as important — what it does not claim.

## 1. Research question and verdict

**Question.** Can a dedicated ML ENTRY indicator be calibrated per S&P 500 asset — one XGBoost
model on 1-hour bars and one LSTM on daily bars per ticker — under a frozen, mechanical TP/SL
contract, and how does it compare to buy-and-hold (HODL)?

**Scope.** One sealed per-asset artifact per model per asset (this release: 498 XGBoost +
495 LSTM = 993 — the authoritative counts live in `research_run`, which the console reads
rather than hardcodes). Each artifact carries a
single out-of-sample (OOS) evaluation:

| model | bars | Train window | OOS window |
|---|---|---|---|
| XGBoost | 1h | 2016-10-17 to 2023-12-29 | 2024-01-02 to 2026-05-29 |
| LSTM | 1d | 2017-01-01 to 2023-12-31 | 2024-01-01 to 2026-04-30 |

These are the declared boundaries, straight from `config/{xgb,lstm}.json`. The LSTM ones are
calendar-round on purpose, so every sealed LSTM row reports its first *realized* session,
2024-01-02 — 1 January is a market holiday. The XGB config already declares session dates, so
its rows and its declaration read the same. No bar is added or dropped either way.

**Honest verdict.** Over the strong 2024-2026 bull OOS window the strategies do not beat
buy-and-hold on most assets. The demonstrated product is the causal, OOS-isolated *method*: a reproducible,
Train-only calibration procedure that yields, per asset, an ENTRY indicator with an explicit
description of when it acts and when it stays idle. Three things must never be conflated:

```text
Train-derived interpretation  !=  OOS result  !=  live trading signal
```

The interpretation layer describes what the sealed model did on its own Train data; the OOS row
is a single frozen read; neither is a recommendation to trade.

## 2. Label contract: ATR Triple Barrier with mechanical exits

The model decides ENTRY only. TP and SL are never a per-asset model decision — they are fixed by
the label contract and identical in research, verification and the sealed artifact:

- **Direction (meta-labeling primary).** A candidate event exists at every eligible bar whose
  momentum feature is non-zero; its side is the sign of that momentum. The model then classifies
  whether the candidate is worth taking.
- **Barriers.** Asymmetric ATR Triple Barrier: TP = 2.0 x ATR(14), SL = 1.0 x ATR(14), both
  widths read at t0 (strictly causal, data up to close[t0] only). Reward:risk b = 2.
- **Horizon.** XGB: 24 hourly bars; LSTM: 10 daily sessions. If neither barrier is hit, the
  trade exits at the horizon.
- **Label.** Y = 1 iff the realized net return of the barrier trade is positive — costs on
  both sides and the gapped fill included. This is stricter than "TP before SL": a target
  touch whose next-open fill turns negative after costs is labelled 0, and a time-barrier
  exit that ends positive is labelled 1.
- **Execution realism.** Entry fills at the next bar open; commission and slippage are charged
  on both sides; capital policy is all-in compounding per asset (no cross-asset portfolio).
- **Overlap control.** Overlapping label windows are down-weighted by label-uniqueness weights.

Because exits are mechanical, "the model knows when to enter" is the entire claim — never
"the model chooses ENTRY, TP and SL".

## 3. Train-only calibration of the operating point

The per-asset operating point (entry threshold theta_entry, plus direction mode for the LSTM) is
chosen exclusively on Train data, via one shared implementation (`src/shared/op_select.py`) used
by both pipelines, so the theta spectrum, the trade floor and the robustness constraints cannot
drift between scorers.

**Accumulate-then-select.** Every fold's out-of-fold (OOF) predictions from purged walk-forward
CV are replayed through the deployed trade engine at each grid point; per-fold log-growth and
trade counts are accumulated *across* folds, and ONE point is selected for the whole Train
window. Selecting a best theta per fold (a "fold oracle") is forbidden: no deployable strategy
can switch theta between folds, so per-fold selection upward-biases every score it touches —
HPO trials and feature gains included.

Selection order, deterministic and pure:

1. **Trade floor** — only points with at least `min_oof_trades` total OOF trades are viable.
2. **Fold spread** — prefer points with trades in enough folds and no single fold holding a
   dominant share (a one-fold burst is not evidence); relaxing this is flagged, never silent.
3. **Plateau** — within one standard error of the best log-growth, prefer an interior theta
   over a spectrum edge.
4. **Conservative ties** — higher theta first, then smaller sizing fraction.

If nothing clears the floor, the most-trading point is scored and flagged
`trade_floor_met = false`; such a result is never promoted — it surfaces in the results as
`TRAIN_OOF_FLOOR_NOT_MET`, i.e. the system explicitly reports when the model should stay idle.

**Purged walk-forward CV.** Expanding folds with an interval purge (a train event whose label
window overlaps the test start is dropped) and an embargo; feature scaling uses train-fold
statistics only. Leakage controls are structural, not statistical.

## 4. Feature search: golden calibration

"Golden calibration" is this project's name for the search policy: the goal is not a global
numerical optimum but the *best observed stable calibration* for the pinned Train period and
recipe hash. A result-affecting number is always interpreted through its range, spectrum, scope
and dependencies — never as an isolated magic value.

### 4.1 Families, spectrum, plateau, representative

- Every candidate feature belongs to exactly one **OHLCV relationship family** (return/momentum,
  acceleration, distance/location, slope, volatility level, volatility structure, range
  position, compression/efficiency, volume/liquidity, divergence/spread, ratio, alignment,
  session position). The family maps live in `config/feature_families_xgb.json` and
  `config/feature_families_lstm.json` and are validated fail-closed against the candidate pool.
- A family's tested variants (windows, lags, transforms) form its **spectrum**.
- The variants statistically indistinguishable from the family best — within one standard error
  of the best variant's per-fold deltas (sample std, ddof = 1) — form the **one-SE plateau**.
- The **representative** is the *simplest* member of that plateau: already implemented over
  newly proposed, fewer columns, lower warm-up loss, less specialized scope, deterministic
  tie-break. `best point != golden calibration` — the chosen point is an interior, parsimonious
  member of the plateau, not the largest point estimate at the spectrum boundary.

### 4.2 Two-stage family-first greedy (XGB)

1. Features are computed ONCE on the superset manifest, fixing one common event set and one
   purged-fold partition for every candidate subset; a subset evaluation is a pure column slice.
2. Baseline = the frozen 1h core only; ONE Train-only HPO is run and its parameters are frozen
   for all subset comparisons.
3. Marginal prefilter: each candidate must show a robust single-feature Train-OOF log-growth
   gain over core (`min_feature_gain`) in a sufficient fraction of folds (`min_fold_win_frac`),
   always scored at one shared floor-respecting operating point (never a per-fold theta).
4. Stage 1: a single greedy pass over the family *representatives*; each is added only if it
   lifts the complexity-penalized score AND beats the current subset in a majority of folds.
   Stage 2: the remaining variants of *accepted* families only, capped per family. `max_select`
   caps total additions.
5. The subset is applied only if it beats core by `min_gain` AND the baseline produced at least
   the minimum Train-CV trades — no economic power implies no selection; thin tickers stop
   before paying for the search.

The LSTM search follows the same policy with forward selection under a complexity penalty,
seed-averaged evaluations, and the same family-first ordering.

### 4.3 Gates and stop rules

Acceptance thresholds are guards, not targets: they are never lowered to rescue a weak ticker.
A ticker stops immediately on the first applicable rule — baseline below the trade floor
(`thin_no_trades`), no family survivor (`core_only`), family exhausted, plateau found, or the
final subset failing the gain gate (override cleared, core kept). A rejected direction is
recorded and not re-tested under the same recipe. Every result is recorded as *best observed*
for {ticker, model, Train period, folds, seeds, recipe hash} — never as a global optimum.

## 5. LSTM universal-backbone warm start

To make per-asset LSTM training tractable without giving up per-asset feature selection, ONE
universal LSTM is pretrained on the pooled Train-region panel of all tickers over the feature
superset at a fixed architecture. Every per-asset training — feature-search evaluations and the
sealed refit alike — warm-starts from that checkpoint; the input weight matrix is column-aligned
by feature id, so a subset manifest inherits exactly the universal columns it uses. Which
features a ticker keeps remains a separate per-asset decision; only the weight initialization is
shared.

**Accepted bias.** The shared init has seen Train rows beyond any single fold's boundary (never
OOS), so absolute Train-CV scores are upper-biased and must not be compared across different
inits. What stays fair: every subset and grid comparison shares the identical init, so rankings
and the operating-point choice are unaffected in ordering; the final honest read remains the
ledgered OOS read.

## 6. Known biases and limitations

| bias / limitation | status |
|---|---|
| Superset-HPO gain inflation (XGB) | Accepted, guarded. HPO params are frozen on the superset, co-adapted to features the core baseline lacks, so reported search *gains* are upper-biased in magnitude. Subset ranking stays fair (identical events, folds, params); `min_gain` and fold-win gates absorb the inflation; the sealed run re-runs HPO fresh on the chosen manifest, so nothing co-adapted is deployed. |
| Universal warm-start init (LSTM) | Accepted, guarded (section 5). |
| Survivorship | The universe is built from present-day S&P 500 constituents; delisted losers are absent. Aggregate results are optimistic. |
| Single data feed | One equities feed; no cross-vendor reconciliation of bars. |
| Corporate actions | Prices are **split-adjusted** from a reviewed event table (83 events / 69 tickers) applied at the 1h level before any roll-up. Spin-offs and special dividends are deliberately NOT adjusted — for a price benchmark they are real drops, the same class as a dividend. Splits below 3:2 are not detectable from bars and are missed unless entered by hand. |
| Barrier timing | Barriers are evaluated on closes. Measured effect: the 1xATR stop is pierced intra-bar more often than the 2xATR target, so the close-scan is net **conservative** on win-rate by roughly 5 pp. |
| Bull-market OOS window | 2024-2026 is a strongly rising market; HODL is a hard benchmark and the beats-HODL ratio is regime-dependent. |
| In-sample interpretation | All ENTRY ranges, contributions and occlusion values are Train-derived descriptions of the sealed model, not OOS evidence. |

A broader audit of the whole research program (in Polish), including the full
declared-limitations list and the measured read-ledger figures, lives in
`docs-facts-infos/Raport_Spojnosci_Badan.md`, next to the data audit
(`Dane_OHLCV.md`) and the methodological-integrity audit
(`Raport_Poprawnosci_Metodologicznej.md`).

## 7. The ledgered OOS read

OOS data never feeds back into any choice — not HPO, not feature selection, not the operating
point, not even the ML-vs-HODL verdict wording. Feature search runs on Train-CV only; the
sealing pass scores each asset at the verdict step, and every read of the OOS window —
including re-reads from an interrupted, resumed pass — is recorded in an append-only ledger.
`oos_read_summary` in the sealed store carries that ledger summarised per pipeline: reads in
this epoch, and the min / mean / max of the cumulative per-asset counter (the per-asset rows
themselves stay in the research tree's ledger files). Each asset's
outcome is classified by `result_mode` (see ARCHITECTURE.md for the full enum): multi-trade
result, single-trade low-evidence result, HODL fallback when the model produced zero OOS
trades, or an explicit not-promoted verdict when the Train-OOF trade floor was never met (such
an asset usually never trades OOS, but it can trade and simply not be promoted). Profit factor is always
reported with coverage (how many assets have at least 2 model trades), never as a bare mean.

## 8. The interpretation layer

A read-side description of the sealed models on their own Train window — pure math, no
training, no OOS rows, no writes into decision artifacts. Its identity is pinned by an
`interpretation_recipe_hash` (algorithm, frozen thresholds, sigma methods, rounding policy,
source hash) carried in every payload and database row.

### 8.1 XGB: split thresholds, total gain, ENTRY ranges

- **Contribution** = *XGB split total-gain contribution*: the SUM of the gains of every split of
  a feature across the ensemble (the loss reduction it accounts for), normalized to shares —
  never a mean per split.
- **ENTRY ranges**: each feature's axis is cut at the model's own split thresholds (the only
  points where the decision function can change along that feature) and merged into segments.
  ALL segments with sufficient Train support (>= 20 rows) are recorded — including the ones
  where the model does not enter; there is no cherry-picking of the Train distribution.
  `candidate_entry_region` (entry lift >= 1.25 and >= 10 ENTRY events) is a highlight flag,
  never a storage filter. Each segment carries its bounds in raw feature values AND in sigmas,
  plus the TP-before-SL rate among its ENTRY rows.
- **Semantics**: these are *model-derived conditional ENTRY regions* — a projection of
  whole-model behaviour onto one feature (splits depend on earlier splits of other features) —
  never standalone causal rules.

### 8.2 LSTM: occlusion and sequence trajectories

The LSTM has weights, not trees, so there are no thresholds to harvest. Instead:

- **Deterministic occlusion** per input channel over one frozen set of all purged Train windows
  in fixed order; the *primary* view is ENTRY-conditioned (windows where p >= theta), the
  *secondary* view is global.
- **Sequence trajectories**: per-channel quantile bands over the 60-step input windows and t0
  quantiles, showing the sequence of market states that historically preceded ENTRY.

### 8.3 The sigma basis difference (must never be blurred)

| model | what sigma means |
|---|---|
| XGB | *descriptive* standardization of the Train candidate-row distribution — presentation aid only; the model consumes raw feature values |
| LSTM | the *actual* input transform of the model — the frozen per-asset normalization statistics embedded in the sealed artifact |

The two pipelines must never be presented as using "the same mechanism" here.

### 8.4 Mandatory labeling

Every interpretation view renders the banner carried in the payload itself:

```text
TRAIN-DERIVED INTERPRETATION · NOT AN OOS RESULT · NOT A LIVE TRADING SIGNAL
```

Quantiles and rates with fewer than 10 supporting ENTRY events carry a LOW_EVIDENCE flag next
to the number. Statuses are text, never color alone.
