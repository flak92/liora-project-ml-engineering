# Smart Methodology — Golden Calibration

*The single source of truth for the method. Everything revolves around one loop:*
**train an estimator (XGB / LSTM) with Optuna → calibrate its configuration ranges → keep going only while a new attempt still confirms a feature → stop, and gate every claim on data that played no part in choosing it.**

> **PL TL;DR.** Nie szukamy „najlepszej" konfiguracji na ślepo. Uczymy XGB/LSTM (Optuna), **kalibrujemy zakresy configurables do geometrii danych**, i przechodzimy **z góry autoryzowaną drabinę hipotez tak długo, jak kolejna próba dodaje potwierdzoną cechę** — potem stop (smart convergence). Standard dowodu (OOS, nulle) jest **zamrożony**: pętla nigdy nie luzuje własnej poprzeczki. „Golden Calibration" = *best-observed stable*, **nie** ślepe globalne optimum.

---

## 0. Thesis · status · seal

The most important result is **not a configuration** — it is an auditable process that, per asset, builds its own OHLCV description, calibrates parameter *ranges* to the data, freezes the rules before OOS, and returns an honest empty set when the evidence does not support a feature. This panel is a **development panel, not a certification**.

Status: `XGB = VALIDATED` · `LSTM = PENDING` · Rung 7/9 `SPECIFIED / UNVALIDATED` · **Certification: NOT STARTED** (a panel that shaped the method cannot certify itself; the name *Golden Calibration v1* is earned only by Rung 9 on a fresh panel).

<!-- CALIBRATION-SEAL {"contract_hash": "c45d57b3b60389ea", "contract_version": "golden-calibration-dc.1", "data_boundary": ["2023-12-29", "2024-01-02"], "funnel": [26, 11, 9, 2], "retained_b_over_M": ["0/50", "1/50"], "retained_representatives": [112], "retained_units": ["ORLY/1/flat/112", "ORLY/1/hierarchical/oscillator_rsi"], "seed": 42} CALIBRATION-SEAL -->

> **Seal.** Every headline number is recomputed from the frozen contract + snapshot by `make verify-calibration-docs`; if either drifts and this doc is not refreshed, the lint goes red. `contract_hash c45d57b3b60389ea` · `golden-calibration-dc.1` · seed 42 · train ≤ 2023-12-29 / oos ≥ 2024-01-02.

## 1. What we train, and what tunes it

The **trained objects** are per-asset estimators; the **tuner** is Optuna (with a research random-search twin). Two HPO regimes coexist — do not conflate them:

- **A2 — sealed deployable model:** Optuna **TPE + MedianPruner**, `N_TRIALS = 80`, `cv_folds = 4` (inner purged walk-forward), deterministic at seed. This is the shipped XGB/LSTM.
- **A1 — research Calibration-DAG:** **random search** over a **hessian-relative** space (`gamma`, `min_child_weight` are multiplied by the fold's own hessian `H`, so one relative value means the same thing on every asset), `hpo_trials = 30`.

`XGB` runs on 1h bars; `LSTM` on 1d, kept simple/regularized because independent daily sequences are scarce. Both walk the **same ladder** (§3); LSTM has not yet produced a result through it.

## 2. Configuration Calibration — four classes of number

The whole discipline is that every number is exactly one of four kinds. **Calibratable** numbers are the ranges that move with the data (this is "Configuration Calibration"); **Methodology-governing** numbers are the frozen proof standard.

| Class | What it is | Range? | Fixed at |
|---|---|---|---|
| **Structural** | interval, label horizon `H=24`, barriers, costs, embargo, the OOS ban | one frozen value | Rung 0 (`data_boundary`) |
| **Calibratable** | `gamma`, `min_child_weight`, θ-grid, block length, permutation count, HPO budget | **data-fitted range**, never one absolute number across assets | per-asset, from its own data |
| **Searched** | features, families, interactions, arms | the *objects* the ladder looks for | Rungs 3→8 |
| **Methodology-governing** | viability floor, rotations, α, early-stop | the **proof standard** — frozen so a null result cannot be tuned away | contract kernel |

Widening a **Calibratable** range is a legal hypothesis move; widening a **Methodology-governing** number is a *different science* and is forbidden as an automatic action. Full catalog: [`docs/archive/CALIBRATION_CONFIGURABLES.md`](archive/CALIBRATION_CONFIGURABLES.md).

## 3. The ladder — Rung 0–9

A feature is not "good" because it ranked high; it is confirmed only if it survives being chosen and judged by data that played no part in choosing it, and beats the maximum a search produces by itself. Each asset walks this as a state machine (`engine/states.py`), driven by immutable artifacts + the frozen contract — never by the scheduler, spare CPU, or another asset.

| Rung | Question | Unit |
|---|---|---|
| 0 | Is the problem frozen — data, labels, splits, OOS boundary, hashes? | run |
| 1 | Can the model learn at all? | asset |
| 2 | Does the operating point transfer? *(folded into 3–4)* | asset |
| 3 | Does a feature improve a model that can learn? | asset |
| 4 | Does the choice survive data that did not choose it? | asset |
| 5 | Is the edge bigger than the maximum a search produces by itself? | asset |
| 6 | How much more is a survivor worth under its own tuned model? | asset |
| 7 | Do survivors combine — interactions and sequences? | asset · *specified, unvalidated* |
| 8 | Which OHLCV families travel across assets? | panel |
| 9 | Does the whole method hold on a fresh panel? | new panel |

**Four ways a candidate dies** (why the funnel narrows): *search-inflation* (the edge is only the max of a wide search — Rung 5 max-null), *regime-dependence*, *tuning-dependence* (does not hold once the model tunes around it — Rung 6), *asset-specificity* (does not travel — Rung 8). Full method: [`docs/archive/FEATURE_DISCOVERY_METHODOLOGY.md`](archive/FEATURE_DISCOVERY_METHODOLOGY.md).

## 4. Golden Calibration = smart convergence (not blind search)

`best point != golden calibration`. The loop does not hunt a global optimum; it walks a **finite, human-authored ladder** of pre-authorized hypotheses (`config/iteration_loop_policy.json`) and **stops when a new attempt no longer changes the answer**:

- A **confirmed feature** is a null-validated stable survivor (A1∩A2∩B), a pair `(asset, unit)`. The cumulative set is monotone (Rung 6 tunes survivors, adds none). After ≥1 variant epoch, if `patience` epochs add nothing new → **`CONVERGED`**.
- **Variation, not widening.** A rung varies only the ADMISSIBLE hypothesis space (`model_space`, `operating_point`, `arms`, `rung_6_survivor_hpo`, `rung_7_interactions`). A negative result **never loosens** the bar — it advances to the next pre-declared hypothesis, or returns `NEEDS_CONTRACT` for a **human** to mint the next version. Auto-widening is co-adaptation and is forbidden.
- `engine/contract_patch.guard` enforces this **field by field**: an admissible section cannot smuggle a frozen leaf (`rung_6_survivor_hpo.alpha`, `own_null.permutations`, `operating_point.mode`). The convergence policy is deliberately **outside** `contract_hash` — it governs *how* the loop computes, never *whether* a feature is real.

Loop mechanics deep-dive: [`docs/archive/ITERATIVE_CALIBRATION_LOOP.md`](archive/ITERATIVE_CALIBRATION_LOOP.md).

## 5. The honest gate — OOS + three nulls (frozen)

Every confirmation is judged by the frozen proof standard; OOS reads stay `0` in every epoch.

| Gate | Value | Rung |
|---|---|---|
| viability floor | `split_nodes ≥ 20`, `pred_std ≥ 0.005` | 1 |
| acceptance | `min_rotations = 2`, `complexity_penalty = 0.004`, majority-positive | 4 |
| max-null (marginal × regime × conditional) | **M = 50**, **α = 0.10**, pass `b ≤ 4` (`p_mc = (1+b)/51`), futility at `b = 5` | 5 |
| rung-6 own-null | **M = 50**, `α = 0.10` (= `max_null.alpha`), retain iff `b ≤ floor(α·(M+1)−1) = 4` | 6 |

`α = 0.10` is one lab constant shared by `max_null` and `rung_6.own_null`; the pass-count bound `b` is *derived* from α and M, never hard-coded. `H = 24` is one number in four places (label horizon, `data_boundary`, null block length, Kelly geometry) — moving it invalidates labels, folds and the null at once.

## 6. Result — the whole story in five numbers

```
26 provisional (cross-fit accepted)
   → 11 passed the procedure-level max-null (marginal)
      → 9 stable across all three nulls (marginal × regime × conditional)
         → 2 retained after survivor-specific tuning
            → 1 unique feature (representative 112)
```

Both retained arms — `ORLY/1 flat 112` and `ORLY/1 hierarchical oscillator_rsi` — resolve to the *same* feature (rep **112**), decisively against their own tuning null: `b = 0/50` (p = 0.020) and `b = 1/50` (p = 0.039). The 7 demoted arms hit `b = 5` early — **CENSORED** by futility. Everything else showed dependence on the asset, the regime, or the tuning. These numbers are a **snapshot of one run, not a target** — a fresh panel may produce `30 → 7 → 2 → 0` and be exactly as correct.

## 7. Present · Reproduce · Deep-dive

```bash
make methodology-report        # print the funnel from the frozen snapshot, in a blink
make on                        # read-only Streamlit console (Overview · Simulator · Smart Methodology)
make engine-smoke              # the full DAG on three assets (a validation run)
```

- **Present:** `results/methodology_snapshot/` + `make methodology-report` (no numeric stack, no data).
- **Reproduce:** additionally `pip install -r requirements-research.txt` (numpy, xgboost, optuna, duckdb, sklearn). Honest limit: the raw bar store and deployable pipeline live under a gitignored `/xgb/` tree — a clone reads every published artifact but cannot run a *new* pass without that private tree.
- **Deep-dive:** [`docs/archive/`](archive/) (full method, loop, configurables, engine, PL audits) · [`docs/RUNBOOK.md`](RUNBOOK.md) · [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) · Polish summary [`docs/SUMMARY.md`](SUMMARY.md).
