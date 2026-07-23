# Golden Calibration — automatic OHLCV feature discovery

This branch does not start from a model or a backtest. It starts from a question:

> **How do you automatically, reproducibly and cheaply go from raw OHLCV to the smallest confirmed set
> of features for a given asset — without mistaking the maximum of a wide search for a real edge?**

The answer is a **Calibration DAG**: a ladder of questions where each rung has a statistic, a
threshold fixed before the data is seen, a negative control, a stop condition, and one admissible
next step. A feature is not "good" because it ranked high; it is confirmed only if it survives being
chosen and judged by data that played no part in choosing it — and if it beats the maximum that
searching a pool of candidates produces by itself.

## The funnel — the whole story in four numbers

On the twenty-asset development panel, the ladder discriminated hard:

```
26 provisional (cross-fit accepted)
   → 11 passed the procedure-level max-null (marginal)
      → 9 stable across all three nulls (marginal × regime × conditional)
         → 2 retained after survivor-specific tuning
            → 1 unique feature (representative 112)
```

The retained set is **both `ORLY/1` arms** — `flat 112` and `hierarchical oscillator_rsi` — two
discovery paths that resolve to the *same* feature (representative 112): stable across every null
*and* retained once the model was allowed to tune around it, each decisively (b = 0/50 and 1/50
against its own tuning null, p = 0.020 and 0.039). Everything else showed dependence on the asset,
the regime, or the tuning. That is the point of the method: it does not chase one universal
strategy, it shows **which OHLCV relationships hold, where, and how strongly** — and returns an empty
set, honestly, when the evidence does not support one.

These numbers are a *snapshot* of one research run, not a target. A fresh panel may produce
`30 → 7 → 2 → 0` and be exactly as correct — the funnel is a property of the data, computed from
artifacts, never hard-coded.

```bash
make methodology-report        # prints the funnel from the frozen snapshot, in a blink
```

## Two layers, kept apart

**A — the science.** The Calibration DAG, its statistics, the frozen contract, the negative and
positive controls, the acceptance rules, the stop conditions. This is what decides what is true. It
lives in [`scripts/`](scripts/) and [`config/`](config/) and is described in
[`docs/FEATURE_DISCOVERY_METHODOLOGY.md`](docs/FEATURE_DISCOVERY_METHODOLOGY.md).

**B — the execution engine.** A planner, a task queue, a pool of workers, a tmux session, resume,
guards and two technical ledgers. This is what *runs* the science across many assets — long,
parallel, deterministic, without an operator hand-picking favourable results. It lives in
[`engine/`](engine/) and [`ops/`](ops/) and is described in
[`docs/METHODOLOGY_ENGINE.md`](docs/METHODOLOGY_ENGINE.md).

The engine executes the science; it never changes it. Every transition between rungs comes from an
immutable result artifact and the frozen contract — never from the scheduler, spare CPU, another
asset's result, or a human nudging the queue. tmux and cron are how the proof is *carried out*, not
where the proof lives.

## The Calibration DAG

| Rung | Question | Unit |
|---|---|---|
| 0 | Is the problem frozen — data, labels, splits, the OOS boundary, the hashes? | run |
| 1 | Can the model learn at all? | asset |
| 2 | Does the operating point transfer? *(folded into 3–4 here)* | asset |
| 3 | Does a feature improve a model that can learn? | asset |
| 4 | Does the choice survive data that did not choose it? | asset |
| 5 | Is the edge bigger than the maximum a search produces by itself? | asset |
| 6 | How much more is a survivor worth under its own tuned model? | asset |
| 7 | Do survivors combine — interactions and sequences? | asset · *specified, unvalidated* |
| 8 | Which OHLCV families travel across assets? | panel |
| 9 | Does the whole method hold on a fresh panel? | new panel |

Each asset walks this ladder as a state machine:

```
PENDING_VIABILITY → VIABLE → OPERATING_POINT_VALID → UTILITY_REGISTERED → CONFIRMED
   → NULL_VALIDATED → LOCALLY_OPTIMIZED → INTERACTIONS_EVALUATED → RESOLVED
```

and, just as validly, into `RESOLVED_EMPTY` (nothing survived) or `NEEDS_CONTRACT` (the current rules
cannot honestly continue — a human must mint a new contract version; the engine never does it itself).

## Two ways to use the branch

```bash
# PRESENTATION — read the frozen development-panel artifacts, print the funnel and per-asset verdicts.
make methodology-report

# REPRODUCTION — run the Calibration DAG per asset in a detached tmux session, produce new artifacts.
make engine-smoke              # the full DAG on three assets (a validation run)
make engine-start ASSETS="…"   # the full panel, detached; survives closing the terminal
make engine-status             # session, queue, per-asset states, ledgers, memory
make engine-plan               # the deterministic plan, read before anything runs

make engine-selftest           # prove the execution guarantees (no science runs)
```

## The five pages

`make on` opens a read-only Streamlit console — five pages, flat sidebar, counts derived from the store,
nothing trains at runtime:

- **Data Flow 3D Visualization** — the pipeline as a moving graph.
- **Overview** — what came out of this: per-asset verdicts and the funnel.
- **Basket Simulator** — build a basket and read what the sealed models did with it.
- **Data Pipeline Lego Plan** — the procedure as an 18-brick board.
- **Calibration Configurables** — every tunable number as a range, its FROZEN/ADMISSIBLE state, and which knob to widen when a step comes up empty.

## Reproducibility — what runs from a fresh clone, and what does not

Two dependency tiers, deliberately separate:

- **Presentation** needs only `pip install -r requirements.txt` (streamlit/pandas/plotly). It reads the
  frozen artifacts under `results/methodology_snapshot/` and prints the funnel — no numeric stack, no data.
- **Reproduction** additionally needs `pip install -r requirements-research.txt` (numpy, xgboost, optuna,
  duckdb, scikit-learn, …) — the stack the Calibration DAG, the null and the engine actually import.

Honest limits of a fresh clone: the pipeline source (`xgb/src`, `xgb/tools`) and the bar store
(`xgb/data/liora.duckdb`) live under the top-level `/xgb/` tree, which is **gitignored** — so a clone can
install the deps and read every published artifact, but cannot run a *new* research pass without that
private tree. The methodology, the frozen contract, the per-asset artifacts and the selftests are fully
public; the raw bars and the deployable pipeline code are not. Every calibratable number and its range is
catalogued in [`docs/CALIBRATION_CONFIGURABLES.md`](docs/CALIBRATION_CONFIGURABLES.md).

## What the project is actually about

The most important result is not a configuration or a backtest. It is an **auditable process** that,
for each asset, builds its own mathematical description, selects stable features, calibrates parameter
ranges to the geometry of the data, freezes the rules before OOS, and shows honestly both where the
model works and where it should stay idle. The full argument is in
[`docs/SUMMARY.md`](docs/SUMMARY.md).

---

*This is the `methodology` branch — the executable method and its engine. The `main` branch is the
sealed-model presentation console; this branch answers how those models are discovered, not how they
are shown.*
