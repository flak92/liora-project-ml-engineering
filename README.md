# Golden Calibration — automatic OHLCV feature discovery

*Golden Calibration = **smart convergence**, not a blind search for one best configuration.*

This branch does not start from a model or a backtest. It starts from a question:

> **How do you automatically, reproducibly and cheaply go from raw OHLCV to the smallest confirmed set
> of features for a given asset — without mistaking the maximum of a wide search for a real edge?**

The method trains an estimator (XGB / LSTM) with Optuna, **calibrates its configuration ranges to the
geometry of the data**, and walks a finite, human-authored ladder of hypotheses **only while a new
attempt still confirms a feature** — then stops (convergence). A feature is confirmed only if it
survives being chosen and judged by data that played no part in choosing it, and beats the maximum a
search produces by itself. The proof standard is frozen: the loop can never loosen its own bar.

**Single source of truth: [`docs/SMART_METHODOLOGY.md`](docs/SMART_METHODOLOGY.md).**

## The funnel — the whole story in five numbers

```
26 provisional (cross-fit accepted)
   → 11 passed the procedure-level max-null (marginal)
      → 9 stable across all three nulls (marginal × regime × conditional)
         → 2 retained after survivor-specific tuning
            → 1 unique feature (representative 112)
```

Both retained arms — `ORLY/1 flat 112` and `ORLY/1 hierarchical oscillator_rsi` — resolve to the *same*
feature (rep 112), decisively against their own tuning null (b = 0/50 and 1/50). These numbers are a
snapshot of one run, not a target: a fresh panel may produce `30 → 7 → 2 → 0` and be exactly as correct.
**Certification: NOT STARTED** — this panel shaped the method, so it cannot certify itself.

```bash
make methodology-report        # prints the funnel from the frozen snapshot, in a blink
```

## Two layers, kept apart

**A — the science.** The Calibration DAG, its statistics, the frozen contract, the negative and positive
controls, the acceptance rules, the stop conditions. This decides what is true. It lives in
[`scripts/`](scripts/) and [`config/`](config/).

**B — the execution engine.** A per-asset driver that runs the science across many assets — long,
parallel, deterministic, without an operator hand-picking favourable results. It lives in
[`engine/`](engine/) and [`ops/`](ops/). The engine executes the science; it never changes it. Every
transition between rungs comes from an immutable artifact and the frozen contract.

## The two pages

`make on` opens a read-only Streamlit console — two pages, flat sidebar, counts derived from the
store, nothing trains at runtime:

- **Data Journey** — the methodology as the road the data travels, because you cannot show the method
  without showing how the data is prepared first: raw OHLCV → warmup → Train/OOS split (purge + embargo,
  oos_reads = 0) → feature search (Train-only) → the rung ladder → OOS verdict, on seven real assets, one
  arc through every terminal of the funnel.
- **Smart Methodology** — the calibration-configurables map and the run replay (the field-level guard
  runs live on every visit), in two tabs.

The sealed-model product console (per-asset outcomes across the 498/495 universe, the basket simulator)
lives on the `main` branch; this branch is the method, not the sealed models.

## Reproducibility — what runs from a fresh clone, and what does not

- **Presentation** needs only `pip install -r requirements.txt` (streamlit/pandas/plotly). It reads the
  frozen artifacts under `results/methodology_snapshot/` and prints the funnel — no numeric stack, no data.
- **Reproduction** additionally needs `pip install -r requirements-research.txt` (numpy, xgboost, optuna,
  duckdb, scikit-learn). Honest limit: the pipeline source (`xgb/src`) and the bar store
  (`xgb/data/liora.duckdb`) live under a **gitignored** `/xgb/` tree — a clone reads every published
  artifact but cannot run a *new* research pass without that private tree. Every calibratable number and
  its range is catalogued in [`docs/archive/CALIBRATION_CONFIGURABLES.md`](docs/archive/CALIBRATION_CONFIGURABLES.md).

## Deep-dive

The full ordered method, the loop mechanics, the configurables catalog, the engine, the two pipeline maps
and the Polish audits now live under [`docs/archive/`](docs/archive/), reachable from the SSOT.
Operational notes: [`docs/RUNBOOK.md`](docs/RUNBOOK.md) · architecture:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Polish summary: [`docs/SUMMARY.md`](docs/SUMMARY.md).

---

*This is the `methodology` branch — the executable method and its engine. The `main` branch is the
sealed-model presentation console; this branch answers how those models are discovered, not how they
are shown.*
