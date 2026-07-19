# S&P 500 ML Indicator Study

One dedicated, sealed machine-learning ENTRY indicator per S&P 500 asset — XGBoost on
1-hour bars and an LSTM on daily bars, evaluated once out-of-sample against buy-and-hold —
presented through a read-only Streamlit console.

## Architecture

```text
OHLCV (1h / 1d)
  -> features / sequences
  -> Train-only calibration (theta, trade floor, OOF operating point)
  -> XGB | LSTM  (sealed per-asset models)
  -> per-asset artifact  (strategy + manifest + parameters + metrics + interpretation)
  -> data/results.db  (SQLite, read-only)
  -> Streamlit console  (seven pages)
```

## Quickstart

```bash
git clone --depth 1 --branch Stable_Presentable_Version \
  https://github.com/flak92/liora-project-ml-engineering.git

cd liora-project-ml-engineering

make verify     # optional, stdlib only: recompute every artifact hash
make setup
make on
```

The app serves on `http://localhost:8503`; `make off` stops it again (it kills only the
process listening on that port), and both accept `PORT=…` if 8503 is taken. The shallow clone is ~260 MB (every sealed
artifact travels with the repo — 993 in this release). `make setup` installs the presentation
dependencies (`streamlit`, `pandas`, `plotly`) plus `anthropic`, which only the optional
Formular questionnaire on the Basket Simulator uses and which needs no key unless you open
it; nothing is trained, recomputed or written at runtime.

Do not take the numbers on trust: `make verify` needs no dependencies and no network. It
recomputes the SHA-256 of every file in all 993 artifact folders, rebuilds each
`folder_sha256` from those digests, checks the manifest's count arithmetic, and confirms
every sealed row resolves to a folder the manifest knows.

## The two models

- **XGBoost (1h)** — per asset: engineered 1h features (with 1d/1w roll-ups), an ATR
  triple-barrier label, Train-only HPO and threshold calibration. The interpretation
  layer projects the sealed model into per-feature ENTRY value ranges (raw and in
  Train sigmas) with TP-before-SL rates.
- **LSTM (1d)** — per asset: 60-session sequences of normalized daily state channels,
  deterministic CPU training warm-started from a universal backbone. The
  interpretation layer measures channel occlusion (ENTRY-conditioned vs global) and
  state-sequence trajectories.

Both models only decide ENTRY; take-profit and stop-loss are a mechanical ATR
triple-barrier contract. An asset with no robust Train operating point stays idle by
design. See `docs/METHODOLOGY.md`.

## The seven pages

The sidebar groups them the way a defence walks: **Playground** (do something), **Results**
(what the models did), **Method & proof** (how it was built, and how to check it).

1. **Basket Simulator** — pick assets, by preset or by hand, and read what the sealed
   models did with them against the same basket simply held. Three numbers, never one:
   the executed path, the model result, and the price-only benchmark.
2. **Overview** — the universe verdict in one table: median return against each model's
   own buy-and-hold over the same window, how many assets beat it, and median profit
   factor with its coverage.
3. **Feature Logic** — what each sealed model reads: XGB ENTRY ranges and LSTM channel
   occlusion (Train-derived interpretation).
4. **Model Comparison** — four charts: return, profit factor, trades, beats-HODL share.
5. **Integrity** — the dataset's own record: epoch and recipe hashes, the frozen
   parameters, the OOS read ledger (reads per pipeline, and the spread of the cumulative
   per-asset counter), interpretation coverage, when the model is not promoted, every
   integrity check, and the known limits.
6. **Data Pipeline Lego Plan** — the procedure as an 18-brick ladder: contract, reasoning
   and lesson per brick, with the layer id the code uses (XGB L4-L9, LSTM D1-D9).
7. **Data Flow 3D Visualization** — the build path drawn twice: the whole study as eight
   boxes, then the same path as a 2.5D canvas map — sixteen levels, both pipelines in one
   ladder, every contract a click away.

## Repository structure

```text
app.py            Streamlit entry point (seven pages under app/pages/, in three sections)
app/              console code; app/data.py is the ONLY module opening the database
src/xgb/          XGB research code (pipeline L4-L9, feature search, artifact writers)
src/lstm/         LSTM research code (pipeline D1-D6, model D7-D8, feature search)
src/shared/       contracts shared by both pipelines (op_select, golden_calibration,
                  interpretation)
config/           frozen configuration the code reads
artifacts/        sealed per-asset artifacts (xgb/<T>/, lstm/<T>/) + manifest.json
data/results.db   sealed SQLite results store (read-only)
examples/         two executed notebooks for one asset (NVDA), one per model:
                  Example_XGB.ipynb (L4→L9) and Example_LSTM.ipynb (D2→D9)
scripts/          the offline verifiers behind `make verify`: artifact hashes and
                  the notebooks' parity with the store
docs/             METHODOLOGY.md, ARCHITECTURE.md
docs-facts-infos/ written audits (Polish): OHLCV data, methodological integrity,
                  and the research-consistency report
data_pipeline_lego_plan.html   standalone 18-brick pipeline map (embedded by the Lego Plan page)
data_flow_3d_visualization.html  standalone 2.5D build-path map (embedded by the 3D Visualization page)
```

The code under `src/` is the real research code that produced and describes the sealed
artifacts: both pipelines, the feature searches, the artifact writers and the shared
contracts, unmodified and readable. The acquisition and orchestration layer around it —
bar loading with the corporate-action correction, the per-asset runner, the compute-run
harness — stays on the research branch, together with the raw bar stores and the training
stack (`torch`, `xgboost`, `optuna`, `duckdb`). So `src/` is here to be read and audited,
not to re-run the universe; what you can re-verify on this branch is the artifact tree
(`make verify`) and the two executed notebooks under `examples/`, one per model on the
same ticker.

## Limitation

All results are historical out-of-sample reads of sealed models over fixed windows —
every OOS read is counted in an append-only ledger (Integrity page). They are research
output — **not a live trading signal** and not investment advice. The interpretation layer is Train-derived and must not be read as an OOS
result.
