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
  -> Streamlit console  (nine pages)
```

## Quickstart

```bash
git clone --depth 1 --branch Stable_Presentable_Version \
  https://github.com/flak92/liora-project-ml-engineering.git

cd liora-project-ml-engineering

make setup
make app
```

The app serves on `http://localhost:8503`. The shallow clone is ~260 MB (every sealed
artifact travels with the repo — 993 in this release). `make setup` installs only the presentation
dependencies (`streamlit`, `pandas`, `plotly`); nothing is trained, recomputed or
written at runtime.

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

## The nine pages

1. **Overview** — what the study is, Train/OOS timeline, median outcomes, main finding.
2. **Universe** — one operational table over all assets; search, filter, jump to an asset.
3. **Asset Indicator** — the dedicated indicator of one asset: results, calibrated
   threshold, direction mode, selected features, artifact path.
4. **Feature Logic** — what each sealed model reads: XGB ENTRY ranges, LSTM occlusion
   and trajectories (Train-derived interpretation).
5. **Model Comparison** — four charts: return, profit factor, trades, beats-HODL share.
6. **Architecture** — the data flow above plus a map from the presentation to the code.
7. **Integrity** — the dataset's own record: epoch and recipe hashes, the frozen
   parameters, the OOS read ledger (reads per pipeline, and the spread of the cumulative
   per-asset counter), interpretation coverage, when the model is not promoted, every
   integrity check, and the known limits.
8. **Pipeline Blueprint** — the procedure as an 18-brick ladder: contract, reasoning and
   lesson per brick, with the layer id the code uses (XGB L1-L9, LSTM D1-D9).
9. **Data Flow** — the per-asset build path as a 2.5D canvas map, both pipelines in one
   ladder, with the universe-level verdicts on the OOS scenes.

## Repository structure

```text
app.py            Streamlit entry point (nine pages under app/pages/)
app/              console code; app/data.py is the ONLY module opening the database
src/xgb/          XGB research code (pipeline L4-L9, feature search, artifact writers)
src/lstm/         LSTM research code (pipeline D1-D6, model D7-D8, feature search)
src/shared/       contracts shared by both pipelines (op_select, golden_calibration,
                  interpretation)
config/           frozen configuration the code reads
artifacts/        sealed per-asset artifacts (xgb/<T>/, lstm/<T>/) + manifest.json
data/results.db   sealed SQLite results store (read-only)
examples/         two executed notebooks: the full XGB path for AAPL and NVDA
docs/             METHODOLOGY.md, ARCHITECTURE.md
docs-facts-infos/ written audits (Polish): OHLCV data, methodological integrity,
                  and the research-consistency report
pipeline_lego_blueprint.html   standalone 18-brick pipeline map (embedded by the Blueprint page)
data_flow_3d.html              standalone 2.5D build-path map (embedded by the Data Flow page)
```

The code under `src/` is the real research code, kept complete and importable for
reading; running it end-to-end would additionally require the training stack
(`torch`, `xgboost`, `optuna`, `duckdb`) and the raw bar stores, which are not part
of this presentation branch.

## Limitation

All results are historical out-of-sample reads of sealed models over fixed windows —
every OOS read is counted in an append-only ledger (Integrity page). They are research
output — **not a live trading signal** and not investment advice. The interpretation layer is Train-derived and must not be read as an OOS
result.
