# ARCHITECTURE

Frozen research snapshot (this release: 498 XGBoost + 495 LSTM = 993; the counts are declared
in `research_run` and verified against the store, never hardcoded in the app): sealed per-asset artifacts, one results
database, one read-only Streamlit console. All 16/16 integrity checks PASS.

## 1. Data flow

```text
OHLCV bars
  -> features (XGB) / normalized sequences (LSTM)
  -> Train-only calibration (purged walk-forward CV, shared operating-point selection)
  -> XGB pipeline (L4-L9)  |  LSTM pipeline (D1-D9)
  -> per-asset sealed artifact (5 files, hash-manifested)
  -> data/results.db (collector output, sealed)
  -> Streamlit console (read-only, fail-closed)
```

No component to the right ever writes to the left. The app trains nothing, reads no OOS store,
and writes nothing.

## 2. XGB pipeline layers (src/xgb/pipeline.py)

| layer | one sentence |
|---|---|
| L4 | Bar snapshot to a clean 1h frame plus deterministic 1d/1w roll-ups, with fail-closed source QC. |
| L5 | Warmup / Train / OOS time split, with purge (= label horizon) and embargo at every boundary. |
| L6 | Causal entry candidates and asymmetric ATR Triple-Barrier labels become the "Output B" feature matrix. |
| L7 | Optuna tunes XGBoost by maximizing Train out-of-fold trading log-growth through the real engine. |
| L8 | Per-asset operating-point calibration (theta_entry via shared op_select), final fit on full Train, and the self-contained `strategy_<TICKER>.py` artifact (base64 booster + selfcheck). |
| L9 | The ledgered OOS read with HODL fallback on zero trades, producing exactly one metrics row. |

## 3. LSTM pipeline phases (src/lstm/pipeline.py, model.py)

| phase | one sentence |
|---|---|
| D1 | The daily bar store is the frozen input; no acquisition happens in the pipeline. (The store itself stays in the unpublished research tree — this repository ships the sealed artifacts it produced.) |
| D2 | Load one ticker ordered by date with fail-closed source QC — corrupt OHLCV raises, never gets cleaned. |
| D3 | Warmup / Train / OOS masks with purge (= label horizon) and embargo on events. |
| D4 | Causal daily indicators, z-scored with Train-only per-asset statistics. |
| D5 | Momentum-sided candidates and asymmetric ATR Triple-Barrier labels (entry next open, costs both sides, label-uniqueness weights). |
| D6 | The per-candidate sequence tensor: the SEQ_LEN x n_features window of normalized features ending at t0. |
| D7 | The LSTM classifier with deterministic CPU training (BCE loss weighted by class balance and label uniqueness). No per-asset Optuna ran in this epoch: one committed universal backbone supplies the architecture and warm-starts every refit; the cold-start study exists behind `LSTM_COLD_START=1`. |
| D8 | Operating-point calibration (theta_entry, direction mode) via shared op_select, then the final refit and the sealed strategy artifact. |
| D9 | The ledgered OOS read: the frozen window is scored into one metrics row per asset, every read recorded in the append-only ledger. Nothing downstream may feed back into D1-D8. |

## 4. Shared modules (src/shared/)

- `op_select.py` — ONE accumulate-then-select operating-point implementation for both pipelines
  (trade floor, fold spread, one-SE plateau, conservative ties), so scorers cannot drift.
- `golden_calibration.py` — pure search-policy helpers: family maps (fail-closed), one-SE
  plateau, simplest-representative choice; acceptance gates stay in the workers.
- `interpretation.py` — pure interpretation math: split harvest, total-gain shares, segment
  construction, sigma handling, and the `interpretation_recipe_hash` provenance identity.

## 5. Per-asset artifact contract

Each `artifacts/{xgb,lstm}/<TICKER>/` folder contains exactly five files:

| file | content |
|---|---|
| `strategy_<TICKER>.py` | Standalone strategy: the model embedded as base64 with `MODEL_HASH`, the feature manifest, `THRESHOLD_ENTRY` (the calibrated per-asset theta), the full calibration record, the execution contract, and a golden-vector selfcheck that verifies the reloaded model on import. |
| `parameters.json` | HPO winner, feature manifest, CV metadata (and for the LSTM the frozen normalization statistics). |
| `metrics.json` | The sealed result row plus the full calibration block (theta, floor, OOF details, direction mode). |
| `interpretation.json` | The Train-derived interpretation payload: contributions, ENTRY-range segments or occlusion/trajectories, labels and disclaimer. |
| `manifest.json` | Per-file SHA-256 for the other four files, plus `folder_sha256` and `model_hash`. |

A global `artifacts/manifest.json` (schema_version `artifact.v1`) records the expected counts
per model and the folder hash of every asset, so any byte drift is detectable offline.
`folder_sha256` is built from the four hashed files of that folder (its own `manifest.json`
is excluded): sort the filenames, join `"<filename>:<sha256>"` pairs with a single `\n`
(no trailing newline), UTF-8 encode, SHA-256. `make verify` recomputes all of it — every
per-file hash, every folder hash and the manifest's own count arithmetic.

Manifest sizes of the NVDA example shipped under `examples/`: the XGB manifest is the frozen
1h core alone — 17 features, because the per-asset search selected nothing for this ticker,
one of 218 XGB assets where that is the outcome — while the LSTM input has 17 channels
(13 core daily indicators + 4 selected optional) at SEQ_LEN = 60. Both counts are per asset;
the feature search selects a different subset for every ticker, and selecting none is a
result, not a gap.

## 6. results.db schema

Nine tables and two views. Key columns only:

| table | key columns |
|---|---|
| `research_run` | run identity, git_sha, per-model recipe hashes, Train/OOS windows, declared asset counts, `research_status`, `presentation_freeze` |
| `asset_results` | (ticker, model) PK; `result_mode`, return_pct, profit_factor, model_trades, hodl_return_pct, beats_hodl, max_drawdown_pct, win_rate_pct, theta_entry, trade_floor_met, oof_trades, oof_log_growth, theta_boundary, fold_spread_relaxed, selected_feature_count, recipe_hash, `artifact_path`, oos_window |
| `asset_features` | (ticker, model, feature_id) PK; feature_name, feature_family, formula, kind — the selected per-asset features |
| `feature_search_summary` | (ticker, model) PK; verdict, stop_reason, baseline_score, baseline_trades, final_gain, n_families_surviving, selected_ids, recipe_hash |
| `integrity_checks` | check_name, scope, status, severity, message |
| `feature_train_stats` | (ticker, model, feature_key); mu_train, sigma_train, `sigma_basis`, n_finite, model_hash, interpretation_recipe_hash |
| `feature_contributions` | (ticker, model, feature_key); contribution_kind, contribution_share, feature_total_gain, split_count, family_share, low_evidence |
| `xgb_entry_ranges` | (ticker, direction, feature_key, segment_no); interval_lo/hi, interval_lo/hi_sigma, n_rows, n_entry_events, entry_lift, tp_before_sl_rate, candidate_entry_region, low_evidence |
| `oos_read_summary` | (pipe, epoch) PK; tickers, reads_this_epoch, cum_read_min/max/mean, recipe_hash, `reason` (carries the corporate-action `events_sha256`), opened_at, closed — the OOS read ledger, summarised per pipeline |

Views: `v_model_summary` (per-model result_mode counts, beats-HODL, positive strategies) and
`v_universe_summary` (one row per ticker, both models pivoted).

`asset_results.result_mode` enum (6 values):

| value | meaning |
|---|---|
| ML_MULTI_TRADE | The model produced >= 2 OOS trades; metrics are the model's own trades. |
| ML_ONE_TRADE_LOW_EVIDENCE | Exactly one model trade — reported, flagged as low evidence. |
| HODL_FALLBACK_NO_MODEL_TRADES | Zero model trades in OOS; the row reports the buy-and-hold fallback. |
| TRAIN_OOF_FLOOR_NOT_MET | The Train-OOF trade floor was never met — the model is deliberately idle. |
| NO_VALID_FOLDS | Assigned by the collector when a run had no valid CV folds. |
| TRAINING_FAILED | Assigned by the collector when a run produced no result row. |

## 7. Single data-access layer (app/data.py)

`app/data.py` is the ONLY module that opens `data/results.db` (SQLite, `mode=ro`), holds every
SQL query, caches small aggregates, and lazy-loads the per-asset artifact JSONs strictly via
`asset_results.artifact_path` — only after an asset is selected, never by scanning folders. No
page opens SQLite or touches the filesystem on its own.

Fail-closed statuses, bannered by the app:

| status | trigger |
|---|---|
| OK | Schema, integrity checks, freeze marker and asset counts all verified. |
| NOT FOUND | `data/results.db` missing or unreadable. |
| SCHEMA MISMATCH | An expected table is absent. |
| DATA INTEGRITY: FAILED | Any `integrity_checks` row is not PASS. |
| DATASET STATUS: PARTIAL | Freeze/status marker wrong, or per-model row counts differ from the counts `research_run` declares. |

## 8. Repository map

```text
README.md, Makefile, requirements.txt, LICENSE
app.py                      Streamlit entry point
app/data.py                 the single data-access layer (section 7)
app/pages/                   4 pages, flat sidebar, in reading order:
                            Data Flow 3D Visualization, Overview,
                            Basket Simulator, Data Pipeline Lego Plan
app/basket.py               basket arithmetic (no Streamlit): the three-number split
app/venn.py                 the pixel agreement diagram (no imports at all)
src/xgb/                    pipeline.py (L4-L9), feature_search.py, artifact.py, train_cv_eval.py
src/lstm/                   pipeline.py (D1-D6), model.py (D7-D8), features.py,
                            feature_search.py, universal.py, artifact.py
                            (the D9 read is driven by the research runner, which
                             is not part of this repository)
src/shared/                 op_select.py, golden_calibration.py, interpretation.py
config/                     xgb.json, lstm.json, feature_namespaces_xgb.json,
                            feature_families_{xgb,lstm}.json, xgboost_optuna_search_space.json,
                            feature_registries/
artifacts/{xgb,lstm}/<TICKER>/   5 sealed files per asset (section 5)
artifacts/manifest.json     global counts + per-asset folder hashes
data/results.db             the sealed results database (section 6)
examples/                   Example_XGB.ipynb, Example_LSTM.ipynb — one asset (NVDA)
                            end-to-end, once per model
scripts/                    verify_artifacts.py, verify_notebooks.py (`make verify`)
docs/                       METHODOLOGY.md, ARCHITECTURE.md
```

## 9. Provenance contract

Version/epoch names do not appear in public paths, in the console, in README or in the
presentation database; they may persist as immutable provenance inside the hash-sealed
artifact JSONs. The written audits under `docs/archive/facts-infos/` are the deliberate exception:
they are dated records of the research tree and name its epoch, which is why they also
state how the identity fields were anonymized here (`Raport_Spojnosci_Badan.md` §2).
