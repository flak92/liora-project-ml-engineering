"""Architecture — from the presentation back to the code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import data
import theme

C.page_header("Architecture", "How the sealed study flows into this console — and where "
                              "to read the code.")
C.guard(stop=False)  # static page: banner without stopping

# counts are DERIVED from the store, never typed into prose — a new epoch must not be able
# to leave a stale number behind in the copy.
_run = data.research_run()
_n_xgb, _n_lstm = _run.get("xgb_assets", 0), _run.get("lstm_assets", 0)
_n_assets = _n_xgb + _n_lstm

st.subheader("Data flow")
DOT = f"""
digraph flow {{
  rankdir=LR;
  bgcolor="{theme.BG}";
  node [shape=box, style=filled, fillcolor="{theme.SURFACE}", color="{theme.BORDER}",
        fontcolor="{theme.TEXT}", fontname="monospace", fontsize=11];
  edge [color="{theme.TEXT_DIM}"];
  ohlcv [label="OHLCV\\n1h / 1d bars"];
  feat  [label="features /\\nsequences"];
  cal   [label="Train-only\\ncalibration"];
  xgb   [label="XGB", fillcolor="{theme.ACCENT}", fontcolor="{theme.BG}"];
  lstm  [label="LSTM", fillcolor="{theme.ACCENT}", fontcolor="{theme.BG}"];
  art   [label="per-asset\\nartifact"];
  db    [label="results.db\\n(read-only)"];
  app   [label="Streamlit\\nconsole"];
  ohlcv -> feat -> cal;
  cal -> xgb; cal -> lstm;
  xgb -> art; lstm -> art;
  art -> db -> app;
}}
"""
st.graphviz_chart(DOT, width="stretch")

st.subheader("Repository map")
st.dataframe(
    [
        {"path": "app.py + app/", "role": "this console — nine read-only pages; app/data.py is the ONLY module that opens the database"},
        {"path": "src/xgb/", "role": "XGB research code: pipeline.py (layers L4–L9), feature_search.py, train_cv_eval.py, artifact.py"},
        {"path": "src/lstm/", "role": "LSTM research code: pipeline.py (D1–D6), model.py (D7–D8), features.py, universal.py, feature_search.py, artifact.py"},
        {"path": "src/shared/", "role": "contracts shared by both pipelines: op_select.py (operating point), golden_calibration.py (search policy), interpretation.py (range math)"},
        {"path": "config/", "role": "the frozen configuration the code reads: xgb.json, lstm.json, feature families, feature registries"},
        {"path": "artifacts/xgb/<T>/, artifacts/lstm/<T>/", "role": f"{_n_assets} sealed per-asset artifacts (strategy, manifest, parameters, metrics, interpretation) + artifacts/manifest.json completeness contract"},
        {"path": "data/results.db", "role": "SQLite results store (9 tables + 2 views), opened read-only"},
        {"path": "examples/", "role": "two executed notebooks: the full XGB path for AAPL and NVDA"},
        {"path": "docs/", "role": "METHODOLOGY.md and ARCHITECTURE.md"},
    ],
    hide_index=True, width="stretch")

st.subheader("Contracts in one paragraph each")
st.markdown(
    "**Label contract.** Every trade label is an ATR triple-barrier: take-profit and "
    "stop-loss are mechanical barriers derived from Train-window volatility — the model "
    "only decides ENTRY, never the exit. \n\n"
    "**Train-only calibration.** The entry threshold θ, the trade floor and the "
    "operating point are selected on accumulated out-of-fold Train predictions "
    "(src/shared/op_select.py) — one shared point, never a per-fold optimum. The OOS "
    "window feeds no Train-side choice; every read of it is counted in an append-only "
    "ledger, summarised per pipeline on the Integrity page (reads this epoch, and the "
    "min / mean / max of the cumulative per-asset counter). \n\n"
    "**Sealed artifacts.** Each asset ships five files whose SHA-256 hashes chain into "
    "manifest.json (per-file → folder → model hash); artifacts/manifest.json states the "
    f"completeness contract ({_n_xgb} XGB + {_n_lstm} LSTM = {_n_assets}). The console verifies dataset "
    "health fail-closed and renders everything read-only.")

run = data.research_run()
st.caption(f"Windows — XGB Train {run.get('xgb_train', '—')}, OOS {run.get('xgb_oos', '—')} · "
           f"LSTM Train {run.get('lstm_train', '—')}, OOS {run.get('lstm_oos', '—')}")
C.integrity_footer()
