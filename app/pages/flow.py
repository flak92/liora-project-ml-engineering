"""Data Flow — the per-asset build path as a standalone 2.5D canvas map.

The map is a self-contained HTML file at the repository root: zero dependencies, zero network
requests, and it opens in any browser on its own. This page only embeds it.

The map's own figures are frozen (the research snapshot is frozen too). The caption below is
DERIVED from the store on every render, so if the two ever disagree, the disagreement is on
screen rather than hidden — the same house rule the Architecture page follows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import data

ROOT = Path(__file__).resolve().parents[2]
FLOW = ROOT / "data_flow_3d.html"


@st.cache_data
def _load(mtime: float) -> str:
    """The self-contained map, cached per file mtime."""
    return FLOW.read_text(encoding="utf-8")


C.page_header("Data Flow", "The per-asset build path — two independent sealed pipelines, drawn "
                           "as one ladder.")
C.guard(stop=False)  # static page: banner without stopping

_run = data.research_run()
_n_xgb, _n_lstm = _run.get("xgb_assets", 0), _run.get("lstm_assets", 0)

st.write(
    "The ladder is **one pass** of the procedure: split-adjusted bars → time split with purge and "
    "embargo → ATR Triple-Barrier labels → profit-aligned Train-only tuning (per-asset Optuna for "
    "XGB; one warm-started committed backbone for LSTM) → sealed artifact → the ledgered OOS "
    "read → Train-derived interpretation. The same pass runs independently for every asset — "
    "nothing is pooled, and each ticker gets its own model. The models decide **ENTRY only**: "
    "take-profit and stop-loss are a mechanical ATR contract, never a model decision."
)
st.caption(
    f"Sealed indicators in this release: {_n_xgb} XGB (1h) · {_n_lstm} LSTM (daily) — "
    f"{_n_xgb + _n_lstm} artifacts, presentation freeze "
    f"`{_run.get('presentation_freeze') or '—'}`. Counts and freeze read from the store; the "
    f"map's own figures are frozen. "
    "Some scenes are marked SCHEMATIC (the tuning-trial scatter and the sample OHLCV rows carry "
    "the mark; the detector wall and feature-matrix cells are likewise illustrative): they "
    "illustrate the shape of a process and are not measurements."
)
st.caption(
    "Drag to pan · **Ctrl + wheel** (or trackpad pinch) to zoom · a plain wheel scrolls this "
    "page · click a node for its contract · Esc closes. The map is deliberately tall: sixteen "
    "ladder levels need vertical room before their labels stop colliding, so scroll down "
    "through it rather than expecting it on one screen."
)

if not FLOW.exists():
    st.error(f"Flow map not found: {FLOW.name} (expected at the repository root).")
    st.stop()

# The map fits itself to the height it is given, and that fit is bound by the vertical: the
# ladder's world is 1:11.4, so width has ~20x the slack. At 820 the sixteen levels landed 43px
# apart and the labels — which the map floors at 80% however far it zooms out — overlapped into
# the mess this height exists to fix. 4750 puts the fit just past that font floor, so text and
# artwork are drawn in the proportion the map was designed for, ~306px per level.
st.iframe(_load(FLOW.stat().st_mtime), height=4750)
st.caption("Standalone file: data_flow_3d.html (repository root — opens in any browser, with "
           "its own title, the whole ladder on one screen, and plain-wheel zoom).")
