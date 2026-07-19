"""Data Flow 3D Visualization — the build path, twice: one glance and one ladder.

Two drawings, no prose. The graphviz sketch is the whole study in eight boxes — bars to
features to Train-only calibration, the two models side by side, one artifact per asset,
the sealed store, this console. Below it, the standalone 2.5D map draws the same path at
full depth: sixteen levels, every contract clickable.

The map is a self-contained HTML file at the repository root — zero dependencies, zero
network requests, and it opens in any browser on its own. This page only embeds it, and
everything the reader needs to know is written inside the drawings themselves: the scenes
that illustrate rather than measure carry their own SCHEMATIC mark on the canvas.

The graphviz source is a raw DOT string, so Streamlit ships it to the browser and renders
it there with its bundled viz.js — no `graphviz` package and no system `dot` binary, which
is what keeps this page working on a fresh clone from requirements.txt.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import theme

ROOT = Path(__file__).resolve().parents[2]
FLOW = ROOT / "data_flow_3d_visualization.html"


@st.cache_data
def _load(mtime: float) -> str:
    """The self-contained map, cached per file mtime."""
    return FLOW.read_text(encoding="utf-8")


C.page_header("Data Flow 3D Visualization", "")
C.guard(stop=False)  # static page: banner without stopping

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

if not FLOW.exists():
    st.error(f"Flow map not found: {FLOW.name} (expected at the repository root).")
    st.stop()

# The map fits itself to the height it is given, and that fit is bound by the vertical: the
# ladder's world is 1:11.4, so width has ~20x the slack. At 820 the sixteen levels landed 43px
# apart and the labels — which the map floors at 80% however far it zooms out — overlapped into
# the mess this height exists to fix. 4750 puts the fit just past that font floor, so text and
# artwork are drawn in the proportion the map was designed for, ~306px per level.
st.iframe(_load(FLOW.stat().st_mtime), height=4750)
st.caption("Standalone file: data_flow_3d_visualization.html (repository root — opens in any browser).")
