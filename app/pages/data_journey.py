"""Data Journey — the methodology shown as the road the data travels.

A self-contained board (built by scripts/build_data_journey.py, sealed by scripts/verify_data_journey.py)
reconstructed from results/methodology_snapshot/compiled/* and the frozen contract. It opens with the
data-preparation timeline (warmup → Train/OOS split with purge/embargo, oos_reads=0) — the fundament —
then the pipeline stages (rungs 0→6 → OOS), then seven real assets travelling those stages to their
honest terminal. Nothing trains at runtime.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C

ROOT = Path(__file__).resolve().parents[2]
BOARD = ROOT / "data_journey.html"


@st.cache_data
def _load(path: str, mtime: float) -> str:
    """The self-contained board, cached per (file, mtime)."""
    return Path(path).read_text(encoding="utf-8")


C.page_header("Data Journey",
              "Raw OHLCV → warmup → Train/OOS split (purge + embargo, oos_reads = 0) → the methodology")
C.guard(stop=False)  # reference page: banner without stopping

if not BOARD.exists():
    st.error(f"Not found: {BOARD.name} — run `make data-journey`.")
else:
    # A flowing document (timeline + stages + per-asset tracks + funnels), sized to read on one scroll.
    st.iframe(_load(str(BOARD), BOARD.stat().st_mtime), height=1400)
    st.caption("Standalone data_journey.html (repository root — opens in any browser). Built by "
               "scripts/build_data_journey.py from results/methodology_snapshot/compiled/*, sealed by "
               "scripts/verify_data_journey.py. Every number is read from the frozen snapshot.")
