"""Calibration Configurables — every tunable number of the ladder as a range.

For each configurable: its range, whether it is FROZEN (proof standard) or ADMISSIBLE (a
hypothesis you may widen), where in the ladder it is calibrated, what it depends on, and which
knob to widen when a rung comes up empty. Grouped by the three dependency series — HPO/XGB,
strategy/theta, features/OHLCV — plus the frozen proof-standard gates.

A self-contained HTML file at the repository root (dark, matched to this console); this page
embeds it. The full ordered method is docs/FEATURE_DISCOVERY_METHODOLOGY.md; the machine-readable
catalog is docs/CALIBRATION_CONFIGURABLES.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "calibration_configurables.html"


@st.cache_data
def _load(mtime: float) -> str:
    """The self-contained catalog, cached per file mtime."""
    return PAGE.read_text(encoding="utf-8")


C.page_header("Calibration Configurables", "")
C.guard(stop=False)  # static reference page: banner without stopping

if not PAGE.exists():
    st.error(f"Not found: {PAGE.name} (expected at the repository root).")
    st.stop()

# A flowing document, not a fixed-aspect board: sized to fit its content (the methodology map + the
# catalog) so the whole page reads on one scroll rather than inside a short inner scrollbar.
st.iframe(_load(PAGE.stat().st_mtime), height=5400)
st.caption("Standalone file: calibration_configurables.html (repository root — opens in any browser). "
           "Machine-readable catalog: docs/CALIBRATION_CONFIGURABLES.md.")
