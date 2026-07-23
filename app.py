"""Entry point: `streamlit run app.py` (or `make on`). Three read-only pages, flat.

A list rather than a dict: st.navigation only draws section headers when it is given groups,
and three pages do not need headers over them. Order is the reading order — the result first,
then the thing to play with, then the method in full. Overview stays the landing page; it is
the one that answers "what came out of this". The build-path maps and the full configurables
catalog now live under docs/archive/ (deep-dive), reachable from docs/SMART_METHODOLOGY.md.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

st.set_page_config(
    page_title="S&P 500 ML Indicator Study",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation([
    st.Page("app/pages/overview.py", title="Overview", url_path="overview", default=True),
    st.Page("app/pages/simulator.py", title="Basket Simulator", url_path="simulator"),
    st.Page("app/pages/methodology.py", title="Smart Methodology", url_path="methodology"),
], expanded=True)
pages.run()
