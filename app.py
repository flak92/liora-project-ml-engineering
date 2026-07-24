"""Entry point: `streamlit run app.py` (or `make on`). Two read-only pages, flat.

A list rather than a dict: st.navigation only draws section headers when it is given groups, and two
pages do not need headers over them. Order is the reading order — the data journey first (how the
methodology prepares data and walks it through the ladder on real assets), then Smart Methodology (the
configurables map and the run replay). Both read the frozen snapshot; nothing trains at runtime. The
sealed-model product console (per-asset outcomes, the basket simulator) lives on the `main` branch —
this branch is the method, not the sealed models.
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
    st.Page("app/pages/data_journey.py", title="Data Journey", url_path="data-journey", default=True),
    st.Page("app/pages/methodology.py", title="Smart Methodology", url_path="methodology"),
], expanded=True)
pages.run()
