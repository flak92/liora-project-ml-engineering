"""Entry point: `streamlit run app.py` (or `make on`). Six read-only pages, flat.

A list rather than a dict: st.navigation only draws section headers when it is given
groups, and six pages do not need three headers over them. Order is the reading order —
the build path, then the result, then the thing to play with, then the procedure in full,
then the procedure re-lived as a timed replay. Overview stays the landing page; it is the
one that answers "what came out of this".
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
    st.Page("app/pages/flow.py", title="Data Flow 3D Visualization", url_path="flow"),
    st.Page("app/pages/overview.py", title="Overview", url_path="overview", default=True),
    st.Page("app/pages/simulator.py", title="Basket Simulator", url_path="simulator"),
    st.Page("app/pages/blueprint.py", title="Data Pipeline Lego Plan", url_path="blueprint"),
    st.Page("app/pages/calibration.py", title="Calibration Configurables", url_path="calibration"),
    st.Page("app/pages/methodology_replay.py", title="Methodology Replay", url_path="replay"),
], expanded=True)
pages.run()
