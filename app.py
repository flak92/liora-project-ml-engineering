"""Entry point: `streamlit run app.py` (or `make on`). Six read-only pages in three
sections: Playground (build a basket), Results (what the models did),
Method & proof (how it was built and how to check it)."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))

st.set_page_config(
    page_title="S&P 500 ML Indicator Study",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = st.navigation({
    "Playground": [
        st.Page("app/pages/simulator.py", title="Basket Simulator", url_path="simulator"),
    ],
    "Results": [
        st.Page("app/pages/overview.py", title="Overview", url_path="overview", default=True),
        st.Page("app/pages/features.py", title="Feature Logic", url_path="features"),
        st.Page("app/pages/comparison.py", title="Model Comparison", url_path="comparison"),
    ],
    "Method & proof": [
        st.Page("app/pages/blueprint.py", title="Data Pipeline Lego Plan", url_path="blueprint"),
        st.Page("app/pages/flow.py", title="Data Flow 3D Visualization", url_path="flow"),
    ],
}, expanded=True)
pages.run()
