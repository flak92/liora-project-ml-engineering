"""Entry point: `streamlit run app.py` (or `make app`). Six read-only pages."""
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
    st.Page("app/pages/universe.py", title="Universe", url_path="universe"),
    st.Page("app/pages/asset.py", title="Asset Indicator", url_path="asset"),
    st.Page("app/pages/features.py", title="Feature Logic", url_path="features"),
    st.Page("app/pages/comparison.py", title="Model Comparison", url_path="comparison"),
    st.Page("app/pages/architecture.py", title="Architecture", url_path="architecture"),
])
pages.run()
