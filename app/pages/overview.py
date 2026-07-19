"""Overview — the universe verdict in one table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import data

C.page_header(
    "S&P 500 ML Indicator Study",
    "One dedicated, sealed ML entry indicator per asset — XGBoost (1h bars) and "
    "LSTM (daily bars) against a buy-and-hold benchmark.")
C.guard()

st.subheader("Median outcomes")
stats = data.overview_stats()
lines = []
for model in ("xgb", "lstm"):
    s = stats.get(model)
    if not s:
        continue
    lines.append({
        "model": model.upper(),
        "assets": s["n_assets"],
        "median return": f"{s['median_return_pct']:+.1f}%",
        "median HODL (same window)": f"{s['median_hodl_return_pct']:+.1f}%",
        "beats HODL": f"{s['beats_hodl_n']} ({s['beats_hodl_pct']:.1f}%)",
        "median PF": ("—" if s["median_profit_factor"] is None
                      else f"{s['median_profit_factor']:.2f}"),
        "PF coverage": f"{s['pf_coverage_n']}/{s['n_assets']}",
    })
st.dataframe(lines, hide_index=True, width="stretch")
st.caption(
    "Profit factor is only computed on assets with at least 2 model trades and a "
    "defined PF — the coverage column states how many assets that is. HODL medians "
    "use each model's own OOS window.")
