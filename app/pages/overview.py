"""Overview — the universe verdict, then what one asset's XGB model reads."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

import components as C
import data
import theme

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

st.subheader("Feature Logic")
st.caption("Train-derived interpretation of the sealed models — not an OOS result.")
ticker = C.ticker_picker()
if data.asset(ticker, "xgb") is None:
    st.markdown('<span class="status-dim">NO XGB RUN for this ticker</span>',
                unsafe_allow_html=True)
else:
    C.disclaimer_box(ticker, "xgb")

    st.markdown("**Feature contributions (XGB split total-gain)**")
    contribs = data.contributions(ticker, "xgb")
    if contribs:
        fig = go.Figure(go.Bar(
            x=[c["contribution_share"] for c in contribs],
            y=[c["feature_name"] for c in contribs],
            orientation="h", marker_color=theme.ACCENT,
            customdata=[[c.get("split_count"), c.get("feature_family")] for c in contribs],
            hovertemplate="%{y}: share %{x:.3f} · splits %{customdata[0]} · "
                          "family %{customdata[1]}<extra></extra>"))
        fig.update_layout(**theme.plotly_layout(
            height=max(220, 26 * len(contribs)),
            xaxis=dict(gridcolor=theme.BORDER, range=[0, 1], title="contribution share"),
            yaxis=dict(gridcolor=theme.BORDER, autorange="reversed")))
        st.plotly_chart(fig, width="stretch")
        low_ev = [c["feature_name"] for c in contribs if c.get("low_evidence")]
        if low_ev:
            st.markdown('<span class="status-warn">LOW EVIDENCE</span>: ' +
                        ", ".join(low_ev), unsafe_allow_html=True)
        st.caption("Contribution shares are a projection of the WHOLE model's split "
                   "gains — features act jointly, these are not standalone rules.")
