"""Model Comparison — XGB vs LSTM vs HODL in exactly four charts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

import components as C
import data
import theme

C.page_header("Model Comparison", "Return, profit factor, trade counts and beats-HODL share.")
C.guard()

df = data.results_df()
xgb = df[df["model"] == "xgb"]
lstm = df[df["model"] == "lstm"]
hodl = xgb.drop_duplicates("ticker")["hodl_return_pct"]

row1 = st.columns(2)
row2 = st.columns(2)

with row1[0]:
    st.subheader("Return distribution")
    fig = go.Figure()
    for series, name, color in ((xgb["return_pct"], "XGB", theme.MODEL_COLORS["xgb"]),
                                (lstm["return_pct"], "LSTM", theme.MODEL_COLORS["lstm"]),
                                (hodl, "HODL", theme.MODEL_COLORS["hodl"])):
        fig.add_trace(go.Histogram(x=series.clip(-100, 300), name=name, opacity=0.6,
                                   marker_color=color, nbinsx=60))
    fig.update_layout(**theme.plotly_layout(
        height=300, barmode="overlay",
        xaxis=dict(gridcolor=theme.BORDER, title="OOS return % (clipped to [-100, 300])"),
        yaxis=dict(gridcolor=theme.BORDER, title="assets")))
    st.plotly_chart(fig, width="stretch")
    st.caption("HODL uses the XGB OOS window, one value per ticker; tails beyond the "
               "clip range are stacked at the edges.")

with row1[1]:
    st.subheader("Profit factor distribution")
    fig = go.Figure()
    counts = {}
    for frame, name, color in ((xgb, "XGB", theme.MODEL_COLORS["xgb"]),
                               (lstm, "LSTM", theme.MODEL_COLORS["lstm"])):
        pf = frame[(frame["model_trades"] >= 2) & frame["profit_factor"].notna()]["profit_factor"]
        counts[name] = (len(pf), len(frame))
        fig.add_trace(go.Histogram(x=pf.clip(0, 5), name=name, opacity=0.6,
                                   marker_color=color, nbinsx=40))
    fig.update_layout(**theme.plotly_layout(
        height=300, barmode="overlay",
        xaxis=dict(gridcolor=theme.BORDER, title="profit factor (clipped to [0, 5])"),
        yaxis=dict(gridcolor=theme.BORDER, title="assets")))
    st.plotly_chart(fig, width="stretch")
    st.caption("Coverage — only assets with ≥ 2 model trades and a defined PF: " +
               " · ".join(f"{k} {n}/{total}" for k, (n, total) in counts.items()))

with row2[0]:
    st.subheader("Model trades per asset")
    bins = [(0, 0, "0"), (1, 1, "1"), (2, 10, "2–10"), (11, 50, "11–50"),
            (51, 200, "51–200"), (201, 10**9, "200+")]
    fig = go.Figure()
    for frame, name, color in ((xgb, "XGB", theme.MODEL_COLORS["xgb"]),
                               (lstm, "LSTM", theme.MODEL_COLORS["lstm"])):
        values = [int(((frame["model_trades"] >= lo) & (frame["model_trades"] <= hi)).sum())
                  for lo, hi, _ in bins]
        fig.add_trace(go.Bar(x=[label for _, _, label in bins], y=values, name=name,
                             marker_color=color, text=values, textposition="outside"))
    fig.update_layout(**theme.plotly_layout(
        height=300, barmode="group",
        xaxis=dict(gridcolor=theme.BORDER, title="model trades (OOS)"),
        yaxis=dict(gridcolor=theme.BORDER, title="assets")))
    st.plotly_chart(fig, width="stretch")

with row2[1]:
    st.subheader("Assets beating HODL")
    fig = go.Figure()
    for frame, name, color in ((xgb, "XGB", theme.MODEL_COLORS["xgb"]),
                               (lstm, "LSTM", theme.MODEL_COLORS["lstm"])):
        n, total = int(frame["beats_hodl"].sum()), len(frame)
        fig.add_trace(go.Bar(x=[name], y=[100.0 * n / total], marker_color=color,
                             text=[f"{n}/{total} ({100.0 * n / total:.1f}%)"],
                             textposition="outside", showlegend=False))
    fig.update_layout(**theme.plotly_layout(
        height=300,
        xaxis=dict(gridcolor=theme.BORDER),
        yaxis=dict(gridcolor=theme.BORDER, title="% of assets", range=[0, 50])))
    st.plotly_chart(fig, width="stretch")
    st.caption("beats_hodl is evaluated inside each model's own OOS window.")

C.integrity_footer()
