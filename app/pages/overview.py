"""Overview — the universe verdict, one asset's feature logic, and the four comparison charts."""
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

st.subheader("Model Comparison")
st.caption("Return, profit factor, trade counts and beats-HODL share.")

df = data.results_df()
xgb = df[df["model"] == "xgb"]
lstm = df[df["model"] == "lstm"]
hodl = xgb.drop_duplicates("ticker")["hodl_return_pct"]

row1 = st.columns(2)
row2 = st.columns(2)

with row1[0]:
    st.markdown("**Return distribution**")
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
    st.markdown("**Profit factor distribution**")
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
    st.markdown("**Model trades per asset**")
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
    st.markdown("**Assets beating HODL**")
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
