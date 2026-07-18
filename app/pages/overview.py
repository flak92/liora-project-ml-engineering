"""Overview — what this project is and what came out of it."""
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

st.markdown(
    "For each S&P 500 asset the study calibrates a dedicated ENTRY indicator on the "
    "Train window only: XGBoost reads value ranges of engineered 1h features, the LSTM "
    "reads a 60-session sequence of daily state channels. Take-profit and stop-loss are "
    "a mechanical ATR triple-barrier contract — never a model decision. Buy-and-hold "
    "(HODL) over the same out-of-sample window is the benchmark, and a model that finds "
    "no robust operating point stays idle by design. All results below are one-shot "
    "out-of-sample reads of sealed artifacts; nothing is trained or recomputed here.")

run = data.research_run()
checks = data.integrity()
n_pass = sum(1 for c in checks if c.get("status") == "PASS")
C.metric_row([
    ("Assets", f"{len(data.tickers())}"),
    ("XGB runs", f"{run.get('xgb_assets', '—')}"),
    ("LSTM runs", f"{run.get('lstm_assets', '—')}"),
    ("Integrity checks", f"{n_pass}/{len(checks)} PASS"),
])

st.subheader("Train / OOS split")
fig = go.Figure()
rows = [("XGB", run.get("xgb_train", ""), run.get("xgb_oos", "")),
        ("LSTM", run.get("lstm_train", ""), run.get("lstm_oos", ""))]
for label, train_w, oos_w in rows:
    for window, color, name in ((train_w, theme.TEXT_DIM, "Train"),
                                (oos_w, theme.ACCENT, "OOS")):
        if " -> " not in (window or ""):
            continue
        start, end = window.split(" -> ")
        fig.add_trace(go.Scatter(
            x=[start, end], y=[label, label], mode="lines",
            line=dict(color=color, width=16), name=name,
            showlegend=(label == "XGB"),
            hovertemplate=f"{label} {name}: {start} → {end}<extra></extra>"))
fig.update_layout(**theme.plotly_layout(height=170, xaxis=dict(gridcolor=theme.BORDER, type="date")))
st.plotly_chart(fig, width="stretch")

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

with st.expander("When the models stay idle (result modes)"):
    shares = data.result_mode_shares()
    table = [{"model": r["model"].upper(),
              "result mode": r["result_mode"],
              "status": C.status_label(r["result_mode"]),
              "assets": r["n"]} for r in shares]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(
        "A NO-TRADE (floor) or IDLE / HODL asset is a deliberate outcome: the Train-only "
        "calibration found no operating point clearing the trade floor, so the indicator "
        "abstains instead of forcing trades.")

st.subheader("Main finding")
st.markdown(
    "Per-asset, Train-calibrated entry indicators are honest but hard to convert into "
    "out-of-sample outperformance: in this bull OOS window the median strategy return "
    "trails buy-and-hold for both model families, and only a minority of assets beat "
    "HODL (XGB {x_b}, LSTM {l_b}). The value of the study is the sealed, auditable "
    "per-asset methodology — calibrated thresholds, explicit idleness and a full "
    "interpretation layer — not a live trading signal.".format(
        x_b=f"{stats['xgb']['beats_hodl_pct']:.0f}%" if "xgb" in stats else "—",
        l_b=f"{stats['lstm']['beats_hodl_pct']:.0f}%" if "lstm" in stats else "—"))

C.integrity_footer()
