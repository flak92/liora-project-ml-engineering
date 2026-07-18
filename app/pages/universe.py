"""Universe — the single operational table, one row per ticker."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import data

C.page_header("Universe", "All assets at a glance — select a row to open its indicator.")
C.guard()

df = data.universe_df().copy()
df["XGB status"] = df["xgb_status"].map(C.status_label)
df["LSTM status"] = df["lstm_status"].map(lambda m: C.status_label(m) if m else "NO LSTM RUN")

f1, f2, f3, f4 = st.columns([2, 1, 2, 1])
query = f1.text_input("Ticker search", "").strip().upper()
model_filter = f2.selectbox("Model", ["both", "xgb", "lstm"])
status_options = sorted(set(df["XGB status"]) | set(df["LSTM status"]))
status_filter = f3.multiselect("Status", status_options)

view = df
if query:
    view = view[view["ticker"].str.contains(query, regex=False)]
if status_filter:
    mask = view["XGB status"].isin(status_filter) | view["LSTM status"].isin(status_filter)
    if model_filter == "xgb":
        mask = view["XGB status"].isin(status_filter)
    elif model_filter == "lstm":
        mask = view["LSTM status"].isin(status_filter)
    view = view[mask]

columns = {
    "ticker": st.column_config.TextColumn("Ticker"),
    "XGB status": st.column_config.TextColumn("XGB status"),
    "LSTM status": st.column_config.TextColumn("LSTM status"),
    "xgb_return_pct": st.column_config.NumberColumn("XGB return", format="%+.1f%%"),
    "lstm_return_pct": st.column_config.NumberColumn("LSTM return", format="%+.1f%%"),
    "hodl_return_pct": st.column_config.NumberColumn("HODL return", format="%+.1f%%"),
    "xgb_trades": st.column_config.NumberColumn("XGB trades", format="%d"),
    "lstm_trades": st.column_config.NumberColumn("LSTM trades", format="%d"),
}
if model_filter == "xgb":
    shown = ["ticker", "XGB status", "xgb_return_pct", "hodl_return_pct", "xgb_trades"]
elif model_filter == "lstm":
    shown = ["ticker", "LSTM status", "lstm_return_pct", "hodl_return_pct", "lstm_trades"]
else:
    shown = ["ticker", "XGB status", "LSTM status", "xgb_return_pct", "lstm_return_pct",
             "hodl_return_pct", "xgb_trades", "lstm_trades"]
view = view[shown].reset_index(drop=True)

f4.markdown(f'<div class="metric-label">Rows</div>'
            f'<div class="metric-value">{len(view)}</div>', unsafe_allow_html=True)

event = st.dataframe(
    view, hide_index=True, width="stretch", height=560,
    column_config={k: v for k, v in columns.items() if k in shown},
    on_select="rerun", selection_mode="single-row")

selected = event.selection.rows if event and event.selection else []
if selected:
    C.goto_asset(view.iloc[selected[0]]["ticker"])

st.caption(
    "Select a row to open the asset view. HODL return in this table is computed over "
    "the XGB OOS window; the asset page shows each model's own-window benchmark. "
    "Sort by clicking a column header.")
C.integrity_footer()
