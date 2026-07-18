"""Shared UI components. Pages use these + data.py and nothing else."""
import html

import streamlit as st

import data
import theme

# result_mode -> short operational label (full mode is always shown alongside in mono)
STATUS_LABELS = {
    "ML_MULTI_TRADE": "ACTIVE",
    "ML_ONE_TRADE_LOW_EVIDENCE": "LOW EVIDENCE",
    "HODL_FALLBACK_NO_MODEL_TRADES": "IDLE / HODL",
    "TRAIN_OOF_FLOOR_NOT_MET": "NO-TRADE (floor)",
    "NO_VALID_FOLDS": "NO VALID FOLDS",
    "TRAINING_FAILED": "FAILED",
}
STATUS_CSS = {
    "ML_MULTI_TRADE": "status-pass",
    "ML_ONE_TRADE_LOW_EVIDENCE": "status-warn",
    "HODL_FALLBACK_NO_MODEL_TRADES": "status-dim",
    "TRAIN_OOF_FLOOR_NOT_MET": "status-dim",
    "NO_VALID_FOLDS": "status-warn",
    "TRAINING_FAILED": "status-fail",
}


def page_header(title, purpose):
    st.markdown(theme.CSS, unsafe_allow_html=True)
    st.title(title)
    st.caption(purpose)


def guard(stop=True):
    """Fail-closed gate every page starts with. Status is written as text."""
    h = data.health()
    if h["status"] != data.OK:
        st.error(f"{h['status']} — {h.get('detail', '')}")
        if stop:
            st.stop()
    return h


def status_label(mode):
    return STATUS_LABELS.get(mode, mode or "—")


def status_html(mode):
    css = STATUS_CSS.get(mode, "status-dim")
    return (f'<span class="{css}">{html.escape(status_label(mode))}</span> '
            f'<span class="mono status-dim">{html.escape(str(mode))}</span>')


def mono(text):
    return f'<span class="mono">{html.escape(str(text))}</span>'


def pct(x, digits=1):
    return "—" if x is None else f"{x:+.{digits}f}%"


def num(x):
    return "—" if x is None else f"{x:,.0f}" if isinstance(x, float) else f"{x:,}"


def metric_row(pairs):
    """pairs = [(label, value_str), ...] — aligned numbers, no delta arrows."""
    cols = st.columns(len(pairs))
    for col, (label, value) in zip(cols, pairs):
        col.markdown(
            f'<div class="metric-label">{html.escape(label)}</div>'
            f'<div class="metric-value">{html.escape(str(value))}</div>',
            unsafe_allow_html=True)


def disclaimer_box(ticker, model):
    """Mandatory interpretation banner (payload labels with a hard fallback)."""
    labels, disclaimer = data.interpretation_labels(ticker, model)
    body = f"<strong>{html.escape(labels)}</strong>"
    if disclaimer:
        body += f"<br>{html.escape(disclaimer)}"
    st.markdown(f'<div class="disclaimer-box">{body}</div>', unsafe_allow_html=True)


def integrity_footer():
    run = data.research_run()
    checks = data.integrity()
    n_pass = sum(1 for c in checks if c.get("status") == "PASS")
    st.caption(
        f"sealed dataset · built {run.get('created_at', '—')} · "
        f"integrity {n_pass}/{len(checks)} PASS · "
        f"XGB OOS {run.get('xgb_oos', '—')} · LSTM OOS {run.get('lstm_oos', '—')}")


def goto_asset(ticker):
    st.session_state["selected_ticker"] = ticker
    st.switch_page("app/pages/asset.py")


def ticker_picker():
    """Shared ticker selector; keeps the selection across pages."""
    tickers = data.tickers()
    current = st.session_state.get("selected_ticker")
    index = tickers.index(current) if current in tickers else 0
    ticker = st.selectbox("Ticker", tickers, index=index)
    st.session_state["selected_ticker"] = ticker
    return ticker
