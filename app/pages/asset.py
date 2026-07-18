"""Asset Indicator — the dedicated per-asset indicator, one ticker at a time."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import data

C.page_header("Asset Indicator", "One dedicated, sealed entry indicator per asset.")
C.guard()

ticker = C.ticker_picker()
rows = {r["model"]: r for r in data.asset_models(ticker)}


def render_model(col, model):
    with col:
        st.subheader(model.upper())
        row = rows.get(model)
        if row is None:
            st.markdown('<span class="status-dim">NO LSTM RUN for this ticker</span>',
                        unsafe_allow_html=True)
            return
        st.markdown(C.status_html(row["result_mode"]), unsafe_allow_html=True)
        C.metric_row([
            ("OOS return", C.pct(row["return_pct"])),
            ("HODL (same window)", C.pct(row["hodl_return_pct"])),
            ("Model trades", f"{row['model_trades']}"),
            ("Win rate", C.pct(row["win_rate_pct"]) if row["win_rate_pct"] is not None else "—"),
        ])
        st.markdown(f"OOS window: {C.mono(row.get('oos_window') or '—')}",
                    unsafe_allow_html=True)

        st.markdown("**Calibration (Train-only)**")
        theta = row.get("theta_entry")
        boundary = " — <span class='status-warn'>θ AT SPECTRUM BOUNDARY</span>" \
            if row.get("theta_boundary") == 1 else ""
        st.markdown(f"ENTRY threshold θ = {C.mono(theta)}{boundary}",
                    unsafe_allow_html=True)
        cal = data.calibration(ticker, model) or {}
        if model == "lstm":
            direction = cal.get("direction_mode") or "—"
            st.markdown(f"Direction mode: {C.mono(direction)}", unsafe_allow_html=True)
        else:
            st.markdown("Direction mode: long + short (both projections in the "
                        "interpretation layer)")
        details = []
        if cal.get("trade_floor_met") is not None:
            details.append(f"trade floor met: {cal['trade_floor_met']}")
        if cal.get("oof_trades") is not None:
            details.append(f"OOF trades: {cal['oof_trades']}")
        if cal.get("oof_log_growth") is not None:
            details.append(f"OOF log-growth: {cal['oof_log_growth']:.4f}")
        if details:
            st.caption(" · ".join(details))

        st.markdown("**Selected features**")
        feats = data.features(ticker, model)
        if feats:
            st.dataframe(
                [{"feature": f["feature_name"], "family": f["feature_family"],
                  "formula": f["formula"]} for f in feats],
                hide_index=True, width="stretch")
        else:
            st.caption("No optional features selected — the frozen core manifest runs alone.")
        st.caption(
            f"Selected optional features: {row.get('selected_feature_count', 0)} "
            "(on top of the frozen core manifest — full breakdown on Feature Logic).")

        st.markdown("**Artifact**")
        st.markdown(C.mono(row.get("artifact_path") or "—"), unsafe_allow_html=True)
        if st.toggle(f"Show {model.upper()} artifact manifest", key=f"man_{model}"):
            man = data.manifest(ticker, model)
            if man is None:
                st.markdown('<span class="status-warn">ARTIFACT PAYLOAD MISSING</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f"model_hash: {C.mono(man.get('model_hash', '—'))}",
                            unsafe_allow_html=True)
                st.dataframe(
                    [{"file": name, "bytes": meta.get("bytes"),
                      "sha256": (meta.get("sha256") or "")[:12]}
                     for name, meta in sorted((man.get("files") or {}).items())],
                    hide_index=True, width="stretch")


left, right = st.columns(2)
render_model(left, "xgb")
render_model(right, "lstm")

if st.button("Open Feature Logic for this ticker"):
    st.switch_page("app/pages/features.py")

C.integrity_footer()
