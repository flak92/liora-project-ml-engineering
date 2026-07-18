"""Feature Logic — what each sealed indicator actually reads (Train-derived)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

import components as C
import data
import theme

C.page_header("Feature Logic",
              "Train-derived interpretation of the sealed models — not an OOS result.")
C.guard()

ticker = C.ticker_picker()
tab_xgb, tab_lstm = st.tabs(["XGB — entry ranges", "LSTM — sequence & occlusion"])


def fmt_bound(value, sigma=False):
    if value is None:
        return "±∞"
    return f"{value:.3f}σ" if sigma else f"{value:.5g}"


with tab_xgb:
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

        st.markdown("**Family shares**")
        st.dataframe(
            [{"family": f["feature_family"], "family share": round(f["family_share"], 3),
              "features": f["n_features"]} for f in data.family_shares(ticker, "xgb")],
            hide_index=True, width="stretch")

        st.markdown("**ENTRY range explorer**")
        keys = data.range_feature_keys(ticker)
        if keys:
            c1, c2 = st.columns([3, 1])
            feature_key = c1.selectbox("Feature", keys)
            direction = c2.selectbox("Direction", ["long", "short"])
            segs = data.entry_ranges(ticker, direction, feature_key)
            if segs:
                finite = [s[k] for s in segs for k in ("interval_lo_sigma", "interval_hi_sigma")
                          if s[k] is not None]
                lo_clip = min(finite) - 0.5 if finite else -3.0
                hi_clip = max(finite) + 0.5 if finite else 3.0
                fig = go.Figure()
                for s in segs:
                    x0 = s["interval_lo_sigma"] if s["interval_lo_sigma"] is not None else lo_clip
                    x1 = s["interval_hi_sigma"] if s["interval_hi_sigma"] is not None else hi_clip
                    color = theme.ACCENT if s.get("candidate_entry_region") else theme.TEXT_DIM
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[s.get("entry_lift", 0)] * 2, mode="lines",
                        line=dict(color=color, width=10), showlegend=False,
                        hovertemplate=(f"{fmt_bound(s['interval_lo'])} … "
                                       f"{fmt_bound(s['interval_hi'])} · lift %{{y:.2f}} · "
                                       f"TP-before-SL {s.get('tp_before_sl_rate')}<extra></extra>")))
                fig.update_layout(**theme.plotly_layout(
                    height=260,
                    xaxis=dict(gridcolor=theme.BORDER, title="feature value (Train sigmas)"),
                    yaxis=dict(gridcolor=theme.BORDER, title="entry lift")))
                st.plotly_chart(fig, width="stretch")
                st.dataframe(
                    [{"raw interval": f"{fmt_bound(s['interval_lo'])} … {fmt_bound(s['interval_hi'])}",
                      "sigma interval": (f"{fmt_bound(s['interval_lo_sigma'], True)} … "
                                         f"{fmt_bound(s['interval_hi_sigma'], True)}"),
                      "rows": s["n_rows"], "entry events": s["n_entry_events"],
                      "entry share": round(s["entry_share"], 4) if s["entry_share"] is not None else None,
                      "entry lift": round(s["entry_lift"], 3) if s["entry_lift"] is not None else None,
                      "TP-before-SL": (round(s["tp_before_sl_rate"], 3)
                                       if s["tp_before_sl_rate"] is not None else None),
                      "lift vs Train baseline": (round(s["lift_vs_train_baseline"], 3)
                                                 if s["lift_vs_train_baseline"] is not None else None),
                      "status": ("CANDIDATE" if s.get("candidate_entry_region") else "") +
                                (" LOW EVIDENCE" if s.get("low_evidence") else "")}
                     for s in segs],
                    hide_index=True, width="stretch")
                st.caption(
                    "Sigmas standardize the Train distribution of candidate rows "
                    "(descriptive basis: train_candidate_rows). Accented segments are "
                    "candidate ENTRY regions; open ends render as ±∞.")

            with st.expander("Bin detail (lazy, from interpretation.json)"):
                doc = data.interpretation(ticker, "xgb")
                node = (((doc or {}).get("per_direction") or {}).get(direction) or {})
                feat = ((node.get("features") or {}).get(feature_key) or {})
                bins = (feat.get("ranges") or {}).get("per_bin") or {}
                if bins:
                    rows = []
                    for lo, hi, er, lift in zip(bins.get("lo", []), bins.get("hi", []),
                                                bins.get("entry_rate", []), bins.get("lift", [])):
                        rows.append({"lo": lo, "hi": hi, "entry rate": er, "lift": lift})
                    st.dataframe(rows, hide_index=True, width="stretch")
                elif doc is None:
                    st.markdown('<span class="status-warn">ARTIFACT PAYLOAD MISSING</span>',
                                unsafe_allow_html=True)
                else:
                    st.caption("No bin detail for this feature/direction.")

with tab_lstm:
    if data.asset(ticker, "lstm") is None:
        st.markdown('<span class="status-dim">NO LSTM RUN for this ticker</span>',
                    unsafe_allow_html=True)
    else:
        C.disclaimer_box(ticker, "lstm")
        doc = data.interpretation(ticker, "lstm")
        occl = ((doc or {}).get("features_global") or {})

        st.markdown("**Input-channel contributions (occlusion)**")
        contribs = data.contributions(ticker, "lstm")
        if contribs:
            names = [c["feature_name"] for c in contribs]
            entry_shares = [c["contribution_share"] for c in contribs]
            global_shares = [
                ((occl.get(c["feature_key"]) or {}).get("occlusion") or {}).get("share_global")
                for c in contribs]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=entry_shares, y=names, orientation="h",
                                 name="ENTRY-conditioned (p ≥ θ)", marker_color=theme.ACCENT))
            if any(g is not None for g in global_shares):
                fig.add_trace(go.Bar(x=[g or 0 for g in global_shares], y=names,
                                     orientation="h", name="global",
                                     marker_color=theme.TEXT_DIM))
            fig.update_layout(**theme.plotly_layout(
                height=max(220, 30 * len(names)), barmode="group",
                xaxis=dict(gridcolor=theme.BORDER, title="occlusion share"),
                yaxis=dict(gridcolor=theme.BORDER, autorange="reversed")))
            st.plotly_chart(fig, width="stretch")
            low_ev = [c["feature_name"] for c in contribs if c.get("low_evidence")]
            if low_ev:
                st.markdown('<span class="status-warn">LOW EVIDENCE</span>: ' +
                            ", ".join(low_ev), unsafe_allow_html=True)
            st.caption("Primary bars: occlusion measured only on windows the model wants "
                       "to enter (p ≥ θ). Dim bars: occlusion over all windows.")

        st.markdown("**Train normalization (mean / std per channel)**")
        st.dataframe(
            [{"channel": t["feature_name"], "Train mean": round(t["mu_train"], 6),
              "Train std": round(t["sigma_train"], 6)} for t in data.train_stats(ticker, "lstm")],
            hide_index=True, width="stretch")
        st.caption(
            "sigma_basis = artifact_norm_stats: these are the ACTUAL input normalization "
            "constants of the sealed network — unlike the XGB sigmas, which only "
            "describe the Train distribution.")

        st.markdown("**State-sequence significance (60-step trajectories)**")
        per_dir = (doc or {}).get("per_direction") or {}
        if per_dir:
            c1, c2 = st.columns([3, 1])
            directions = sorted(per_dir.keys())
            direction = c2.selectbox("Direction", directions, key="lstm_dir")
            node = per_dir.get(direction) or {}
            trajectories = node.get("trajectories") or {}
            if trajectories:
                traj_key = c1.selectbox("Channel", sorted(trajectories.keys()))
                tr = trajectories.get(traj_key) or {}
                steps = list(range(-len(tr.get("base_mean", [])) + 1, 1))
                fig = go.Figure()
                for mean_key, std_key, color, name in (
                        ("base_mean", "base_std", theme.TEXT_DIM, "all windows"),
                        ("entry_mean", "entry_std", theme.ACCENT, "ENTRY windows")):
                    mean = tr.get(mean_key) or []
                    std = tr.get(std_key) or []
                    if not mean:
                        continue
                    if std and len(std) == len(mean):
                        upper = [m + s for m, s in zip(mean, std)]
                        lower = [m - s for m, s in zip(mean, std)]
                        fig.add_trace(go.Scatter(x=steps + steps[::-1], y=upper + lower[::-1],
                                                 fill="toself", fillcolor=color, opacity=0.12,
                                                 line=dict(width=0), showlegend=False,
                                                 hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=steps, y=mean, mode="lines", name=name,
                                             line=dict(color=color, width=2)))
                fig.update_layout(**theme.plotly_layout(
                    height=300,
                    xaxis=dict(gridcolor=theme.BORDER, title="sessions before decision (0 = now)"),
                    yaxis=dict(gridcolor=theme.BORDER, title="normalized channel value")))
                st.plotly_chart(fig, width="stretch")
                n_entry = tr.get("n_entry")
                note = f"windows: {tr.get('n_base', '—')} all / {n_entry} ENTRY"
                if node.get("low_evidence"):
                    note += ' — <span class="status-warn">LOW EVIDENCE (few ENTRY windows)</span>'
                st.markdown(note, unsafe_allow_html=True)
                tp = node.get("tp_before_sl_rate_entry")
                if tp is not None:
                    st.caption(f"TP-before-SL rate on ENTRY windows ({direction}): {tp:.3f}")
        elif doc is None:
            st.markdown('<span class="status-warn">ARTIFACT PAYLOAD MISSING</span>',
                        unsafe_allow_html=True)

C.integrity_footer()
