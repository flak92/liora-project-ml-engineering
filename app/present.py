"""Tiny presentation helpers shared by the native methodology pages — facts, funnels, status chips.

The house style (Rung 8 presentation): tables, funnels, statuses, short fact lists — NO long prose.
Every chart gets a title, a one-sentence fact, and the data behind it stays available as a table.
"""
import plotly.graph_objects as go
import streamlit as st

import snapshot as S
import theme as T

STATUS_COLOR = {
    "PANEL_STABLE": T.GREEN, "ASSET_CONDITIONAL": T.ACCENT, "TUNING_DEPENDENT": T.AMBER,
    "CORE_CONDITIONAL_FAILURE": T.AMBER, "REGIME_DEPENDENT": T.AMBER, "SEARCH_INFLATED": T.MUTED,
    "CROSSFIT_UNSTABLE": T.MUTED, "NO_POSITIVE_UTILITY": T.TEXT_DIM, "INSUFFICIENT_EVIDENCE": T.TEXT_DIM,
}


def snapshot_guard():
    """Fail-closed banner reading the frozen snapshot (not the product DB). Non-blocking."""
    h = S.health()
    if h["status"] != S.OK:
        st.warning("%s — %s" % (h["status"], h["detail"]))
    return h["status"] == S.OK


def fact(text):
    """A one-sentence fact under a chart (dim, no decoration)."""
    st.caption(text)


def funnel_chart(stages, title):
    """stages = [(label, value)] top→bottom. A real funnel, values labelled on the bars."""
    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    fig = go.Figure(go.Funnel(
        y=labels, x=values, textposition="inside",
        texttemplate="%{label}: %{value}", marker=dict(color=T.ACCENT),
        connector=dict(line=dict(color=T.BORDER))))
    fig.update_layout(**T.plotly_layout(height=max(220, 52 * len(stages)),
                                        title=dict(text=title, font=dict(size=13)), showlegend=False))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def status_bar(status_counts, title):
    """Horizontal bar of family statuses, ordered by the funnel, colored by severity."""
    order = [s for s in S.STATUS_ORDER if s in status_counts]
    vals = [status_counts[s] for s in order]
    fig = go.Figure(go.Bar(
        x=vals, y=order, orientation="h",
        marker=dict(color=[STATUS_COLOR.get(s, T.MUTED) for s in order]),
        text=vals, textposition="outside"))
    fig.update_layout(**T.plotly_layout(height=max(220, 34 * len(order)),
                                        title=dict(text=title, font=dict(size=13)),
                                        showlegend=False, xaxis=dict(title="families")))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def heatmap(z, x, y, title, colorscale="Blues", zmid=None):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, zmid=zmid,
                               xgap=1, ygap=1, colorbar=dict(thickness=10)))
    fig.update_layout(**T.plotly_layout(height=max(240, 26 * len(y) + 80),
                                        title=dict(text=title, font=dict(size=13))))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
