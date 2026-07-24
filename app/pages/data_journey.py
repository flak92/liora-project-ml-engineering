"""Data Journey — where the features come from, and the wall the data never crosses.

Native Plotly, read-only, from the frozen snapshot + contract. OHLCV → frozen 1h core → 45 searchable
candidates in 12 families → Train-only validation → the OOS boundary (oos_reads = 0). Facts, not prose.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

import components as C
import present as P
import snapshot as S
import theme as T

C.page_header("Data Journey",
              "Raw OHLCV → frozen 1h core → 45 candidates / 12 families → Train-only → OOS boundary (oos_reads = 0)")
P.snapshot_guard()

db = S.data_boundary()
idy = S.contract_identity()
fam = S.families_config()
n_candidates = sum(len(v) for v in fam.values())

# ── the road, as a labelled flow ────────────────────────────────────────────────────────────────────
st.subheader("The road the data travels")
stages = ["OHLCV bars (1h)", "Frozen 1h core (17)", "45 candidates", "12 families",
          "Train-only validation", "Frozen OOS boundary"]
fig = go.Figure()
for i, s in enumerate(stages):
    fig.add_shape(type="rect", x0=i, x1=i + 0.9, y0=0, y1=1,
                  line=dict(color=T.ACCENT), fillcolor=T.SURFACE)
    fig.add_annotation(x=i + 0.45, y=0.5, text=s, showarrow=False,
                       font=dict(color=T.TEXT, size=11))
    if i < len(stages) - 1:
        fig.add_annotation(x=i + 0.95, y=0.5, text="→", showarrow=False, font=dict(color=T.TEXT_DIM))
fig.update_layout(**T.plotly_layout(height=110, showlegend=False,
                                    xaxis=dict(visible=False, range=[-0.1, len(stages)]),
                                    yaxis=dict(visible=False, range=[-0.1, 1.1])))
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
P.fact("The 1h core (ids 1–17) is frozen and never searched; only the 45 non-1h candidates are.")

# ── the facts table ─────────────────────────────────────────────────────────────────────────────────
st.subheader("The facts")
C.metric_row([("timeframe", "1h"), ("core (frozen)", "17"), ("candidates", str(n_candidates)),
              ("families", str(len(fam)))])
C.metric_row([("train end", db["train_end"] or "—"), ("oos start", db["oos_start"] or "—"),
              ("oos reads", str(db["oos_reads"])), ("embargo bars", str(db["embargo_bars"] or "—"))])
st.caption("contract %s · sample %s… · the OOS window is never read before certification"
           % (idy["contract_version"], (idy["sample_sha256_prefix"] or "")[:8]))

# ── the 12 families ─────────────────────────────────────────────────────────────────────────────────
st.subheader("The 45 candidates, grouped by mechanism (12 families)")
rows = [{"family": k, "candidates": len(v), "feature ids": ", ".join(str(i) for i in v)}
        for k, v in sorted(fam.items())]
st.dataframe(rows, width="stretch", hide_index=True)
P.fact("A family is a taxonomy of an information mechanism — the search evaluates candidates per family "
       "and keeps the simplest one-SE representative. Editing this map is a new search epoch.")

st.divider()
st.info("**No look-ahead, no leakage.** Every feature reads only closed bars; the operating point and "
        "every choice are calibrated on Train out-of-fold; the OOS window (from %s) is scored exactly "
        "once per asset and never used to choose anything (`oos_reads = %s`)."
        % (db["oos_start"] or "—", db["oos_reads"]), icon="🔒")
