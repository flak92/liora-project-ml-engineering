"""Smart Methodology — the Rung 0–9 ladder and the five-number funnel, native.

Read-only, from the frozen snapshot. One genuinely live element remains: the field-level guard, re-run
in Python on every visit, proving the loop cannot loosen its own proof standard. SSOT: docs/SMART_METHODOLOGY.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import present as P
import snapshot as S

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))
import contract_patch as CP                                                    # noqa: E402

C.page_header("Smart Methodology",
              "The frozen proof standard: a feature is confirmed only if it survives data that did not choose it")
P.snapshot_guard()

ft = S.family_transfer("full") or {}
fn = ft.get("funnel", {})

# ── the funnel ──────────────────────────────────────────────────────────────────────────────────────
st.subheader("The whole story in five numbers")
if fn:
    P.funnel_chart([
        ("provisional (cross-fit accepted)", fn.get("provisional_crossfit", 0)),
        ("passed A1 max-null (marginal)", fn.get("passed_a1_marginal", 0)),
        ("stable A1 × A2 × B", fn.get("stable_a1_a2_b", 0)),
        ("retained after survivor tuning", fn.get("retained_rung6", 0)),
        ("unique feature", fn.get("unique_retained_representatives", 0)),
    ], "Funnel — derived from the frozen artifacts")
    P.fact("A property of the data, not a target: a fresh panel may produce 30 → 7 → 2 → 0 and be just "
           "as correct. Both retained arms resolve to the same feature (representative 112).")
else:
    st.info("Run `make family-transfer` to compute the funnel from the snapshot.")

# ── the ladder ──────────────────────────────────────────────────────────────────────────────────────
st.subheader("The ladder — Rung 0–9")
st.dataframe(
    [{"rung": r, "question": q, "unit": u, "status": s} for (r, q, u, s) in S.LADDER],
    width="stretch", hide_index=True)
P.fact("Each asset walks this as a state machine driven by immutable artifacts + the frozen contract — "
       "never by the scheduler or another asset. Rung 8 (family transfer) is added by this work.")

# ── four ways a candidate dies ──────────────────────────────────────────────────────────────────────
st.subheader("Four ways a candidate dies")
C.metric_row([("search-inflation", "Rung 5 max-null"), ("regime-dependence", "A2"),
              ("tuning-dependence", "Rung 6"), ("asset-specificity", "Rung 8")])

# ── the live guard ──────────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("The proof standard cannot be loosened — proven live")
LOOSEN = {"rung_6_survivor_hpo": {"own_null": {"permutations": 5}}}
try:
    CP.guard(LOOSEN)
    st.error("REGRESSION — the guard did NOT reject a patch weakening the null from M=50 to M=5.", icon="⛔")
except CP.PatchRejected as e:
    st.success("The guard rejects weakening the headline null (M=50 → M=5): a frozen leaf. "
               "Re-run in Python on every visit.", icon="🔒")
    st.caption(str(e))
