"""Family Transfer (Rung 8) — which OHLCV families travel across assets.

Native Plotly over the frozen family_transfer.json. Facts, funnels, statuses, coverage heatmaps,
the minimal panel family set, and the taxonomy review — no long prose. Read-only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import present as P
import snapshot as S

C.page_header("Family Transfer · Rung 8",
              "The smallest set of independent OHLCV mechanism families whose representatives survive and transfer")

# panel toggle: canonical 20-asset (parity) vs the six-asset study panel
view = st.radio("Panel", ["Full snapshot (20 assets · parity)", "Study panel (6 assets · development)"],
                horizontal=True, label_visibility="collapsed")
panel = "full" if view.startswith("Full") else "panel6"
ft = S.family_transfer(panel)

if not ft:
    st.warning("family_transfer%s.json not found — run `make family-transfer%s`."
               % (("" if panel == "full" else "_panel6"), ("" if panel == "full" else "-panel6")))
    st.stop()

fn = ft["funnel"]
fams = ft["families"]
integ = ft["integrity"]

if panel == "panel6":
    st.caption("Six-asset development experiment (%s) — **not a certification**; partial artifact "
               "coverage is expected and reported honestly." % ", ".join(S.panel6_assets()))

# ── transfer funnel ─────────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2])
with c1:
    P.funnel_chart([
        ("provisional (cross-fit)", fn["provisional_crossfit"]),
        ("passed A1 (marginal)", fn["passed_a1_marginal"]),
        ("stable A1×A2×B", fn["stable_a1_a2_b"]),
        ("retained (Rung 6)", fn["retained_rung6"]),
        ("unique representative", fn["unique_retained_representatives"]),
    ], "Transfer funnel")
with c2:
    counts = {}
    for f in fams:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    P.status_bar(counts, "Family statuses (12)")
P.fact("Global note: on the frozen snapshot no family is PANEL_STABLE — the one retained feature "
       "(rep 112, oscillator_rsi) transfers on ORLY alone. Asset-specificity, made mechanical.")

# ── family status table (filterable) ────────────────────────────────────────────────────────────────
st.subheader("Family status table")
all_status = [s for s in S.STATUS_ORDER if any(f["status"] == s for f in fams)]
pick = st.multiselect("Filter by status", all_status, default=all_status)
rows = []
for f in sorted(fams, key=lambda x: (S.STATUS_ORDER.index(x["status"]) if x["status"] in S.STATUS_ORDER else 99, x["family"])):
    if f["status"] not in pick:
        continue
    rows.append({
        "family": f["family"], "status": f["status"],
        "cross-fit": f["crossfit_accepted_count"], "A1": f["a1_pass_count"], "A2": f["a2_pass_count"],
        "B": f["b_pass_count"], "stable": f["stable_a1_a2_b_count"],
        "retained": f["rung6_retained_count"], "demoted": f["rung6_demoted_count"],
        "asset cov.": f["asset_coverage"], "uniq reps": f["unique_retained_representatives"],
    })
st.dataframe(rows, width="stretch", hide_index=True)
P.fact("Exactly one status per family, by the furthest funnel stage its arms reached. "
       + " · ".join("%s: %s" % (s, S.STATUS_MEANING[s]) for s in all_status[:3]))

# ── coverage heatmap: family × funnel stage ─────────────────────────────────────────────────────────
st.subheader("Where each family reaches in the funnel")
order = [f["family"] for f in sorted(fams, key=lambda x: -(x["stable_a1_a2_b_count"] * 10 + x["crossfit_accepted_count"]))]
stage_cols = ["cross-fit", "A1", "A2", "B", "stable", "retained"]
zmap = {f["family"]: [f["crossfit_accepted_count"], f["a1_pass_count"], f["a2_pass_count"],
                      f["b_pass_count"], f["stable_a1_a2_b_count"], f["rung6_retained_count"]] for f in fams}
z = [zmap[f] for f in order]
P.heatmap(z, stage_cols, order, "Arms per family at each funnel stage (darker = more)")

# ── minimal panel family set ────────────────────────────────────────────────────────────────────────
st.subheader("Minimal panel family set")
mp = ft["minimal_panel"]
mset = ft["minimal_panel_family_set"]
if mset:
    cov = mp["coverage"]
    C.metric_row([("families", str(len(mset))),
                  ("representatives", ", ".join(str(r) for r in mp["representatives"])),
                  ("asset-folds covered", "%d / %d" % (cov["confirmed_units_covered"], cov["confirmed_units_total"]))])
    st.dataframe([{"family": f} for f in mset], width="stretch", hide_index=True)
else:
    st.info("Empty set — a valid, honest result: no family's representative both survives the full "
            "procedure and covers a confirmed asset-fold on this panel.", icon="✅")
P.fact("A set-cover over confirmed asset × outer_fold units, NOT a ranking. Ties break: fewer families, "
       "smaller complexity, better confirmation sign, then name.")

# ── taxonomy diagnostics ────────────────────────────────────────────────────────────────────────────
st.subheader("Is the 12-family split right?")
tax = ft["taxonomy_diagnostics"]
trows = [{"family": t["family"], "width": t["family_width"],
          "rep switch rate": t["representative_switch_rate"],
          "effect heterogeneity": t["family_effect_heterogeneity"],
          "suggestion": t["suggestion"]} for t in tax]
st.dataframe(trows, width="stretch", hide_index=True)
review = [t["family"] for t in tax if str(t["suggestion"]).startswith("REVIEW")]
if review:
    P.fact("Flagged for review: %s. A suggestion only — changing feature_families_xgb.json is a new "
           "search epoch, never an automatic edit." % ", ".join(review))

# ── integrity ───────────────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Integrity")
ok = integ["status"] in ("PASS", "MISSING_REQUIRED")
(st.success if ok else st.error)(
    "integrity: %s · run_id %s · contract %s · seed %s"
    % (integ["status"], ft["run_identity"]["run_id"], ft["run_identity"]["contract_version"],
       ft["run_identity"]["seed"]), icon="✅" if ok else "⛔")
if integ["missing_inputs"]:
    st.caption("missing inputs (single-utility metrics unavailable here): %s — present in a fresh "
               "`--run-dir` run." % ", ".join(integ["missing_inputs"]))
for k in ("unknown_feature_ids", "unmapped_feature_ids", "duplicate_family_memberships"):
    if integ.get(k):
        st.warning("%s: %s" % (k, integ[k]))
st.caption("Pure read of frozen artifacts + family registry + frozen contract. Never trains, never "
           "opens the bar store, never reads OOS. OOS read count = 0.")
