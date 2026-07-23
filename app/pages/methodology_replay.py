"""Methodology Replay — the real run, time-compressed into seven epistemic acts.

A REPLAY, not a live computation: every event (the funnel 26→11→9→2, the rung transitions, the
max-null b-counter, the futility-stops, the verdicts, the terminals) is reconstructed from the
committed snapshot panels — the board computes nothing. One element IS genuinely live: the
field-level guard. This page re-runs it in Python on every visit and renders its verdict above the
board, so "the loop cannot loosen its own proof standard" is demonstrated here, not merely asserted.
The board itself is the self-contained methodology_replay.html (built by scripts/build_replay.py,
which embeds a seal-audited fallback of the same guard). Full method: docs/FEATURE_DISCOVERY_METHODOLOGY.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "methodology_replay.html"

sys.path.insert(0, str(ROOT / "engine"))
import contract_patch as CP                                                    # noqa: E402

# The loosening patch Act 1b runs LIVE: weaken the headline null from M=50 to M=5. own_null.permutations
# is a frozen leaf, so the guard must reject it — every visit, in Python, not from a cache.
LOOSEN = {"rung_6_survivor_hpo": {"own_null": {"permutations": 5}}}


def _guard_verdict():
    try:
        CP.guard(LOOSEN)
        return False, "GUARD NIE ODRZUCIŁ patcha — REGRESJA proof-standard"
    except CP.PatchRejected as e:
        return True, str(e)


@st.cache_data
def _load(mtime: float) -> str:
    """The self-contained replay board, cached per file mtime."""
    return PAGE.read_text(encoding="utf-8")


C.page_header("Methodology Replay", "")
C.guard(stop=False)  # narrative reference page: banner without stopping

if not PAGE.exists():
    st.error(f"Not found: {PAGE.name} — run `make replay`.")
    st.stop()

rejected, message = _guard_verdict()
if rejected:
    st.success(f"**Act 1b — guard LIVE (run now, in Python).** Patch "
               f"`rung_6_survivor_hpo.own_null.permutations: 50 → 5` → **PatchRejected**. {message}")
else:
    st.error(f"**Act 1b — guard REGRESSION.** {message}")

# Fixed-aspect board (built HTML pins itself to 900px, overflow:hidden), not a flowing document.
st.iframe(_load(PAGE.stat().st_mtime), height=900)
st.caption("Standalone file: methodology_replay.html (repository root — opens in any browser). "
           "Built by scripts/build_replay.py from results/methodology_snapshot/*; sealed by "
           "scripts/verify_replay.py. The permutation stream is thinned for the eye; the b-counter "
           "and the per-unit seconds are exact from the artifacts.")
