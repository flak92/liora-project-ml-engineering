"""Smart Methodology — the calibration-configurables map and the run replay, in one page.

Two tabs, both read-only:
- **Map & Configurables** embeds calibration_configurables.html (the methodology map + every tunable
  number as a range, its FROZEN/ADMISSIBLE state, and which knob to widen when a rung comes up empty).
- **Replay** embeds methodology_replay.html — the real run reconstructed from the committed snapshot,
  with ONE genuinely live element: the field-level guard, re-run in Python on every visit, proving the
  loop cannot loosen its own proof standard. SSOT: docs/SMART_METHODOLOGY.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "calibration_configurables.html"
REPLAY = ROOT / "methodology_replay.html"

sys.path.insert(0, str(ROOT / "engine"))
import contract_patch as CP                                                    # noqa: E402

# The loosening patch the replay runs LIVE: weaken the headline null from M=50 to M=5.
# own_null.permutations is a frozen leaf, so the guard must reject it — every visit, in Python.
LOOSEN = {"rung_6_survivor_hpo": {"own_null": {"permutations": 5}}}


def _guard_verdict():
    try:
        CP.guard(LOOSEN)
        return False, "GUARD NIE ODRZUCIŁ patcha — REGRESJA proof-standard"
    except CP.PatchRejected as e:
        return True, str(e)


@st.cache_data
def _load(path: str, mtime: float) -> str:
    """A self-contained board, cached per (file, mtime)."""
    return Path(path).read_text(encoding="utf-8")


C.page_header("Smart Methodology", "")
C.guard(stop=False)  # reference page: banner without stopping

tab_map, tab_replay = st.tabs(["Map & Configurables", "Replay"])

with tab_map:
    if not CATALOG.exists():
        st.error(f"Not found: {CATALOG.name} (expected at the repository root).")
    else:
        # A flowing document sized to its content (the methodology map + the catalog).
        st.iframe(_load(str(CATALOG), CATALOG.stat().st_mtime), height=5400)
        st.caption("Standalone file: calibration_configurables.html (repository root — opens in any "
                   "browser). SSOT: docs/SMART_METHODOLOGY.md.")

with tab_replay:
    if not REPLAY.exists():
        st.error(f"Not found: {REPLAY.name} — run `make replay`.")
    else:
        rejected, message = _guard_verdict()
        if rejected:
            st.success("**Guard LIVE (run now, in Python).** Patch "
                       "`rung_6_survivor_hpo.own_null.permutations: 50 → 5` → "
                       f"**PatchRejected**. {message}")
        else:
            st.error(f"**Guard REGRESSION.** {message}")
        # Fixed-aspect board (the built HTML pins itself to 900px, overflow:hidden).
        st.iframe(_load(str(REPLAY), REPLAY.stat().st_mtime), height=900)
        st.caption("Standalone file: methodology_replay.html (repository root). Built by "
                   "scripts/build_replay.py from results/methodology_snapshot/*; sealed by "
                   "scripts/verify_replay.py. The permutation stream is thinned for the eye; the "
                   "b-counter and per-unit seconds are exact from the artifacts.")
