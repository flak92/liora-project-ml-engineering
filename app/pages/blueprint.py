"""Pipeline LEGO Blueprint — the procedure as an 18-brick ladder, with the reasoning kept.

The blueprint is a self-contained HTML file at the repository root: zero dependencies, zero
network requests, and it opens in any browser on its own. This page only embeds it.

Each brick carries its contract (input, transform, output, invariants, knobs, tests), the layer
id the code actually uses (XGB L1–L9, LSTM D1–D9), and a "HOW WE THOUGHT · WHAT WE LEARNED"
record. Ladder order is pipeline order, fixed by declaration — there is nothing to drag.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

import components as C
import data

ROOT = Path(__file__).resolve().parents[2]
BLUEPRINT = ROOT / "pipeline_lego_blueprint.html"


@st.cache_data
def _load(mtime: float) -> str:
    """The self-contained blueprint, cached per file mtime."""
    return BLUEPRINT.read_text(encoding="utf-8")


LESSONS = """\
**Five lessons the ladder is built around**

1. **Optimize the thing you will be judged on.** The tuning objective moved from a ranking score
   to Train-out-of-fold trading log-growth replayed through the real engine — the proxy metric had
   been selecting models that ranked well and traded poorly (brick C1).
2. **Degrees of freedom are earned, not granted.** The first per-asset threshold calibration
   overfit; the knob was un-pinned only once selection became floor-respecting — one shared
   operating point per trial, plus a trade floor (brick C2).
3. **Purge and embargo are not optional decorations.** With horizon labels the naive
   warm-up/train/OOS split leaks by construction (bricks A4, G.1).
4. **An honest negative from a trustworthy method beats a positive from a leaky one.** The
   full-universe one-shot verdict is committed with the same ceremony a positive would get
   (brick E1).
5. **Explaining a model must be cheaper than building it.** The interpretation layer reads
   artifacts and Train rows only — an explanation that needs the test set is a second read
   (brick E3).
"""

C.page_header("Pipeline LEGO Blueprint", "The sealed procedure as an 18-brick ladder — contract, "
                                         "reasoning and lesson per brick.")
C.guard(stop=False)  # static page: banner without stopping

_run = data.research_run()
_n_xgb, _n_lstm = _run.get("xgb_assets", 0), _run.get("lstm_assets", 0)

st.write(
    "Bottom → top, four named stages. **DATA**: split-adjusted bars (A1) → leakage-safe split "
    "(A4). **CONFIGURATION**: the learning problem itself — which bars become trade candidates, "
    "what counts as a win, and which columns the model is allowed to see (B1–B4). "
    "**TRAINING**: profit-aligned Train-only tuning (per-asset Optuna for XGB; one warm-started "
    "backbone for LSTM) and the per-asset operating point (C1–C3). **ENDPRODUCT**: the ledgered "
    "OOS verdict, the honest benchmark, the Train-derived interpretation and this console "
    "(E1–E4). Between them sit fail-closed **VALIDATORS** — they stop a run outright; there is "
    "no warning tier and they never repair what they find. The **XGBOOST | LSTM** switch flips "
    "only the bricks whose logic depends on the model."
)
st.caption(
    "Two words worth pinning down, because both are used loosely elsewhere. CONFIGURATION here "
    "configures the **problem**, not the model — model hyper-parameters are not set by hand at "
    "all, they are searched under TRAINING. And VALIDATORS are gates, not the validation split: "
    "the purged walk-forward folds live inside TRAINING and are a different object entirely."
)
st.caption(
    f"Sealed in this release: {_n_xgb} XGB · {_n_lstm} LSTM. Counts read from the store; the "
    "blueprint's own figures are frozen. Click a brick for its contract and its lesson; the "
    "tuning-trial cell values are marked SCHEMATIC and are not recorded trials — for LSTM no "
    "per-asset study ran in this epoch (the cold-start path exists behind LSTM_COLD_START=1)."
)
st.caption(
    "Drag to pan · **Ctrl + wheel** (or trackpad pinch) to zoom · a plain wheel scrolls this "
    "page · double-click the background to re-fit."
)

if not BLUEPRINT.exists():
    st.error(f"Blueprint not found: {BLUEPRINT.name} (expected at the repository root).")
    st.stop()

# The board is 1184x2706 and its fit is height-bound below ~1690px of frame, so 780 rendered
# every brick title at under 3px. 1500 nearly doubles the fit (0.28 -> 0.55) and stays inside
# that ceiling — past it width binds and the extra height would only add empty bands.
st.iframe(_load(BLUEPRINT.stat().st_mtime), height=1500)
st.markdown(LESSONS)
st.caption("Standalone file: pipeline_lego_blueprint.html (repository root — opens in any browser).")
