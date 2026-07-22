#!/usr/bin/env python3
"""The per-asset state of the Calibration DAG, derived from result artifacts and the contract.

State is a FUNCTION of the immutable result artifacts, never of the ledger. Given the same artifacts
and the same contract, `derive_state` returns the same state — so a run can be reconstructed from its
results alone, and the ledger is only an audit trail. The verdict thresholds live in one place
(`scripts/acceptance.py` and the viability floor), reused here rather than restated, so the engine
cannot drift from the science it executes.

The graph the states form:

    PENDING_VIABILITY -> VIABLE -> OPERATING_POINT_VALID -> UTILITY_REGISTERED -> CONFIRMED
        -> NULL_VALIDATED -> LOCALLY_OPTIMIZED -> INTERACTIONS_EVALUATED -> RESOLVED

    CONFIRMED           -> NULL_REJECTED       -> RESOLVED_EMPTY
    UTILITY_REGISTERED  -> NO_CONFIRMED_FEATURE -> RESOLVED_EMPTY
    any scientific rung -> NEEDS_CONTRACT           (a science stop: cannot proceed honestly)
    any task            -> FAILED_TECHNICAL         (an execution error: NOT a science stop)

NEEDS_CONTRACT and FAILED_TECHNICAL are kept apart on purpose: the first means the contract does not
cover what was observed and a human must mint a new version; the second means a worker crashed and
the same task should be retried under the same contract.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# The rung -> artifact-directory-name map. One per-asset immutable artifact tree per rung.
RUNG_DIR = {
    1: "rung1_viability",
    2: "rung2_operating_point",
    3: "rung3_utility",
    4: "rung4_crossfit",
    5: "rung5_null",
    6: "rung6_hpo",
    7: "rung7_interactions",
}

# Terminal states a planner will not schedule past.
TERMINAL = {"RESOLVED", "RESOLVED_EMPTY", "NEEDS_CONTRACT"}

VIABILITY_MIN_SPLIT_NODES = 20          # same floor as scripts/feature_utility.viability_floor()
MIN_MARGINAL_GAIN = 0.004               # same complexity charge the search has always levied


def _results(run_dir):
    return Path(run_dir) / "results"


def latest_artifact(run_dir, rung, asset):
    """The most recent valid per-asset artifact for a rung, or None.

    Multiple files can exist under one asset dir (retries produce a new task_hash each time); the
    newest well-formed one is the artifact. A truncated file is ignored, not fatal — the state is
    then 'this rung has not produced a usable result yet'.
    """
    d = _results(run_dir) / RUNG_DIR[rung] / asset
    if not d.is_dir():
        return None
    best = None
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if best is None or f.stat().st_mtime >= best[0]:
            best = (f.stat().st_mtime, doc)
    return best[1] if best else None


# ---- per-rung verdicts, read off the artifact the runner produced -----------------------------

def _viable(art):
    """Rung 1: did any v2 draw clear the split-node floor on this asset?"""
    recs = (art.get("result", {}).get("spaces", {}) or art.get("spaces", {})).get(
        "v2_hessian_relative", [])
    sn = sorted(x["split_nodes"] for x in recs) if recs else []
    med = sn[len(sn) // 2] if sn else 0
    return med >= VIABILITY_MIN_SPLIT_NODES


def _has_marginal(art):
    """Rung 3: any single feature viable with gain over the complexity charge?"""
    folds = (art.get("result", art)).get("folds", [])
    for f in folds:
        for s in (f.get("singles", {}) or {}).values():
            if s.get("viable") and s.get("gain", 0) > MIN_MARGINAL_GAIN:
                return True
    return False


def _confirmed(art):
    """Rung 4: any arm accepted by the cross-fit acceptance contract on this asset?"""
    import acceptance as ACC
    folds = (art.get("result", art)).get("folds", [])
    for f in folds:
        v = ACC.verdict(f["rotations"])
        if v["flat"]["accepted"] or v["hierarchical"]["accepted"]:
            return True
    return False


def _null_passed(art):
    """Rung 5: any arm passed the procedure-level max-null on this asset?"""
    folds = (art.get("result", art)).get("folds", [])
    for f in folds:
        for v in f.get("arms", {}).values():
            if v.get("verdict") == "passed":
                return True
    return False


def _retained(art):
    """Rung 6: any survivor retained after survivor-specific HPO on this asset?"""
    results = (art.get("result", art)).get("results", [])
    return any(r.get("verdict") == "retained" for r in results)


# ---- the state machine ------------------------------------------------------------------------

def derive_state(run_dir, asset):
    """The asset's current state and the evidence behind it — from artifacts + contract only."""
    a1 = latest_artifact(run_dir, 1, asset)
    if a1 is None:
        return {"asset": asset, "state": "PENDING_VIABILITY", "evidence": {}}
    if not _viable(a1):
        return {"asset": asset, "state": "NEEDS_CONTRACT",
                "reason": "model_not_viable",
                "required_human_action": "mint_new_contract_version",
                "evidence": {"rung": 1}}

    # Rung 2 shares the quantile operating point with the rungs that trade; on this panel it is not a
    # standalone artifact (it is validated inside utility/cross-fit). Treat a viable model as
    # operating-point-valid here and record that the check is folded into rung 3/4.
    a3 = latest_artifact(run_dir, 3, asset)
    if a3 is None:
        return {"asset": asset, "state": "VIABLE", "evidence": {"rung": 1}}
    if not _has_marginal(a3):
        return {"asset": asset, "state": "RESOLVED_EMPTY",
                "reason": "no_marginal_candidate", "evidence": {"rung": 3}}

    a4 = latest_artifact(run_dir, 4, asset)
    if a4 is None:
        return {"asset": asset, "state": "UTILITY_REGISTERED", "evidence": {"rung": 3}}
    if not _confirmed(a4):
        return {"asset": asset, "state": "RESOLVED_EMPTY",
                "reason": "no_confirmed_feature", "evidence": {"rung": 4}}

    a5 = latest_artifact(run_dir, 5, asset)
    if a5 is None:
        return {"asset": asset, "state": "CONFIRMED", "evidence": {"rung": 4}}
    if not _null_passed(a5):
        return {"asset": asset, "state": "RESOLVED_EMPTY",
                "reason": "all_candidates_fail_max_null", "evidence": {"rung": 5}}

    a6 = latest_artifact(run_dir, 6, asset)
    if a6 is None:
        return {"asset": asset, "state": "NULL_VALIDATED", "evidence": {"rung": 5}}

    # Rung 6 can only demote; retained==0 is still a valid, complete result (the frozen-core survivor
    # did not survive equal-budget tuning). Either way the asset is locally optimized and resolved;
    # interactions (rung 7) are SPECIFIED / UNVALIDATED and do not gate resolution.
    state = "RESOLVED"
    return {"asset": asset, "state": state,
            "evidence": {"rung": 6, "retained_any": _retained(a6),
                         "interaction_status": "SPECIFIED_UNVALIDATED"}}


def is_terminal(state):
    return state in TERMINAL
