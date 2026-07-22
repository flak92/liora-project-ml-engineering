#!/usr/bin/env python3
"""The per-asset state of the Calibration DAG, derived from result artifacts and the contract.

State is a FUNCTION of the immutable result artifacts, never of the ledger. Given the same artifacts
and the same contract, `derive_state` returns the same state — so a run can be reconstructed from its
results alone, and the ledger is only an audit trail.

Every scientific threshold comes from the run's frozen contract snapshot, not from this file: the
viability floor (both `min_split_nodes` AND `min_pred_std`), the marginal-gain charge, and the Rung 5
verdict (the canonical A1 ∩ A2 ∩ B stable-survivor test in `scripts/rung5_verdict.py`). The engine
must never hold a definition of "viable" or "confirmed" that could drift from the science it runs.

The graph the states form:

    PENDING_VIABILITY -> VIABLE -> OPERATING_POINT_VALID -> UTILITY_REGISTERED -> CONFIRMED
        -> NULL_VALIDATED -> LOCALLY_OPTIMIZED -> INTERACTIONS_EVALUATED -> RESOLVED

    CONFIRMED           -> NULL_REJECTED       -> RESOLVED_EMPTY
    UTILITY_REGISTERED  -> NO_CONFIRMED_FEATURE -> RESOLVED_EMPTY
    any scientific rung -> NEEDS_CONTRACT           (a science stop: cannot proceed honestly)
    any task            -> FAILED_TECHNICAL         (an execution error: NOT a science stop)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))

RUNG_DIR = {
    1: "rung1_viability", 2: "rung2_operating_point", 3: "rung3_utility",
    4: "rung4_crossfit", 5: "rung5_null", 6: "rung6_hpo", 7: "rung7_interactions",
}
TERMINAL = {"RESOLVED", "RESOLVED_EMPTY", "NEEDS_CONTRACT"}

_CONTRACT = {}          # run_dir -> loaded contract snapshot (cached)


def _contract(run_dir):
    key = str(run_dir)
    if key not in _CONTRACT:
        _CONTRACT[key] = json.loads((Path(run_dir) / "contract.json").read_text(encoding="utf-8"))
    return _CONTRACT[key]


def _thresholds(run_dir):
    """Viability floor and the marginal charge — straight from the frozen contract, never hardcoded."""
    c = _contract(run_dir)["contract"]
    v = c["viability"]
    return (int(v["min_split_nodes"]), float(v["min_pred_std"]),
            float(c["acceptance"]["complexity_penalty"]))


def read_artifact(run_dir, rung, asset):
    """The per-asset artifact for a rung, addressed by its DETERMINISTIC task_hash — not the newest
    file by mtime. Retries of the same logical unit publish to the same path (see worker._publish),
    so there is exactly one artifact per (asset, rung), and reading it is order-independent."""
    import schemas as SC
    run_id = _contract(run_dir)["run_id"]
    th = SC.task_hash({"run_id": run_id, "asset": asset, "rung": rung})
    f = Path(run_dir) / "results" / RUNG_DIR[rung] / asset / f"{th}.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# Backwards-compatible alias — the rest of the engine calls this name.
latest_artifact = read_artifact


# ---- per-rung verdicts, from the artifact + the contract --------------------------------------

def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def _viable(art, min_sn, min_sd):
    """Rung 1: the model can learn iff the v2 draws clear BOTH frozen gates — split nodes and
    prediction spread. A model with splits but no prediction variance is a constant dressed up, so
    both gates matter; checking only split nodes (as an earlier version did) is a weaker definition
    than the science's."""
    recs = (art.get("result", {}).get("spaces", {}) or art.get("spaces", {})).get(
        "v2_hessian_relative", [])
    if not recs:
        return False
    med_sn = _median([x.get("split_nodes", 0) for x in recs])
    med_sd = _median([x.get("pred_std", 0.0) for x in recs])
    return med_sn >= min_sn and med_sd >= min_sd


def _has_marginal(art, min_gain):
    folds = (art.get("result", art)).get("folds", [])
    for f in folds:
        for s in (f.get("singles", {}) or {}).values():
            if s.get("viable") and s.get("gain", 0) > min_gain:
                return True
    return False


def _confirmed(art):
    import acceptance as ACC
    folds = (art.get("result", art)).get("folds", [])
    for f in folds:
        v = ACC.verdict(f["rotations"])
        if v["flat"]["accepted"] or v["hierarchical"]["accepted"]:
            return True
    return False


def _null_validated(art):
    """Rung 5: at least one arm stable across A1 ∩ A2 ∩ B — the canonical science-layer verdict."""
    import rung5_verdict as RV
    return RV.null_validated(art.get("result", art))


def _retained(art):
    results = (art.get("result", art)).get("results", [])
    return any(r.get("verdict") == "retained" for r in results)


# ---- the state machine ------------------------------------------------------------------------

def derive_state(run_dir, asset):
    """The asset's current state and the evidence behind it — from artifacts + contract only."""
    min_sn, min_sd, min_gain = _thresholds(run_dir)

    a1 = read_artifact(run_dir, 1, asset)
    if a1 is None:
        return {"asset": asset, "state": "PENDING_VIABILITY", "evidence": {}}
    if not _viable(a1, min_sn, min_sd):
        return {"asset": asset, "state": "NEEDS_CONTRACT", "reason": "model_not_viable",
                "required_human_action": "mint_new_contract_version", "evidence": {"rung": 1}}

    # Rung 2 (operating-point transfer) is folded into Rungs 3-4 on this panel — the quantile theta is
    # chosen on discovery and applied to confirmation there — so there is no standalone Rung 2
    # artifact; a viable model proceeds to utility.
    a3 = read_artifact(run_dir, 3, asset)
    if a3 is None:
        return {"asset": asset, "state": "VIABLE", "evidence": {"rung": 1}}
    if not _has_marginal(a3, min_gain):
        return {"asset": asset, "state": "RESOLVED_EMPTY",
                "reason": "no_marginal_candidate", "evidence": {"rung": 3}}

    a4 = read_artifact(run_dir, 4, asset)
    if a4 is None:
        return {"asset": asset, "state": "UTILITY_REGISTERED", "evidence": {"rung": 3}}
    if not _confirmed(a4):
        return {"asset": asset, "state": "RESOLVED_EMPTY",
                "reason": "no_confirmed_feature", "evidence": {"rung": 4}}

    a5 = read_artifact(run_dir, 5, asset)
    if a5 is None:
        return {"asset": asset, "state": "CONFIRMED", "evidence": {"rung": 4}}
    if not _null_validated(a5):
        return {"asset": asset, "state": "RESOLVED_EMPTY",
                "reason": "no_stable_survivor_across_nulls", "evidence": {"rung": 5}}

    a6 = read_artifact(run_dir, 6, asset)
    if a6 is None:
        return {"asset": asset, "state": "NULL_VALIDATED", "evidence": {"rung": 5}}

    # Rung 6 can only demote; retained == 0 is still a valid, complete result. Either way the asset is
    # locally optimized and resolved; interactions (rung 7) are SPECIFIED / UNVALIDATED and do not
    # gate resolution.
    return {"asset": asset, "state": "RESOLVED",
            "evidence": {"rung": 6, "retained_any": _retained(a6),
                         "interaction_status": "SPECIFIED_UNVALIDATED"}}


def is_terminal(state):
    return state in TERMINAL
