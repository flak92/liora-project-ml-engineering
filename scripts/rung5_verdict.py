#!/usr/bin/env python3
"""The canonical Rung 5 verdict — the one definition of a stable survivor, so nothing restates it.

A per-asset Rung 5 result holds three nulls: `a1` (marginal), `a2` (regime) and `b` (conditional).
An arm is a *stable survivor* only if it passed the acceptance-contract statistic against ALL THREE,
for the SAME (outer_fold, arm, candidate). A1 alone is the primary null; A2 and B are sensitivity
checks, and a survivor that leans on carrying blocks across regimes (fails A2) or on breaking the
coupling to core (fails B) is not stable. The asset reaches NULL_VALIDATED only if at least one arm
is stable across the three.

Both the engine's state machine and the report import this — there is exactly one place where "who
survived the null" is decided, and it is here, in the science layer.
"""


def passed_arms(entry):
    """(outer_fold, arm, candidate) that PASSED the procedure-level null in one per-asset null entry.

    `entry` is a single asset's null result (the `tables[asset]` a runner produced): {folds: [{
    outer_fold, arms: {flat|hierarchical: {verdict, unit, ...}}}]}. A "passed" verdict already carries
    the full acceptance contract — the null only ever tested cross-fit-accepted arms, and passing
    means the real acceptance statistic survived the permutation distribution.
    """
    out = set()
    for f in (entry or {}).get("folds", []):
        for arm, v in (f.get("arms", {}) or {}).items():
            if v.get("verdict") == "passed":
                out.add((f.get("outer_fold"), arm, str(v.get("unit"))))
    return out


def stable_survivors(a1, a2, b):
    """Arms stable across the three nulls, matched on (outer_fold, arm, candidate).

    A missing sensitivity null (a2 or b absent) cannot confirm stability, so it is treated as failing
    that leg — stability requires positive evidence from every null that the design ran, never the
    benefit of the doubt.
    """
    pa1 = passed_arms(a1)
    if not pa1:
        return set()
    pa2 = passed_arms(a2) if a2 else set()
    pb = passed_arms(b) if b else set()
    return pa1 & pa2 & pb


def null_validated(rung5_result):
    """True iff at least one arm is stable across A1 ∩ A2 ∩ B."""
    r = rung5_result or {}
    return len(stable_survivors(r.get("a1"), r.get("a2"), r.get("b"))) > 0


def stable_units(rung5_result):
    """The confirmed candidates (for the report / compiler output)."""
    r = rung5_result or {}
    return sorted({u for _, _, u in stable_survivors(r.get("a1"), r.get("a2"), r.get("b"))})


def full_strength(a1_panel, crossfit_panel, frozen_max):
    """Did this null run EARN the right to confirm? A survivor counts ONLY from a full-strength run,
    on BOTH axes — else a fast smoke (--permutations / --folds) would silently manufacture a
    confirmation it had no power to make (exactly the ADBE class: a 'discovery' from a run that could
    not confirm it). Returns (ok, reason); ok=True only when:

      permutation budget:  a1_panel.contract.permutations_max >= frozen_max (no --permutations cap)
      fold scope:          every crossfit-accepted (ticker, outer_fold) with T>0 is covered by a1
                           (no --folds cap)

    A weak run's PASSED verdicts are mechanics evidence, never confirmations. The byte-level integrity
    guard stays orthogonal — this gates SCIENTIFIC power, not reproducibility.
    """
    if not a1_panel:
        return False, "brak panelu a1"
    cap = (a1_panel.get("contract") or {}).get("permutations_max")
    if cap is None or cap < frozen_max:
        return False, f"permutations_max={cap} < {frozen_max} (smoke cap)"
    # Explicit fold-axis signal recorded by the runner (procedure_null contract.folds_cap). Checked
    # directly so the fold axis never stops defending silently — the crossfit comparison below is a
    # belt for legacy artifacts that predate the field, not the primary guard.
    folds_cap = (a1_panel.get("contract") or {}).get("folds_cap")
    if folds_cap:
        return False, f"folds_cap={folds_cap} (smoke --folds)"
    null_folds = {(t, f.get("outer_fold")) for t, rec in a1_panel.get("tables", {}).items()
                  for f in (rec.get("folds") or [])}
    scope = set()
    for t, rec in (crossfit_panel or {}).get("tables", {}).items():
        for f in rec.get("folds", []):
            for arm in ("flat", "hierarchical"):
                v = (f.get("verdict") or {}).get(arm, {})
                if v.get("accepted") and float(v.get("T", 0) or 0) > 0:
                    scope.add((t, f.get("outer_fold")))
    missing = scope - null_folds
    if missing:
        return False, f"foldy ograniczone (--folds): {len(missing)}/{len(scope)} scope niepokryte"
    return True, "pełna siła"
