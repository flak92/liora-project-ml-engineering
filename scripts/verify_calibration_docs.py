#!/usr/bin/env python3
"""Staleness lint for the calibration docs — the minimum guard against hand-copied numbers lying.

`docs/CALIBRATION_CONFIGURABLES.md` and `calibration_configurables.html` carry numbers copied from the
frozen contract and the snapshot artifacts: the funnel, the per-arm b/M, the contract hash. If the
contract or the snapshot changes and a doc is not updated, those numbers lie silently. This makes the
lie LOUD: it recomputes the canonical SEAL from `contract_loader.assemble()` + `report.funnel()` + the
rung-6 artifact, and fails if that exact seal is not embedded verbatim in both docs. It computes nothing
scientific — it only reads and compares (a documentation guard, not a generator).

    python3 scripts/verify_calibration_docs.py           # verify; exit 1 on drift
    python3 scripts/verify_calibration_docs.py --emit     # print the current seal to paste into the docs
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))

SNAPSHOT = ROOT / "results" / "methodology_snapshot"
DOCS = [ROOT / "docs" / "CALIBRATION_CONFIGURABLES.md", ROOT / "calibration_configurables.html"]
MARK = "CALIBRATION-SEAL"


def seal():
    """The canonical, deterministic seal — recomputed from the contract + artifacts, no wall-clock."""
    import contract_loader as CL
    import contract_patch as CP
    import report as RE
    assembled = CL.assemble()
    fn = RE.funnel(str(SNAPSHOT))
    r6 = json.loads((SNAPSHOT / "rung6_survivor_hpo.json").read_text(encoding="utf-8"))
    retained = [r for r in r6["results"] if r.get("verdict") == "retained"]
    ident, db, rt = assembled.get("identity", {}), assembled.get("data_boundary", {}), assembled.get("runtime", {})
    return {
        "contract_hash": CP._hash(assembled)[:16],
        "contract_version": ident.get("contract_version"),
        "funnel": [fn["provisional_crossfit"], fn["passed_a1_marginal"],
                   fn["stable_a1_a2_b"], fn["retained_rung6"]],
        "retained_units": sorted(fn["retained_units"]),
        "retained_representatives": sorted({r["representative"] for r in retained}),
        "retained_b_over_M": sorted(f"{r['exceedances']}/{r['permutations']}" for r in retained),
        "data_boundary": [db.get("train_end"), db.get("oos_start")],
        "seed": rt.get("seed", 42),
    }


def seal_str(s):
    return json.dumps(s, sort_keys=True, ensure_ascii=False)


def main():
    payload = seal_str(seal())
    if "--emit" in sys.argv:
        print(f"{MARK} {payload} {MARK}")
        return 0
    print("verify-calibration-docs — pieczęć vs bieżący kontrakt + snapshot\n")
    ok = True
    for doc in DOCS:
        if not doc.exists():
            print(f"  BRAK    {doc.name}")
            ok = False
        elif payload in doc.read_text(encoding="utf-8"):
            print(f"  OK      {doc.name}: pieczęć aktualna")
        else:
            print(f"  STALE   {doc.name}: pieczęć NIE zgadza się z kontraktem/snapshotem")
            ok = False
    if not ok:
        print(f"\n  Odśwież: python3 scripts/verify_calibration_docs.py --emit"
              f"  (wklej ciąg między znacznikami {MARK} … {MARK})")
        print(f"  Oczekiwana pieczęć: {payload}")
    print("\n" + ("PIECZĘĆ ZGODNA" if ok else "PIECZĘĆ NIEAKTUALNA"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
