#!/usr/bin/env python3
"""verify_global_integrity — the one pass the iteration loop runs at the top of every cycle.

The pieces already existed, scattered: the worker guards per-artifact reproducibility, the ledgers
hash-chain themselves, the contract pins its own hash, and the selftest greps runners for a
boundary-aware fold mechanism. This composes them into a single callable the orchestrator can gate
on: if any check fails, the loop refuses to plan another experiment on top of a corrupted run rather
than pouring more compute onto a foundation it cannot trust.

It is deliberately read-only and total: every check is wrapped so a single failure is reported, not
raised, and the loop sees the whole picture. Nothing here decides science — it only asserts that what
is already on disk is internally consistent and stayed inside the frozen boundary.

    ok, report = integrity.verify(run_dir)      # report["checks"] = per-check {name, ok, detail}
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract as CT                                                       # noqa: E402
import contract_patch as CP                                                 # noqa: E402
import schemas as SC                                                        # noqa: E402
from states import RUNG_DIR                                                 # noqa: E402
from exec_ledger import ExecLedger                                          # noqa: E402
from method_ledger import MethodLedger                                      # noqa: E402

BOUNDARY_MARKERS = ("oos_start_idx", "purged_wf_folds", "outer_folds")
RUNNERS = ("model_viability.py", "feature_utility.py", "crossfit_selection.py",
           "procedure_null.py", "rung6_survivor_hpo.py")


def _check(name, fn):
    """Run one predicate, never let it raise. Returns {name, ok, detail}."""
    try:
        ok, detail = fn()
        return {"name": name, "ok": bool(ok), "detail": detail}
    except Exception as e:                                                   # noqa: BLE001
        return {"name": name, "ok": False, "detail": f"{type(e).__name__}: {e}"}


def _contract_self_consistent(run_dir):
    snap = CT.load(run_dir)
    ok = CP.self_consistent(snap)
    return ok, ("hash == sha256(embedded)" if ok else "hash snapshotu ≠ hash osadzonego kontraktu")


def _oos_reads_zero(run_dir):
    """Read the run's OWN frozen boundary, not the current on-disk config — the run is judged by the
    rules it froze."""
    ob = CT.load(run_dir).get("contract", {}).get("data_boundary", {}).get("oos_reads")
    return ob == 0, f"oos_reads={ob}"


def _runner_boundary_markers(_run_dir):
    missing = [r for r in RUNNERS
               if not any(m in (ROOT / "scripts" / r).read_text(encoding="utf-8")
                          for m in BOUNDARY_MARKERS)]
    return (not missing), ("wszystkie runnery świadome granicy" if not missing else f"brak: {missing}")


def _ledger_chain(run_dir, which):
    led = ExecLedger(run_dir) if which == "exec" else MethodLedger(run_dir)
    ok, bad = led.verify()
    return ok, ("łańcuch spójny" if ok else f"pierwsza zła linia: {bad}")


def _artifact_schema_sweep(run_dir):
    """Every published per-asset artifact is a well-formed result envelope. A truncated or half-written
    file must never be mistaken for a verdict; the reducer/report trust this sweep."""
    results = Path(run_dir) / "results"
    bad, n = [], 0
    for rung_dir in RUNG_DIR.values():
        d = results / rung_dir
        if not d.is_dir():
            continue
        for art in d.glob("*/*.json"):
            n += 1
            try:
                problems = SC.validate_result(json.loads(art.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as e:
                problems = [f"nieczytelny: {e}"]
            if problems:
                bad.append(f"{art.relative_to(results)}: {problems}")
    return (not bad), (f"{n} artefaktów, wszystkie poprawne" if not bad
                       else f"{len(bad)}/{n} wadliwych: {bad[:3]}")


def verify(run_dir):
    """Return (ok, report). report['checks'] is a list of {name, ok, detail}; ok is their conjunction.
    Read-only and total — no check can crash the pass, and a corrupted run surfaces every problem."""
    run_dir = str(run_dir)
    checks = [
        _check("contract_self_consistent", lambda: _contract_self_consistent(run_dir)),
        _check("oos_reads_zero", lambda: _oos_reads_zero(run_dir)),
        _check("runner_boundary_markers", lambda: _runner_boundary_markers(run_dir)),
        _check("exec_ledger_chain", lambda: _ledger_chain(run_dir, "exec")),
        _check("method_ledger_chain", lambda: _ledger_chain(run_dir, "method")),
        _check("artifact_schema_sweep", lambda: _artifact_schema_sweep(run_dir)),
    ]
    ok = all(c["ok"] for c in checks)
    return ok, {"run_dir": run_dir, "ok": ok, "checks": checks,
                "problems": [c["name"] for c in checks if not c["ok"]]}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Global integrity verify for one run.")
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    ok, report = verify(args.run_dir)
    for c in report["checks"]:
        print(f"  [{'OK ' if c['ok'] else 'FAIL'}] {c['name']:<26} {c['detail']}")
    print(f"\nintegralność: {'ZIELONA' if ok else 'CZERWONA — ' + str(report['problems'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
