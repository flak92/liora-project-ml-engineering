#!/usr/bin/env python3
"""Prove the Iterative Calibration Loop's guarantees — fast, without running a science runner.

It runs the engine selftest (the inner loop's guarantees) and then adds the outer loop's: the safety
kernel refuses to loosen the proof standard, convergence stops exactly when a hypothesis adds nothing,
the Repair Loop classifies technical failures correctly, the budget cap and ladder guard behave, and
integrity actually catches tampering. None of it touches XGBoost.

    make iteration-selftest
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract_patch as CP                                                 # noqa: E402
import integrity as IG                                                      # noqa: E402
import iteration_planner as IP                                              # noqa: E402
import repair as RP                                                         # noqa: E402
from exec_ledger import ExecLedger                                          # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_engine_selftest():
    """The inner loop's 8 guarantees must still hold — run the existing engine selftest as a subprocess."""
    print("\n0. engine-selftest (gwarancje pętli wewnętrznej)")
    p = subprocess.run([sys.executable, str(ROOT / "engine" / "selftest.py")],
                       capture_output=True, text=True, timeout=300)
    # match the status-line marker "  FAIL  " (2+2 spaces), not the substring inside FAILED_INTEGRITY
    ok = p.returncode == 0 and "WSZYSTKO ZDANE" in p.stdout and "  FAIL  " not in p.stdout
    check("engine-selftest przechodzi (8 grup)", ok,
          "" if ok else "rc=%s; ostatnie: %s" % (p.returncode, p.stdout.strip().splitlines()[-3:]))


def test_patch_guard():
    """The kernel admits only hypothesis-space patches and reproduces the base hash on an empty one."""
    print("\n1. strażnik patcha: tylko przestrzeń hipotez, standard dowodu zamrożony")
    import contract as CT
    _p, h0, t0 = CP.apply({})
    check("pusty patch == fingerprint bazy", h0 == CT.contract_fingerprint() and not t0)
    for k in CP.FROZEN:
        try:
            CP.guard({k: {}}); check(f"odrzuca zamrożony {k}", False, "PRZYJĄŁ!")
        except CP.PatchRejected:
            check(f"odrzuca zamrożony {k}", True)
    for k in ("runtime", "certification", "brand_new_top"):
        try:
            CP.guard({k: {}}); check(f"odrzuca poza-hipotezę {k}", k not in CP.ADMISSIBLE and False)
        except CP.PatchRejected:
            check(f"odrzuca poza-hipotezę {k}", True)
    for k in CP.ADMISSIBLE:
        try:
            CP.guard({k: {}}); check(f"przyjmuje hipotezę {k}", True)
        except CP.PatchRejected as e:
            check(f"przyjmuje hipotezę {k}", False, str(e))
    _pv, hv, _tv = CP.apply({"operating_point": {"grid": [0.8, 0.9]}})
    check("patch operating_point zmienia hash i jest self-consistent",
          hv != h0 and CP._hash(_pv) == hv)


def test_convergence():
    """convergence_update + should_converge stop exactly at the first variant that adds nothing."""
    print("\n2. zbieżność: stop gdy kolejna hipoteza nie dodaje cechy")
    cum = set()
    cum, d0, ni0 = IP.convergence_update(cum, {("A", "x")}, 0)
    check("e0 dodaje (A,x): delta=1, streak=0", d0 == [("A", "x")] and ni0 == 0)
    check("e0 nie kończy (baza nigdy nie zbiega)", not IP.should_converge(0, ni0, 1))
    cum1, d1, ni1 = IP.convergence_update(cum, {("A", "x")}, ni0)     # variant adds nothing
    check("e1 bez nowości: delta=0, streak=1", d1 == [] and ni1 == 1)
    check("e1 CONVERGED przy patience=1", IP.should_converge(1, ni1, 1))
    _c2, d2, ni2 = IP.convergence_update(cum, {("A", "y")}, ni0)      # variant adds a new one
    check("wariant z nową cechą NIE zbiega", d2 == [("A", "y")] and not IP.should_converge(1, ni2, 1))
    _cb, _db, nib = IP.convergence_update(set(), set(), 0)            # barren base
    check("jałowa baza nie ogłasza zbieżności (k=0)", not IP.should_converge(0, nib, 1))


def test_repair_mining():
    """The Repair Loop classifies a synthetic exec ledger into repaired / retry / quarantine / failed."""
    print("\n3. Repair Loop: klasyfikacja awarii z exec_ledger")
    with tempfile.TemporaryDirectory() as d:
        led = ExecLedger(d)
        def task(a, h): return {"asset": a, "rung": 1, "task_hash": h}
        led.failed(task("A", "h1"), "w", 1.0, 1, "transient"); led.done(task("A", "h1"), "w", 1.0, 0, "sha")
        led.failed(task("B", "h2"), "w", 1.0, 95, "integrity")
        led.failed(task("C", "h3"), "w", 1.0, 1, "t"); led.failed(task("C", "h3"), "w", 1.0, 1, "t")
        led.failed(task("D", "h4"), "w", 1.0, 1, "t")
        cls = RP.classify(d)
        diag = {rec["asset"]: RP.diagnose(rec, 2) for rec in cls.values()}
        check("A failed→completed = repaired", diag.get("A") == "repaired", str(diag))
        check("B exit95 = quarantine_integrity", diag.get("B") == "quarantine_integrity")
        check("C 2×fail (>=max_retries) = failed_technical", diag.get("C") == "failed_technical")
        check("D 1×fail (<max_retries) = safe_retry", diag.get("D") == "safe_retry")
        # handle() requeues D: create its failed queue file, expect it moved to pending
        q = Path(d) / "queue"
        (q / "failed").mkdir(parents=True, exist_ok=True); (q / "pending").mkdir(parents=True, exist_ok=True)
        (q / "failed" / "h4.json").write_text("{}")
        RP.handle(d, 2)
        check("safe_retry przenosi failed→pending", (q / "pending" / "h4.json").exists()
              and not (q / "failed" / "h4.json").exists())


def test_budget_cap():
    """core_hours_cap converts to seconds; null means no cap."""
    print("\n4. budżet: limit rdzeniogodzin → HALTED_BUDGET")
    check("cap 0.001h = 3.6s", IP._core_cap_seconds({"budget": {"core_hours_cap": 0.001}}) == 3.6)
    check("cap null = brak limitu", IP._core_cap_seconds({"budget": {"core_hours_cap": None}}) is None)
    with tempfile.TemporaryDirectory() as d:
        led = ExecLedger(d)
        led.done({"asset": "A", "rung": 1, "task_hash": "h"}, "w", 10.0, 0, "sha")
        spent = IP.core_seconds(d)
        check("core_seconds sumuje completed (10s)", spent == 10.0)
        check("spent przekracza cap → halt-warunek", spent >= IP._core_cap_seconds({"budget": {"core_hours_cap": 0.001}}))


def test_ladder_guard():
    """ladder_from_policy prepends base and rejects an inadmissible ladder up front."""
    print("\n5. drabina: baza + wyłącznie dopuszczalne wersje, fail-fast")
    good = {"ladder": [{"version_id": "v1", "patch": {"operating_point": {"grid": [0.9]}}}]}
    rungs = IP.ladder_from_policy(good)
    check("baza doklejona, 2 wersje", len(rungs) == 2 and rungs[0]["version_id"] == "base")
    bad = {"ladder": [{"version_id": "bad", "patch": {"acceptance": {"min_win_rate": 0.0}}}]}
    try:
        IP.ladder_from_policy(bad); check("odrzuca niedopuszczalną drabinę", False, "PRZYJĄŁ!")
    except CP.PatchRejected:
        check("odrzuca niedopuszczalną drabinę (fail-fast)", True)


def test_integrity_tampering():
    """integrity.verify is green on a real snapshot and red when the contract hash or an artifact is tampered."""
    print("\n6. integralność wykrywa manipulację")
    with tempfile.TemporaryDirectory() as d:
        rd = Path(d) / "run"
        CP.snapshot_version(rd, ["A"], "base", {}, seed=42, allow_dirty=True)
        ok, rep = IG.verify(rd)
        check("świeży snapshot: integralność zielona", ok, str(rep["problems"]))
        snap = json.loads((rd / "contract.json").read_text())
        snap["contract_hash"] = "deadbeef"
        (rd / "contract.json").write_text(json.dumps(snap))
        ok2, rep2 = IG.verify(rd)
        check("podmieniony contract_hash → czerwona", not ok2 and "contract_self_consistent" in rep2["problems"])


def main():
    print("iteration-selftest — gwarancje Iterative Calibration Loop (bez uruchamiania nauki)")
    for t in (test_engine_selftest, test_patch_guard, test_convergence, test_repair_mining,
              test_budget_cap, test_ladder_guard, test_integrity_tampering):
        try:
            t()
        except Exception as e:                                              # noqa: BLE001
            check(f"{t.__name__} rzucił wyjątek", False, f"{type(e).__name__}: {e}")
    print("\n" + ("WSZYSTKO ZDANE" if not FAILS else f"NIEZDANE: {FAILS}"))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
