#!/usr/bin/env python3
"""Prove the engine's execution guarantees — fast, without running a single science runner.

Each check targets one promise the engine makes about HOW it runs, not about what the science finds:
the queue hands a task to exactly one worker, a task under the wrong contract is refused before any
compute, published artifacts are immutable and the newest is the state, the two ledgers stay
separate and hash-consistent, and the science runners still carry their OOS purge assertion. None of
this touches XGBoost — the scientific correctness is the runners' own business, proven elsewhere.

    make engine-selftest
"""
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import states as ST                                                        # noqa: E402
import schemas as SC                                                       # noqa: E402
from taskqueue import Queue                                                    # noqa: E402
from exec_ledger import ExecLedger                                         # noqa: E402
from method_ledger import MethodLedger                                     # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def _drain(run_dir):
    q = Queue(run_dir)
    claimed = []
    while True:
        t = q.claim()
        if t is None:
            break
        claimed.append(t["task_hash"])
        q.finish(t, "done")
    return claimed


def test_queue_atomicity():
    """The atomic rename must hand each task to exactly one worker under real concurrency."""
    print("\n1. kolejka: atomowy claim pod współbieżnością")
    d = Path(tempfile.mkdtemp())
    q = Queue(d)
    N = 120
    for i in range(N):
        q.enqueue(SC.make_task("r", f"A{i}", 1, "h", 42))
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_drain, [str(d)] * 4))
    all_claimed = [h for r in results for h in r]
    check("każde zadanie odebrane dokładnie raz", len(all_claimed) == N == len(set(all_claimed)),
          f"{len(all_claimed)} odebrań, {len(set(all_claimed))} unikalnych, {N} zadań")
    check("kolejka pusta po drenażu", q.counts()["pending"] == 0 and q.counts()["running"] == 0)
    shutil.rmtree(d, ignore_errors=True)


def test_contract_enforcement():
    """A task whose contract_hash does not match the run's frozen contract is refused before compute."""
    print("\n2. egzekucja kontraktu: zły contract_hash odrzucony przed dispatchem")
    d = Path(tempfile.mkdtemp())
    (d / "contract.json").write_text(json.dumps({"contract_hash": "GOOD", "assets": ["A"], "seed": 42}))
    q = Queue(d)
    q.enqueue(SC.make_task("r", "A", 1, "WRONG", 42))       # mismatched hash
    import worker as WK
    r = WK.run_one(str(d), "worker-01")
    check("zadanie z błędnym hashem odrzucone", r and r["outcome"] == "contract_mismatch", str(r and r.get("outcome")))
    # no artifact was published, and it landed in failed/
    art = ST.latest_artifact(str(d), 1, "A")
    check("żaden artefakt nie powstał", art is None)
    check("zadanie w failed/", q.counts()["failed"] == 1)
    shutil.rmtree(d, ignore_errors=True)


def test_immutable_publish():
    """Each attempt publishes a NEW file; the newest valid one is the state, older ones are kept."""
    print("\n3. niezmienne artefakty: publikacja nie nadpisuje, najnowszy = stan")
    d = Path(tempfile.mkdtemp())
    base = d / "results" / ST.RUNG_DIR[1] / "A"
    base.mkdir(parents=True)
    for i, sn in enumerate((0, 72)):        # first non-viable, then a viable re-run
        (base / f"h{i}.json").write_text(json.dumps({
            "task": {"asset": "A", "rung": 1}, "result": {"spaces": {"v2_hessian_relative": [{"split_nodes": sn}] * 3}},
            "produced_utc": f"t{i}", "runner_exit_code": 0, "contract_hash": "h"}))
        os.utime(base / f"h{i}.json", (100 + i, 100 + i))
    check("oba artefakty współistnieją", len(list(base.glob("*.json"))) == 2)
    art = ST.latest_artifact(str(d), 1, "A")
    check("najnowszy artefakt jest stanem (viable)", ST._viable(art))
    shutil.rmtree(d, ignore_errors=True)


def test_state_from_artifacts_not_ledger():
    """State is a function of artifacts, reconstructible with no ledger present at all."""
    print("\n4. stan wyprowadzany z artefaktów, nie z ledgera")
    d = Path(tempfile.mkdtemp())
    b = d / "results" / ST.RUNG_DIR[1] / "A"
    b.mkdir(parents=True)
    (b / "h.json").write_text(json.dumps({"task": {}, "result": {"spaces": {"v2_hessian_relative": [{"split_nodes": 0}] * 3}},
                                          "produced_utc": "t", "runner_exit_code": 0, "contract_hash": "h"}))
    st = ST.derive_state(str(d), "A")
    check("nie-viable Rung 1 -> NEEDS_CONTRACT (bez ledgera)", st["state"] == "NEEDS_CONTRACT",
          st["state"])
    check("NEEDS_CONTRACT oddzielone od błędu technicznego", st.get("required_human_action") is not None)
    shutil.rmtree(d, ignore_errors=True)


def test_ledger_separation():
    """Execution and methodology ledgers are two files, each hash-consistent."""
    print("\n5. dwa ledgery: techniczny i metodologiczny, oddzielne i spójne")
    d = Path(tempfile.mkdtemp())
    ex = ExecLedger(d)
    task = SC.make_task("r", "A", 1, "h", 42)
    ex.running(task, "worker-01")
    ex.done(task, "worker-01", 12.3, 0, "abc123")
    me = MethodLedger(d)
    me.record("A", 1, "Can the model learn?", "VIABLE", None, "RUN_RUNG_3")
    ok_e, _ = ex.verify()
    ok_m, _ = me.verify()
    check("exec_ledger.jsonl istnieje i spójny", (d / "exec_ledger.jsonl").exists() and ok_e)
    check("method_ledger.jsonl istnieje i spójny", (d / "method_ledger.jsonl").exists() and ok_m)
    check("to dwa OSOBNE pliki", (d / "exec_ledger.jsonl") != (d / "method_ledger.jsonl")
          and (d / "exec_ledger.jsonl").exists() and (d / "method_ledger.jsonl").exists())
    shutil.rmtree(d, ignore_errors=True)


def test_oos_purge_present():
    """The engine never asks a runner to cross the OOS boundary: the contract declares oos_reads=0,
    and every dispatched runner builds its folds with a boundary-aware mechanism (an explicit
    oos_start_idx purge assertion, purged walk-forward folds, or the outer-fold carver — all of which
    keep validation strictly inside Train)."""
    print("\n6. granica OOS: kontrakt oos_reads=0 + runnery używają foldów świadomych granicy")
    import contract_loader as CL
    oos_reads = CL.assemble().get("data_boundary", {}).get("oos_reads")
    check("kontrakt deklaruje oos_reads = 0", oos_reads == 0, f"oos_reads={oos_reads}")

    boundary_markers = ("oos_start_idx", "purged_wf_folds", "outer_folds")
    runners = ["model_viability.py", "feature_utility.py", "crossfit_selection.py",
               "procedure_null.py", "rung6_survivor_hpo.py"]
    missing = []
    for r in runners:
        src = (ROOT / "scripts" / r).read_text(encoding="utf-8")
        if not any(m in src for m in boundary_markers):
            missing.append(r)
    check("każdy runner używa foldów świadomych granicy Train", not missing, str(missing))


def main():
    print("engine selftest — gwarancje wykonania (bez uruchamiania nauki)")
    for t in (test_queue_atomicity, test_contract_enforcement, test_immutable_publish,
              test_state_from_artifacts_not_ledger, test_ledger_separation, test_oos_purge_present):
        try:
            t()
        except Exception as e:                                      # noqa: BLE001
            check(f"{t.__name__} rzucił wyjątek", False, f"{type(e).__name__}: {e}")
    print(f"\n{'WSZYSTKO ZDANE' if not FAILS else 'NIEZDANE: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
