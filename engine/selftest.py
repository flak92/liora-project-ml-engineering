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


def _mk_run(d, run_id="r", assets=("A",)):
    """A minimal contract snapshot so read_artifact/derive_state can resolve thresholds and hashes."""
    (Path(d) / "contract.json").write_text(json.dumps({
        "run_id": run_id, "contract_hash": "GOOD", "assets": list(assets), "seed": 42,
        "contract": {"viability": {"min_split_nodes": 20, "min_pred_std": 0.005},
                     "acceptance": {"complexity_penalty": 0.004}}}))


def _mk_artifact(d, run_id, rung, asset, result):
    """Write a per-asset artifact at the DETERMINISTIC task_hash path derived from the unit."""
    th = SC.task_hash({"run_id": run_id, "asset": asset, "rung": rung})
    p = Path(d) / "results" / ST.RUNG_DIR[rung] / asset / f"{th}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"task": {"asset": asset, "rung": rung}, "result": result,
                             "produced_utc": "t", "runner_exit_code": 0, "contract_hash": "GOOD",
                             "result_sha256": "x"}))
    return th


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
    _mk_run(d)
    q = Queue(d)
    q.enqueue(SC.make_task("r", "A", 1, "WRONG", 42))       # mismatched hash
    import worker as WK
    ST._CONTRACT.clear()
    r = WK.run_one(str(d), "worker-01")
    check("zadanie z błędnym hashem odrzucone", r and r["outcome"] == "contract_mismatch", str(r and r.get("outcome")))
    check("żaden artefakt nie powstał", ST.read_artifact(str(d), 1, "A") is None)
    check("zadanie w failed/", q.counts()["failed"] == 1)
    shutil.rmtree(d, ignore_errors=True)


def test_idempotent_publish():
    """One deterministic path per unit: identical retry is a no-op success, a divergent result is
    FAILED_INTEGRITY (not silently overwritten)."""
    print("\n3. idempotentny retry: identyczny = no-op, rozbieżny = FAILED_INTEGRITY")
    import worker as WK
    d = Path(tempfile.mkdtemp())
    _mk_run(d)
    ST._CONTRACT.clear()
    task = SC.make_task("r", "A", 1, "GOOD", 42)
    env = SC.wrap_result(task, {"spaces": {"v2_hessian_relative": [{"split_nodes": 72, "pred_std": 0.02}]}}, 0, "t")
    sha1, p1 = WK._publish(str(d), task, env)
    sha2, p2 = WK._publish(str(d), task, env)                # identical retry
    check("pierwsza publikacja", p1 == "published")
    check("identyczny retry to no-op (idempotent)", p2 == "idempotent" and sha1 == sha2)
    env2 = SC.wrap_result(task, {"spaces": {"v2_hessian_relative": [{"split_nodes": 99, "pred_std": 0.9}]}}, 0, "t")
    _sha3, p3 = WK._publish(str(d), task, env2)              # divergent result, same unit
    check("rozbieżny wynik -> integrity_mismatch (bez nadpisania)", p3 == "integrity_mismatch")
    check("jeden plik na jednostkę (deterministyczna ścieżka)",
          len(list((d / "results" / ST.RUNG_DIR[1] / "A").glob("*.json"))) == 1)
    shutil.rmtree(d, ignore_errors=True)


def test_state_from_artifacts_not_ledger():
    """State is a function of artifacts + contract, reconstructible with no ledger present at all."""
    print("\n4. stan wyprowadzany z artefaktów, nie z ledgera")
    d = Path(tempfile.mkdtemp())
    _mk_run(d)
    ST._CONTRACT.clear()
    _mk_artifact(d, "r", 1, "A", {"spaces": {"v2_hessian_relative": [{"split_nodes": 0, "pred_std": 0.0}] * 3}})
    st = ST.derive_state(str(d), "A")
    check("nie-viable Rung 1 -> NEEDS_CONTRACT (bez ledgera)", st["state"] == "NEEDS_CONTRACT", st["state"])
    check("NEEDS_CONTRACT oddzielone od błędu technicznego", st.get("required_human_action") is not None)
    shutil.rmtree(d, ignore_errors=True)


def test_viability_both_gates():
    """The viability floor is BOTH split_nodes AND pred_std, read from the contract — not the engine."""
    print("\n4b. viability: obie bramki z kontraktu (split_nodes I pred_std)")
    d = Path(tempfile.mkdtemp())
    _mk_run(d)
    ST._CONTRACT.clear()
    # splits fine, but prediction spread below the floor -> a constant dressed up -> NOT viable.
    _mk_artifact(d, "r", 1, "A", {"spaces": {"v2_hessian_relative": [{"split_nodes": 72, "pred_std": 0.001}] * 3}})
    check("splits OK ale pred_std < floor -> NEEDS_CONTRACT",
          ST.derive_state(str(d), "A")["state"] == "NEEDS_CONTRACT")
    check("brak zaszytych progów w engine/states.py",
          "= 20" not in (ROOT / "engine" / "states.py").read_text() and
          "0.005" not in (ROOT / "engine" / "states.py").read_text())
    shutil.rmtree(d, ignore_errors=True)


def test_rung5_stable_survivor():
    """NULL_VALIDATED requires a stable survivor across A1 ∩ A2 ∩ B — A1 alone is not enough, and the
    a1/a2/b result shape (not `folds` at the top level) is read correctly."""
    print("\n4c. werdykt Rung 5: kanoniczny A1∩A2∩B, nie samo A1")
    import rung5_verdict as RV
    passed = {"folds": [{"outer_fold": 0, "arms": {"hierarchical": {"verdict": "passed", "unit": "macd"}}}]}
    rej = {"folds": [{"outer_fold": 0, "arms": {"hierarchical": {"verdict": "rejected_early", "unit": "macd"}}}]}
    check("A1 przeszedł, ale A2 odrzucił -> NIE null_validated",
          not RV.null_validated({"a1": passed, "a2": rej, "b": passed}))
    check("A1∩A2∩B wszystkie przeszły -> null_validated",
          RV.null_validated({"a1": passed, "a2": passed, "b": passed}))
    check("brak a2/b (tylko a1) -> NIE stabilny (bez korzyści wątpliwości)",
          not RV.null_validated({"a1": passed, "a2": None, "b": None}))
    # and states._null_validated reads {a1,a2,b}, not art["result"]["folds"] (the old bug)
    d = Path(tempfile.mkdtemp()); _mk_run(d); ST._CONTRACT.clear()
    _mk_artifact(d, "r", 5, "A", {"a1": passed, "a2": passed, "b": passed})
    art = ST.read_artifact(str(d), 5, "A")
    check("states._null_validated czyta strukturę a1/a2/b", ST._null_validated(art))
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
    for t in (test_queue_atomicity, test_contract_enforcement, test_idempotent_publish,
              test_state_from_artifacts_not_ledger, test_viability_both_gates,
              test_rung5_stable_survivor, test_ledger_separation, test_oos_purge_present):
        try:
            t()
        except Exception as e:                                      # noqa: BLE001
            check(f"{t.__name__} rzucił wyjątek", False, f"{type(e).__name__}: {e}")
    print(f"\n{'WSZYSTKO ZDANE' if not FAILS else 'NIEZDANE: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
