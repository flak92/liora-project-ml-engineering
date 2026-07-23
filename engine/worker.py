#!/usr/bin/env python3
"""One worker step: claim a task, run the science runner it names, publish an immutable result, and
record what happened. The worker knows nothing about rung transitions — that is the planner's job.

    claim -> verify contract hash -> dispatch runner -> validate envelope -> atomic publish
          -> execution ledger -> move task to done/failed

The worker's only scientific responsibility is a negative one: it refuses to run a task whose
`contract_hash` does not match the run's frozen contract, so no result is ever produced under rules
different from the ones the run declared. Everything it writes is immutable and self-describing — a
result file names the task, the contract, the runner exit code and its own content hash — so the
reducer and the report can trust it without re-deriving anything.

    python3 engine/worker.py --run-dir runs/<id> --worker worker-01        # one step
    python3 engine/worker.py --run-dir runs/<id> --worker worker-01 --loop # until the queue drains
"""
import argparse
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_SEC = 60          # how often a running task refreshes its own liveness (< guard STALE_TASK)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "engine"))
import runtime_init                                                        # noqa: E402,F401
runtime_init.apply()
from artifact_io import write_json_atomic                                  # noqa: E402
import contract as CT                                                      # noqa: E402
import dispatch as DP                                                      # noqa: E402
import reducer as RD                                                       # noqa: E402
import schemas as SC                                                       # noqa: E402
from taskqueue import Queue                                                # noqa: E402
from states import RUNG_DIR                                                # noqa: E402
from exec_ledger import ExecLedger                                         # noqa: E402


def _publish(run_dir, task, envelope):
    """Publish the per-asset result idempotently, addressed by the DETERMINISTIC task_hash.

    task_hash identifies the logical scientific unit (run, asset, rung, unit) — it does NOT carry an
    attempt number, so a retry addresses the same path. That is deliberate: a retry of a
    deterministic experiment must reproduce the same bytes.

      absent            -> publish
      present, same sha -> success, no overwrite (an idempotent retry)
      present, diff sha -> FAILED_INTEGRITY (the experiment was not reproducible — do not overwrite)

    This is a stronger determinism guarantee than keeping many attempts and picking the newest.
    """
    import json as _json
    path = (Path(run_dir) / "results" / RUNG_DIR[task["rung"]] / task["asset"]
            / f"{task['task_hash']}.json")
    new_sha = envelope["result_sha256"]
    if path.exists():
        try:
            existing = _json.loads(path.read_text(encoding="utf-8"))
            if existing.get("result_sha256") == new_sha:
                return new_sha, "idempotent"
            return existing.get("result_sha256"), "integrity_mismatch"
        except (_json.JSONDecodeError, OSError):
            pass                                    # a corrupt prior file: overwrite it
    write_json_atomic(path, envelope)
    return new_sha, "published"


def execute_task(run_dir, task, ws, worker_id, led):
    """The queueless core: contract gate -> dispatch (in `ws`) -> validate -> idempotent/integrity
    publish -> ledger. The caller owns work assignment and liveness (no claim, no heartbeat). Shared by
    the legacy queue worker (run_one) and the asset_driver (parallel-over-assets). Returns
    {**task, outcome, ...}; outcome in {contract_mismatch, failed, invalid, failed_integrity, published,
    idempotent}. The one scientific gate the executor owns is the contract_hash check."""
    snap = CT.load(run_dir)
    if task.get("contract_hash") != snap["contract_hash"]:
        led.failed(task, worker_id, 0.0, 7, "contract_hash mismatch")
        return {**task, "outcome": "contract_mismatch"}
    led.running(task, worker_id)
    t0 = time.time()
    result, rc, err = DP.dispatch(task, run_dir, ws=ws)
    secs = time.time() - t0
    if result is None or rc != 0:
        led.failed(task, worker_id, secs, rc, err)
        return {**task, "outcome": "failed", "exit_code": rc, "error": err}
    envelope = SC.wrap_result(task, result, rc,
                              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    problems = SC.validate_result(envelope)
    if problems:
        led.failed(task, worker_id, secs, 92, f"schema: {problems}")
        return {**task, "outcome": "invalid", "problems": problems}
    sha, pub = _publish(run_dir, task, envelope)
    if pub == "integrity_mismatch":
        # A re-run produced different bytes — the experiment was not reproducible. A scientific-integrity
        # failure, not transient; never overwritten, never retried.
        led.failed(task, worker_id, secs, 95, f"FAILED_INTEGRITY: sha {sha} != {envelope['result_sha256']}")
        return {**task, "outcome": "failed_integrity", "existing_sha": sha}
    led.done(task, worker_id, secs, rc, sha)
    return {**task, "outcome": pub, "seconds": round(secs, 1), "artifact_sha256": sha}


def run_one(run_dir, worker_id):
    """Legacy queue path: claim -> heartbeat -> execute_task -> finish. Returns task-with-outcome or None.

    The heartbeat keeps the running-file mtime fresh so the guard's stale sweep never requeues a
    legitimately long task into a concurrent duplicate; `os.utime` (not touch) never recreates a file the
    guard just moved. (The asset_driver needs neither queue nor heartbeat — nothing requeues a running
    task there.)"""
    q = Queue(run_dir)
    led = ExecLedger(run_dir)
    task = q.claim()
    if task is None:
        return None
    running_file = Path(run_dir) / "queue" / "running" / f"{task['task_hash']}.json"
    stop_hb = threading.Event()

    def _beat():
        while not stop_hb.wait(HEARTBEAT_SEC):
            try:
                os.utime(running_file, None)
            except OSError:
                return                              # file gone (finished/requeued) -> stop quietly

    hb = threading.Thread(target=_beat, daemon=True)
    hb.start()
    try:
        r = execute_task(run_dir, task, RD.workspace(run_dir), worker_id, led)
    finally:
        stop_hb.set()
        hb.join(timeout=2)
    q.finish(task, "done" if r["outcome"] in ("published", "idempotent") else "failed")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker", default="worker-00")
    ap.add_argument("--loop", action="store_true", help="claim until the queue is empty")
    ap.add_argument("--idle-exit", type=int, default=1, help="empty claims before a loop worker exits")
    args = ap.parse_args()

    if not args.loop:
        r = run_one(args.run_dir, args.worker)
        print("brak zadań" if r is None else f"{r['asset']}/rung{r['rung']}: {r['outcome']}")
        return 0

    idle = 0
    while idle < args.idle_exit:
        r = run_one(args.run_dir, args.worker)
        if r is None:
            idle += 1
            time.sleep(2)
            continue
        idle = 0
        print(f"[{args.worker}] {r['asset']}/rung{r['rung']}: {r['outcome']}"
              + (f" ({r.get('seconds')}s)" if r.get("seconds") else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
