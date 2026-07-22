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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "engine"))
import runtime_init                                                        # noqa: E402,F401
runtime_init.apply()
from artifact_io import write_json_atomic                                  # noqa: E402
import contract as CT                                                      # noqa: E402
import dispatch as DP                                                      # noqa: E402
import schemas as SC                                                       # noqa: E402
from taskqueue import Queue                                                    # noqa: E402
from states import RUNG_DIR                                                # noqa: E402
from exec_ledger import ExecLedger                                         # noqa: E402


def _publish(run_dir, task, envelope):
    """Write the immutable per-asset-per-task artifact. New task_hash each attempt, so a publish
    never overwrites an earlier result — history is preserved, the newest valid file is the state."""
    d = Path(run_dir) / "results" / RUNG_DIR[task["rung"]] / task["asset"]
    return write_json_atomic(d / f"{task['task_hash']}.json", envelope)


def run_one(run_dir, worker_id):
    """Claim and execute a single task. Returns the task (with outcome) or None if the queue is empty."""
    q = Queue(run_dir)
    led = ExecLedger(run_dir)
    task = q.claim()
    if task is None:
        return None

    # Contract enforcement — the one scientific gate the worker owns.
    snap = CT.load(run_dir)
    if task.get("contract_hash") != snap["contract_hash"]:
        led.failed(task, worker_id, 0.0, 7, "contract_hash mismatch")
        q.finish(task, "failed")
        return {**task, "outcome": "contract_mismatch"}

    led.running(task, worker_id)
    t0 = time.time()
    result, rc, err = DP.dispatch(task)
    secs = time.time() - t0

    if result is None or rc != 0:
        led.failed(task, worker_id, secs, rc, err)
        q.finish(task, "failed")
        return {**task, "outcome": "failed", "exit_code": rc, "error": err}

    envelope = SC.wrap_result(task, result, rc,
                              time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    problems = SC.validate_result(envelope)
    if problems:
        led.failed(task, worker_id, secs, 92, f"schema: {problems}")
        q.finish(task, "failed")
        return {**task, "outcome": "invalid", "problems": problems}

    sha = _publish(run_dir, task, envelope)
    led.done(task, worker_id, secs, rc, sha)
    q.finish(task, "done")
    return {**task, "outcome": "done", "seconds": round(secs, 1), "artifact_sha256": sha}


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
