#!/usr/bin/env python3
"""Execute one task: run the science runner it names, publish an immutable result, record what happened.
The executor knows nothing about rung transitions (the planner's job) or work assignment (the driver's).

    contract gate -> dispatch runner (in a workspace) -> validate envelope -> idempotent/integrity
                     publish -> execution ledger

The executor's only scientific responsibility is a negative one: it refuses to run a task whose
`contract_hash` does not match the run's frozen contract, so no result is ever produced under rules
different from the ones the run declared. Everything it writes is immutable and self-describing — a
result file names the task, the contract, the runner exit code and its own content hash — so the
reducer and the report can trust it without re-deriving anything. `engine/asset_driver.py` calls
`execute_task` once per (asset, rung) in the asset's private workspace.
"""
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
from states import RUNG_DIR                                                # noqa: E402


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
    """The queueless execution core: contract gate -> dispatch (in `ws`) -> validate ->
    idempotent/integrity publish -> ledger. The caller owns work assignment and liveness (no claim,
    no heartbeat). Returns {**task, outcome, ...}; outcome in {contract_mismatch, failed, invalid,
    failed_integrity, published, idempotent}. The one scientific gate the executor owns is the
    contract_hash check."""
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
