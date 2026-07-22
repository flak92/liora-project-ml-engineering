#!/usr/bin/env python3
"""The execution ledger — a technical audit trail, never a source of scientific truth.

It records what the machine did: which worker ran which task, when, with what exit code, how much CPU
it took, and the hash of the artifact it published. If it disagrees with a result artifact, the
artifact wins — the ledger is evidence of execution, not of a verdict. It is the hash-chained,
`flock`-atomic append-only log from the science tree, pointed at a run-scoped file.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ledger import Ledger                                                  # noqa: E402


class ExecLedger:
    def __init__(self, run_dir):
        self._led = Ledger(Path(run_dir) / "exec_ledger.jsonl")

    def running(self, task, worker):
        self._led.append("exec", _unit(task), "running", note=worker)

    def done(self, task, worker, seconds, exit_code, artifact_sha, retries=0):
        self._led.append("exec", _unit(task), "completed", note=worker, payload={
            "worker": worker, "seconds": round(seconds, 2), "exit_code": int(exit_code),
            "artifact_sha256": artifact_sha, "retries": int(retries)})

    def failed(self, task, worker, seconds, exit_code, error, retries=0):
        self._led.append("exec", _unit(task), "failed", note=worker, payload={
            "worker": worker, "seconds": round(seconds, 2), "exit_code": int(exit_code),
            "error": str(error)[:400], "retries": int(retries)})

    def reconcile_orphans(self):
        return self._led.reconcile_orphans("exec")

    def verify(self):
        return self._led.verify_chain()

    def read_all(self):
        return self._led.read_all()


def _unit(task):
    return {k: task[k] for k in ("asset", "rung", "task_hash") if k in task}
