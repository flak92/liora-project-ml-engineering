#!/usr/bin/env python3
"""The shapes that cross the boundary between the science layer and the execution layer.

A task, a published result, and a ledger record each have a small fixed shape. Keeping the shapes
here — with a validator the worker runs before publishing — is what lets the reducer, the planner
and the report trust what they read without re-deriving it. The validation is deliberately shallow:
required keys and types, not scientific plausibility. Whether a result is scientifically sound is the
artifact's own content, judged by the science layer; the schema only guarantees the envelope is
well-formed so a truncated or half-written file is never mistaken for a verdict.
"""
import hashlib
import json


def task_hash(task):
    """Stable identity of a task: what to run, on what, under which contract. Retries of the same
    logical unit share it up to the attempt counter, so the queue and ledger can pair them."""
    core = {k: task[k] for k in ("run_id", "asset", "rung", "unit") if k in task}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]


TASK_REQUIRED = {"run_id": str, "asset": str, "rung": int, "contract_hash": str, "seed": int}


def validate_task(task):
    problems = []
    for k, t in TASK_REQUIRED.items():
        if k not in task:
            problems.append(f"brak pola {k}")
        elif not isinstance(task[k], t):
            problems.append(f"{k}: typ {type(task[k]).__name__}, oczekiwano {t.__name__}")
    return problems


def make_task(run_id, asset, rung, contract_hash, seed, unit=None):
    t = {"run_id": run_id, "asset": asset, "rung": int(rung),
         "contract_hash": contract_hash, "seed": int(seed)}
    if unit is not None:
        t["unit"] = unit
    t["task_hash"] = task_hash(t)
    return t


RESULT_REQUIRED = {"task": dict, "result": dict, "produced_utc": str,
                   "runner_exit_code": int, "contract_hash": str}


def validate_result(doc):
    """The published-artifact envelope. `result` is the science payload (opaque here)."""
    problems = []
    for k, t in RESULT_REQUIRED.items():
        if k not in doc:
            problems.append(f"brak pola {k}")
        elif not isinstance(doc[k], t):
            problems.append(f"{k}: typ {type(doc[k]).__name__}, oczekiwano {t.__name__}")
    if doc.get("runner_exit_code", 0) != 0:
        problems.append(f"runner_exit_code={doc.get('runner_exit_code')} (niezerowy)")
    return problems


def wrap_result(task, result, exit_code, produced_utc):
    """Envelope a runner's per-asset output as an immutable, self-describing artifact.

    Invariant: a sealed `result` contains ONLY science; everything operational (timing, host, pid)
    belongs in the ledger and run_manifest, never here — so `result_sha256` is byte-identical across
    re-runs of the same task (the extractor drops `seconds` for exactly this reason)."""
    payload = json.dumps(result, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"task": task, "result": result, "runner_exit_code": int(exit_code),
            "contract_hash": task["contract_hash"], "produced_utc": produced_utc,
            "result_sha256": hashlib.sha256(payload).hexdigest()}
