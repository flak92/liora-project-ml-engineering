#!/usr/bin/env python3
"""The record of what actually ran — the ten fields the contract demands and no runner writes.

`config/feature_discovery_contract.json` carries a `run_manifest_required_fields` block, and
`identity.contract_commit_sha` is deliberately null with the explanation that the commit which
really executed is recorded here instead. That deferral only works if this file exists. Until now
it did not: no runner in `scripts/` captured a git SHA, a contract hash, a dirty-tree flag or an
artifact hash, and `runtime_init.env_report()` — written precisely to supply the `environment`
field — was never once called.

Two of the fields are the ones that make a result checkable rather than merely reported:

`code_dirty` says whether the working tree carried uncommitted changes. A dirty run is not
worthless, but it is not reproducible from the SHA alone and must never be presented as if it were.

`core_seconds` is measured, not estimated. It comes from `getrusage` over this process and every
child that has been reaped, so it counts what the CPUs actually did rather than wall-clock times a
guessed worker count. Wall and core seconds together are what reveal a run that was starved, or
one whose thread pools were not capped after all.

    man = RunManifest(run_dir / "run_manifest.json", stage="null_a1", workers=4)
    man.start()
    ...
    man.finish("COMPLETED", verdicts={...}, artifacts=[out_path],
               permutations_executed={"ADBE:2": 50, ...})
"""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init                                                        # noqa: E402
from artifact_io import sha256_of, write_json_atomic                       # noqa: E402

CONTRACT = ROOT / "config" / "feature_discovery_contract.json"


def _git(*args, default=""):
    try:
        return subprocess.run(("git", "-C", str(ROOT)) + args, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return default


def _core_seconds():
    """User + system time of this process and of every child already reaped.

    ProcessPoolExecutor reaps its workers on shutdown, so after the pool closes their time is
    included. Read before the pool is torn down, it undercounts — which is why finish() is called
    last.
    """
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return round(me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime, 2)


class RunManifest:
    def __init__(self, path, stage, workers, contract=CONTRACT):
        self.path = Path(path)
        self.stage = stage
        self.workers = int(workers)
        self.contract = Path(contract)
        self._t0 = None
        self._c0 = None
        self.data = {}

    def start(self, extra=None):
        self._t0 = time.time()
        self._c0 = _core_seconds()
        head = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
        self.data = {
            "stage": self.stage,
            "status": "RUNNING",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_head_sha": head,
            "contract_sha256": sha256_of(self.contract) if self.contract.exists() else None,
            "code_dirty": dirty,
            "_dirty_means": ("uncommitted changes were present, so this result is NOT reproducible "
                             "from execution_head_sha alone and must not be reported as if it were")
            if dirty else "clean tree",
            "environment": runtime_init.env_report(),
            "workers": self.workers,
            "wall_seconds": None,
            "core_seconds": None,
            "permutations_executed": {},
            "verdicts": {},
            "artifact_sha256": {},
        }
        if extra:
            self.data.update(extra)
        write_json_atomic(self.path, self.data)
        return self

    def finish(self, status, verdicts=None, artifacts=(), permutations_executed=None, extra=None):
        self.data["status"] = status
        self.data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.data["wall_seconds"] = round(time.time() - self._t0, 1) if self._t0 else None
        self.data["core_seconds"] = round(_core_seconds() - (self._c0 or 0), 2)
        # Re-read the pools now rather than trusting the reading taken at start(). threadpool_info
        # only sees libraries that are already loaded, and at start() nothing numeric has been
        # imported yet — so the field that is supposed to PROVE the pools were capped would have
        # been an empty list every single time. Measured after the work, it shows what the run
        # actually had: [{'blas','openblas',1}, ..., {'openmp','openmp',1}].
        self.data["environment"]["pools"] = runtime_init.thread_report()
        self.data["environment"]["_pools_measured"] = "after the numeric work, not at start"
        if verdicts is not None:
            self.data["verdicts"] = verdicts
        if permutations_executed is not None:
            self.data["permutations_executed"] = permutations_executed
        for a in artifacts:
            a = Path(a)
            if a.exists():
                self.data["artifact_sha256"][a.name] = sha256_of(a)
        if extra:
            self.data.update(extra)
        w = self.data["wall_seconds"] or 0
        self.data["parallel_efficiency"] = (
            round(self.data["core_seconds"] / (w * self.workers), 3)
            if w and self.workers else None)
        write_json_atomic(self.path, self.data)
        return self.data


def check_required_fields(manifest_path, contract=CONTRACT):
    """Every field the contract names must be present. Used by the chain's exit gate.

    Presence, not plausibility — a manifest that silently dropped `core_seconds` would otherwise
    look complete right up until someone tried to compare two runs' costs.
    """
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    req = json.loads(Path(contract).read_text(encoding="utf-8"))["run_manifest_required_fields"]
    missing = [k for k in req if k not in man]
    empty = [k for k in ("execution_head_sha", "contract_sha256", "wall_seconds", "core_seconds")
             if man.get(k) in (None, "", {})]
    return {"ok": not missing and not empty, "missing": missing, "empty": empty}
