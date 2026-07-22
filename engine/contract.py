#!/usr/bin/env python3
"""Rung 0 — freeze the contract for a run. The engine snapshots the existing scientific contract; it
does NOT invent a second contract system.

The rules of the method live in `config/feature_discovery_contract.json` (assembled from
`config/contract/*`). At the start of a run this writes an immutable snapshot — the assembled
contract plus the hashes and environment that pin it — into `runs/<run_id>/contract.json`. Every task
carries the snapshot's `contract_hash`; a worker refuses to run a task whose hash does not match, so
a run can never mix results computed under two different sets of rules. Changing the rules means a
new snapshot and a new run_id, by hand — never a side effect of the engine.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init                                                        # noqa: E402
import contract_loader as CL                                              # noqa: E402
from artifact_io import sha256_of, write_json_atomic                       # noqa: E402

SAMPLE = ROOT / "config" / "sample_20.json"
BARS = ROOT / "xgb" / "data" / "liora.duckdb"


def _git(*a, default=""):
    try:
        return subprocess.run(("git", "-C", str(ROOT)) + a, capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return default


def contract_fingerprint():
    """A hash of the assembled scientific contract as it stands right now."""
    import hashlib
    assembled = CL.assemble()
    payload = json.dumps(assembled, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot(run_dir, assets, seed=42, allow_dirty=False):
    """Write runs/<id>/contract.json. Refuses a dirty tree unless development is explicitly allowed —
    a snapshot that cannot be tied to a clean commit is not reproducible from its SHA."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("drzewo brudne — snapshot kontraktu niereprodukowalny; "
                           "użyj allow_dirty tylko w trybie development")
    assembled = CL.assemble()
    snap = {
        "run_id": run_dir.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_hash": contract_fingerprint(),
        "execution_head_sha": _git("rev-parse", "HEAD"),
        "code_dirty": dirty,
        "bar_store_sha256_prefix": sha256_of(BARS)[:16] if BARS.exists() else None,
        "sample_sha256_prefix": sha256_of(SAMPLE)[:16] if SAMPLE.exists() else None,
        "environment": runtime_init.env_report(),
        "data_boundary": assembled.get("data_boundary"),
        "assets": list(assets),
        "seed": int(seed),
        "contract": assembled,
        "_rule": "every task carries contract_hash; a worker refuses a task whose hash differs. "
                 "New rules = new snapshot + new run_id, by hand.",
    }
    write_json_atomic(run_dir / "contract.json", snap)
    return snap


def load(run_dir):
    return json.loads((Path(run_dir) / "contract.json").read_text(encoding="utf-8"))


def matches_current(run_dir):
    """Does the run's frozen contract still equal the current assembled contract? Enforcement hook —
    if a human edited the contract mid-run, this is False and the run must not continue."""
    return load(run_dir)["contract_hash"] == contract_fingerprint()
