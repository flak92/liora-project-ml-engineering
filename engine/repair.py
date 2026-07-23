#!/usr/bin/env python3
"""The Repair Loop — the third layer, and the one allowed to touch nothing scientific.

It reads the execution ledger and classifies technical failures: which are transient, and which a
retry cannot fix. Its whole vocabulary is execution-layer — worker crashes, timeouts, a
non-reproducible artifact — and it is forbidden, structurally, from reaching a threshold, a fold, a
null, or the OOS boundary. A `FAILED_TECHNICAL` unit is not a scientific verdict; it is a machine that
broke, kept apart from `NEEDS_CONTRACT` (a science stop) on purpose.

The diagnosis is a pure function of a unit's ledger history, so it is testable without a machine that
actually crashes:

    exit 95 (FAILED_INTEGRITY)      -> quarantine: a retry produced different bytes; never retry, never
                                       overwrite. The experiment was not reproducible — a finding.
    exit 7 / 92 (contract / schema) -> failed_technical: deterministic; a retry is futile.
    other non-zero, attempts < N    -> safe_retry: a transient (OOM, timeout) with budget left.
    other non-zero, attempts >= N   -> failed_technical: transient but exhausted.
    a later 'completed' for the unit -> repaired (an earlier failure already recovered).

The driver (engine/asset_driver.py) acts on this: a transient failure re-runs the asset's DAG once
(idempotent via immutable artifacts, so finished rungs are skipped), and `technical_terminal` gives
the orchestrator the units to overlay as FAILED_TECHNICAL. It never elevates a science stop and never
edits a result artifact.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
from exec_ledger import ExecLedger                                          # noqa: E402

EXIT_LABEL = {7: "contract_mismatch", 92: "schema_invalid", 95: "failed_integrity"}
NON_RETRYABLE = {7, 92, 95}                       # deterministic technical failures — retry won't help


def classify(run_dir):
    """Per-unit technical history from the exec ledger: {unit_key: {asset, rung, task_hash, attempts,
    failures, completed, last_status, last_exit, last_error}}. Pure read; no side effects."""
    units = {}
    for r in ExecLedger(run_dir).read_all():
        if r.get("stage") != "exec":
            continue
        u = r.get("unit", {})
        key = json.dumps(u, sort_keys=True, ensure_ascii=False)
        rec = units.setdefault(key, {
            "asset": u.get("asset"), "rung": u.get("rung"), "task_hash": u.get("task_hash"),
            "attempts": 0, "failures": 0, "completed": False,
            "last_status": None, "last_exit": None, "last_error": None})
        st, pay = r.get("status"), r.get("payload", {})
        rec["last_status"] = st
        if st == "failed":
            rec["failures"] += 1
            rec["attempts"] += 1
            rec["last_exit"] = pay.get("exit_code")
            rec["last_error"] = pay.get("error")
        elif st == "completed":
            rec["completed"] = True
            rec["attempts"] += 1
    return units


def diagnose(rec, max_retries):
    """What the Repair Loop should do with one unit — a pure function of its ledger history.

    The CURRENT status (last event) decides, not the latched `completed` flag: a unit that completed
    and was then re-run to a divergent, non-reproducible artifact (exit 95) ends `failed`, and must be
    quarantined — never masked as `repaired` because it once succeeded.
    """
    if rec["last_status"] == "completed":
        return "repaired" if rec["failures"] else "ok"      # last event is success -> done/repaired
    if rec["last_status"] != "failed":
        return "ok"                                         # running, or never ran
    code = rec["last_exit"]
    if code == 95:
        return "quarantine_integrity"                       # non-reproducible — never retry/overwrite
    if code in NON_RETRYABLE:
        return "failed_technical"                           # deterministic technical — retry futile
    if rec["failures"] >= max_retries:
        return "failed_technical"                           # transient but exhausted
    return "safe_retry"                                     # transient with budget left


def technical_terminal(run_dir, max_retries=2):
    """{asset: reason} for assets blocked by a terminal technical failure — what the orchestrator
    overlays as FAILED_TECHNICAL (kept strictly apart from the scientific NEEDS_CONTRACT)."""
    out = {}
    for rec in classify(run_dir).values():
        d = diagnose(rec, max_retries)
        if d in ("failed_technical", "quarantine_integrity") and rec["asset"]:
            out[rec["asset"]] = f"rung{rec['rung']} {EXIT_LABEL.get(rec['last_exit'], rec['last_exit'])}"
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Repair Loop — classify technical failures.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-retries", type=int, default=2)
    args = ap.parse_args()
    for rec in classify(args.run_dir).values():
        print(f"  {rec['asset']:<6} rung{rec['rung']} attempts={rec['attempts']} "
              f"fail={rec['failures']} -> {diagnose(rec, args.max_retries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
