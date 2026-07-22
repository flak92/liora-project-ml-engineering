#!/usr/bin/env python3
"""The Repair Loop — the third layer, and the one allowed to touch nothing scientific.

It reads the execution ledger and decides what to do with technical failures: retry the transient
ones, and stop retrying the ones a retry cannot fix. Its whole vocabulary is execution-layer —
worker crashes, timeouts, a non-reproducible artifact — and it is forbidden, structurally, from
reaching a threshold, a fold, a null, or the OOS boundary. A `FAILED_TECHNICAL` unit is not a
scientific verdict; it is a machine that broke, kept apart from `NEEDS_CONTRACT` (a science stop) on
purpose.

The diagnosis is a pure function of a unit's ledger history, so it is testable without a machine that
actually crashes:

    exit 95 (FAILED_INTEGRITY)      -> quarantine: a retry produced different bytes; never retry, never
                                       overwrite. The experiment was not reproducible — that is a
                                       finding, recorded, not papered over.
    exit 7 / 92 (contract / schema) -> failed_technical: deterministic; a retry is futile.
    other non-zero, attempts < N    -> safe_retry: a transient (OOM, timeout) with budget left.
    other non-zero, attempts >= N   -> failed_technical: transient but exhausted.
    a later 'completed' for the unit -> repaired (an earlier failure the loop already recovered).

`handle()` performs the safe retries by moving the task file failed/ -> pending/ (the same atomic
rename the queue uses everywhere) and returns the terminal-technical units so the orchestrator can
overlay a FAILED_TECHNICAL state onto the asset. It never elevates a science stop and never edits a
result artifact.
"""
import json
import os
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
    """What the Repair Loop should do with one unit — a pure function of its ledger history."""
    if rec["completed"]:
        return "repaired" if rec["failures"] else "ok"      # a later success superseded any failure
    if rec["last_status"] != "failed":
        return "ok"                                         # running, or never failed
    code = rec["last_exit"]
    if code == 95:
        return "quarantine_integrity"                       # non-reproducible — never retry/overwrite
    if code in NON_RETRYABLE:
        return "failed_technical"                           # deterministic technical — retry futile
    if rec["failures"] >= max_retries:
        return "failed_technical"                           # transient but exhausted
    return "safe_retry"                                     # transient with budget left


def _requeue_failed(run_dir, task_hash):
    """Move one failed task back to pending for retry — the same atomic rename the queue uses. Returns
    True if it moved. (Kept here rather than in taskqueue so the live queue module is untouched.)"""
    q = Path(run_dir) / "queue"
    src = q / "failed" / f"{task_hash}.json"
    if not src.exists():
        return False
    os.rename(src, q / "pending" / f"{task_hash}.json")     # atomic within one filesystem
    return True


def handle(run_dir, max_retries=2):
    """Classify, then act: requeue safe retries, leave terminal-technical units in failed/ for audit.

    Returns a summary the orchestrator logs into iteration_trace and overlays onto per-asset state:
    {repaired, safe_retry, quarantine_integrity, failed_technical} — each a list of unit records."""
    buckets = {"repaired": [], "safe_retry": [], "quarantine_integrity": [], "failed_technical": []}
    for rec in classify(run_dir).values():
        d = diagnose(rec, max_retries)
        if d == "ok":
            continue
        entry = {**rec, "label": EXIT_LABEL.get(rec["last_exit"], rec["last_exit"])}
        if d == "safe_retry":
            entry["requeued"] = _requeue_failed(run_dir, rec["task_hash"])
        buckets[d].append(entry)
    return buckets


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
    ap.add_argument("--act", action="store_true", help="wykonaj bezpieczne retry (przenieś failed→pending)")
    args = ap.parse_args()
    if args.act:
        b = handle(args.run_dir, args.max_retries)
        for k, v in b.items():
            if v:
                items = [(e["asset"], "rung%s" % e["rung"], e["label"]) for e in v]
                print("  %s: %s" % (k, items))
        if not any(b.values()):
            print("  brak awarii technicznych")
    else:
        for rec in classify(args.run_dir).values():
            print(f"  {rec['asset']:<6} rung{rec['rung']} attempts={rec['attempts']} "
                  f"fail={rec['failures']} -> {diagnose(rec, args.max_retries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
