#!/usr/bin/env python3
"""Parallel-over-assets driver — the thin execution layer that replaces the queue + worker-pool + guard
+ scheduler + heartbeat.

The whole executable Calibration DAG (1→3→4→5→6) is per-asset: every runner reads only its own asset's
row of a register, never another asset's. So the coordination machinery — atomic-claim mutual exclusion,
guard stale-requeue, single-writer reducer, panel-readiness gating — existed ONLY to serialize a SHARED
workspace. Stop sharing it: give each asset a PRIVATE workspace, run its DAG to a terminal state in one
process, and run assets in a bounded pool. No shared file is written concurrently, so the entire class
of bug that machinery guarded against (and itself caused: requeue → concurrent duplicate → FAILED_INTEGRITY)
cannot occur.

What is preserved verbatim: immutable artifacts + idempotent/integrity `_publish` (resume + the byte-level
seal), the contract gate, the ledger, the seed+PYTHONHASHSEED injection (dispatch), and #2 fold-parallelism
inside the null. What moves here from the guard: a RAM-aware concurrency cap and the wall deadline. A dead
child is simply re-run once — resume from its immutable artifacts skips finished rungs.

    python3 engine/asset_driver.py --run-dir runs/<id> [--assets AZO ADBE ...] [--max-parallel N]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract as CT                                                       # noqa: E402
import planner as PL                                                       # noqa: E402
import reducer as RD                                                       # noqa: E402
import schemas as SC                                                       # noqa: E402
import states as ST                                                        # noqa: E402
import worker as WK                                                        # noqa: E402
from exec_ledger import ExecLedger                                         # noqa: E402

MAX_STEPS = 8                       # 5 rungs + margin; a bound against a mis-derived non-terminal spin
MIN_AVAIL_MB = int(os.environ.get("DRIVER_MIN_AVAIL_MB", "700"))   # RAM+swap headroom to launch a child


# ---- cooperative stop / deadline / RAM (the load-bearing bits moved out of guard.sh) --------------

def _halted(control):
    if not control:
        return False
    try:
        return bool(json.loads(Path(control).read_text(encoding="utf-8")).get("halt"))
    except (OSError, json.JSONDecodeError):
        return False


def _past_deadline(deadline_epoch):
    return deadline_epoch is not None and time.time() >= float(deadline_epoch)


def _headroom_mb():
    """MemAvailable + SwapFree in MB — the same headroom the old guard watched."""
    try:
        vals = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            if k in ("MemAvailable", "SwapFree"):
                vals[k] = int(rest.strip().split()[0]) // 1024
        return vals.get("MemAvailable", 0) + vals.get("SwapFree", 0)
    except (OSError, ValueError):
        return 1 << 30                              # unknown -> do not throttle


def _ram_ok(inflight):
    """Launch another asset only with headroom. Always allow the first (never deadlock at zero)."""
    return inflight == 0 or _headroom_mb() > MIN_AVAIL_MB


# ---- one asset's whole DAG (its private workspace) -----------------------------------------------

def run_asset_dag(run_dir, asset, seed=42, control=None):
    """Drive one asset 1→3→4→5→6 to a terminal state in its PRIVATE workspace. Sequential within the
    asset (natural rung order — no queue, no gating); parallel ACROSS assets is run_panel's job.
    Idempotent: a re-run resumes from immutable artifacts (finished rungs are skipped). Returns the
    terminal state string, 'HALTED', or 'FAILED:<outcome>'."""
    led = ExecLedger(run_dir)
    snap = CT.load(run_dir)
    ws = RD.workspace(run_dir, asset)
    for _ in range(MAX_STEPS):
        if _halted(control):
            return "HALTED"
        st = ST.derive_state(run_dir, asset)["state"]
        if ST.is_terminal(st):
            return st
        rung = PL.NEXT_RUNG.get(st)
        if rung is None:
            return st
        RD.assemble_asset_inputs(run_dir, asset)          # this asset's one-row registers into its ws
        task = SC.make_task(snap["run_id"], asset, rung, snap["contract_hash"], seed)
        r = WK.execute_task(run_dir, task, ws, f"asset-{asset}", led)
        if r["outcome"] not in ("published", "idempotent"):
            return f"FAILED:{r['outcome']}"               # run_panel decides restart vs terminal
    return ST.derive_state(run_dir, asset)["state"]


# ---- bounded parallel-over-assets pool -----------------------------------------------------------

def run_panel(run_dir, assets, seed=42, control=None, deadline_epoch=None, max_parallel=None):
    """Run every asset's DAG in a bounded pool (fork), RAM-aware, with a dead-child re-run and a wall
    deadline; then ONE terminal reducer rollup assembles the full panels for the report / cross-asset
    matrix. fold-jobs (#2, inside the null) nests under this: cap × fold_jobs ≈ cores."""
    fold_jobs = int(os.environ.get("RESEARCH_FOLD_JOBS", "1") or 1)
    cores = os.cpu_count() or 2
    cap = max(1, min(max_parallel or cores, cores // max(1, fold_jobs) or 1))

    pending, attempts, results = list(assets), {a: 0 for a in assets}, {}
    with ProcessPoolExecutor(max_workers=cap, mp_context=get_context("fork")) as ex:
        inflight = {}
        while pending or inflight:
            while pending and len(inflight) < cap and _ram_ok(len(inflight)) and not _halted(control) \
                    and not _past_deadline(deadline_epoch):
                a = pending.pop(0)
                inflight[ex.submit(run_asset_dag, run_dir, a, seed, control)] = a
            if not inflight:
                if _halted(control) or _past_deadline(deadline_epoch):
                    break
                time.sleep(5)                              # RAM too tight to launch even one — wait
                continue
            done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in done:
                a = inflight.pop(fut)
                try:
                    results[a] = fut.result()
                except Exception as e:                     # noqa: BLE001 — a dead child
                    attempts[a] += 1
                    if attempts[a] <= 1:
                        pending.append(a)                  # re-run once; idempotent via artifacts
                    else:
                        results[a] = f"FAILED_TECHNICAL:{type(e).__name__}"
            if _halted(control) or _past_deadline(deadline_epoch):
                break

    RD.assemble_inputs(run_dir)                            # terminal rollup: full panels for report
    return {a: ST.derive_state(run_dir, a)["state"] for a in assets}


def main():
    ap = argparse.ArgumentParser(description="Parallel-over-assets DAG driver.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-parallel", type=int, default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    if not (rd / "contract.json").exists():
        assets = args.assets or json.loads((ROOT / "config" / "sample_20.json").read_text())["sample"]
        CT.snapshot(rd, assets, seed=args.seed, allow_dirty=args.allow_dirty)
    assets = args.assets or CT.load(rd)["assets"]
    control = args.control or str(rd / "control.json")

    t0 = time.time()
    final = run_panel(str(rd), assets, seed=args.seed, control=control, max_parallel=args.max_parallel)
    print(f"\n=== stany końcowe ({time.time()-t0:.0f}s) ===")
    for a in assets:
        print(f"  {a:<7}{final.get(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
