#!/usr/bin/env python3
"""The sequential engine loop — plan, enqueue, drain, repeat, until every asset is terminal.

This is the engine without concurrency: one process, one worker, the plainest possible realization of
`plan -> enqueue -> work -> re-plan`. The tmux worker pool of the concurrency phase does the same
thing in parallel; this is the reference the smoke test and the resume test run against, because a
single deterministic thread is the easiest place to prove the state machine actually converges.

    python3 engine/drive.py --run-dir runs/<id> [--assets AZO ADBE ...] [--max-cycles 40]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract as CT                                                      # noqa: E402
import planner as PL                                                       # noqa: E402
import states as ST                                                        # noqa: E402
import worker as WK                                                        # noqa: E402


def drive(run_dir, max_cycles=40, worker_id="worker-00", log=print):
    for cycle in range(max_cycles):
        actions = PL.plan(run_dir)
        if all(ST.is_terminal(a["state"]) for a in actions):
            log(f"[cykl {cycle}] wszystkie assety terminalne")
            return actions
        n = PL.enqueue(run_dir, actions)
        pending = [a for a in actions if a.get("task")]
        log(f"[cykl {cycle}] zakolejkowano {n}: "
            + ", ".join(f"{a['asset']}→rung{a['rung']}" for a in pending))
        # drain the queue with a single worker
        done = 0
        while True:
            r = WK.run_one(run_dir, worker_id)
            if r is None:
                break
            done += 1
            log(f"    {r['asset']}/rung{r['rung']}: {r['outcome']}"
                + (f" ({r.get('seconds')}s)" if r.get("seconds") else ""))
        if done == 0:
            # nothing ran and nothing terminal-only remains blocked forever -> stop to avoid a spin
            blocked = [a for a in actions if a.get("blocked")]
            if not pending and blocked:
                log(f"[cykl {cycle}] zablokowane bez postępu: {[a['asset'] for a in blocked]}")
            else:
                log(f"[cykl {cycle}] pusty przebieg — stop")
            break
    return PL.plan(run_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--max-cycles", type=int, default=40)
    args = ap.parse_args()

    rd = Path(args.run_dir)
    if not (rd / "contract.json").exists():
        assets = args.assets or __import__("json").loads(
            (ROOT / "config" / "sample_20.json").read_text())["sample"]
        CT.snapshot(rd, assets, seed=args.seed, allow_dirty=args.allow_dirty)

    t0 = time.time()
    final = drive(args.run_dir, max_cycles=args.max_cycles)
    print(f"\n=== stany końcowe ({time.time()-t0:.0f}s) ===")
    for a in final:
        print(f"  {a['asset']:<7}{a['state']}"
              + (f"  ({a.get('reason')})" if a.get("reason") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
