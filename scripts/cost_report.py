#!/usr/bin/env python3
"""The empirical cost model — measured core-seconds per stage and per unit, not a static estimate.

The contract carries a `cost_model_core_seconds_per_ticker` block, but its note says it plainly: the
numbers were measured before the runtime fix, uncapped, and must be re-measured before budgeting a
new panel. This assembles the real figures from what the runs actually recorded, so the budget for
the next asset rests on measurement rather than memory.

Two granularities, from two sources already written:

  per stage   run_manifest.json of each stage: wall_seconds, core_seconds, workers, and
              parallel_efficiency = core / (wall x workers). Efficiency near 1 means the workers
              were busy; well below 1 means the stage was starved or the pools were not capped.
  per unit    the per-permutation `seconds` now recorded in each null ledger, giving the true cost
              of one (ticker, outer_fold, permutation) — the atom the budget multiplies.

From these it derives the quantities the vision asks for: evaluation_cost (one permutation of the
search), null_cost (a whole null artifact), and the projected cost of a fresh panel.

    python3 scripts/cost_report.py
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init  # noqa: E402,F401
runtime_init.apply()
from artifact_io import read_json, write_json_atomic                       # noqa: E402
from ledger import Ledger                                                  # noqa: E402

DATA = XGB / "data"
RUNS = DATA / "runs"
OUT = DATA / "cost_report.json"


def stage_costs():
    """Every run manifest under the runs tree, newest first."""
    out = []
    for man in sorted(RUNS.glob("*/**/run_manifest.json")):
        d = read_json(man)
        if d is None:
            continue
        out.append({"path": str(man.relative_to(RUNS)), "stage": d.get("stage"),
                    "status": d.get("status"), "workers": d.get("workers"),
                    "wall_seconds": d.get("wall_seconds"), "core_seconds": d.get("core_seconds"),
                    "parallel_efficiency": d.get("parallel_efficiency")})
    return out


def unit_costs():
    """Per-permutation seconds pulled from every null ledger — the budget's atom."""
    secs = []
    for led_path in sorted(RUNS.glob("*/null_*/ledger.jsonl")):
        led = Ledger(led_path)
        for r in led.read_all():
            if r["status"] == "completed":
                s = r.get("payload", {}).get("seconds")
                if s is not None:
                    secs.append(float(s))
    if not secs:
        return {"n": 0, "_note": "brak per-unit seconds (starsze jednostki sprzed pola)"}
    secs.sort()
    return {"n": len(secs),
            "core_seconds_per_permutation": {
                "median": round(st.median(secs), 2),
                "mean": round(sum(secs) / len(secs), 2),
                "p10": round(secs[int(0.10 * (len(secs) - 1))], 2),
                "p90": round(secs[int(0.90 * (len(secs) - 1))], 2),
                "min": round(secs[0], 2), "max": round(secs[-1], 2)},
            "_atom": "one (ticker, outer_fold, permutation) evaluation of the full 45-candidate search"}


def null_costs():
    """Total core-seconds per null artifact, from its stage manifest(s)."""
    out = {}
    for kind in ("a1", "a2", "b"):
        cs = [m["core_seconds"] for m in stage_costs()
              if m["stage"] == f"null_{kind}" and m["core_seconds"]]
        if cs:
            out[kind] = {"runs": len(cs), "total_core_seconds": round(sum(cs), 1),
                         "core_hours": round(sum(cs) / 3600, 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    stages = stage_costs()
    units = unit_costs()
    nulls = null_costs()

    # A fresh-panel projection, when the per-unit atom is known: worst case is the full budget on
    # every scoped fold, best case reflects the futility stops the real run showed.
    projection = None
    if units.get("n"):
        per = units["core_seconds_per_permutation"]["median"]
        projection = {
            "core_seconds_per_permutation_median": per,
            "worst_case_16_folds_2_arms_50_perms": round(16 * 50 * per / 3600, 2),
            "_unit": "core-hours, if no fold stopped early",
            "_note": "the real run stops rejected folds at b=5, so the realized cost is lower; see "
                     "null_costs for the measured totals"}

    report = {"_what": "measured cost model — supersedes the static contract estimate",
              "per_stage": stages, "per_unit": units, "per_null_artifact": nulls,
              "fresh_panel_projection": projection}
    sha = write_json_atomic(args.out, report)

    print("Model kosztowy — zmierzony, nie szacowany\n")
    print(f"{'etap':<28}{'workerów':>9}{'wall s':>9}{'core s':>9}{'wydajność':>11}")
    for m in stages:
        if m["core_seconds"]:
            print(f"{(m['stage'] or '?'):<28}{str(m['workers']):>9}{m['wall_seconds']:>9.0f}"
                  f"{m['core_seconds']:>9.0f}{str(m['parallel_efficiency']):>11}")
    if units.get("n"):
        p = units["core_seconds_per_permutation"]
        print(f"\nper permutacja (n={units['n']}): mediana {p['median']}s  "
              f"p10 {p['p10']}s  p90 {p['p90']}s")
    if nulls:
        print("per artefakt null:", {k: v["core_hours"] for k, v in nulls.items()}, "rdz-h")
    print(f"\nwrote {args.out}  sha256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
