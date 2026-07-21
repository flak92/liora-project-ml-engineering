#!/usr/bin/env python3
"""Assemble what a human needs to close Rung 5 — and nothing that decides it for them.

Three things come out of here:

**The verdict table.** Per outer fold and arm: the real statistic, how many permutations ran, the
exceedance count, and the verdict under each null that was run. Where a null stopped early the
p-value is reported as a LOWER BOUND, never as `(1+b)/(1+n)` — a test that chose when to stop does
not get a fixed-budget p-value.

**The paired rotation-versus-procedure comparison.** The rotation-level statistic was recorded as a
by-product of the same permutations, so this answers "how many conclusions change when the whole
acceptance rule is aggregated instead of being tested rotation by rotation" on paired draws rather
than on two independent runs. That difference is a methodological result in itself: it measures how
much multiplicity the four-rotation rule adds on top of the 45-candidate search.

**The Null-A x Null-B cross-tabulation**, in the four cells the contract names, so the case
"A passed, B rejected" is visible as what it is — evidence that the A result came from breaking
`optional <-> core` rather than from information in the feature.

    python3 scripts/rung5_summary.py --run-dir xgb/data/runs/<id>
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
import acceptance as ACC                                                   # noqa: E402
from artifact_io import read_json, write_json_atomic                       # noqa: E402

DATA = XGB / "data"
CROSSFIT = DATA / "crossfit_selection.json"
NULLS = {"a1": DATA / "procedure_null_a1.json",
         "a2": DATA / "procedure_null_a2.json",
         "b": DATA / "procedure_null_b.json"}


def load(kind):
    doc = read_json(NULLS[kind])
    if doc is None:
        return {}
    return {(f["ticker"], f["outer_fold"], a): dict(v, _fold=f)
            for t in doc["tables"].values() for f in t["folds"]
            for a, v in f["arms"].items()}


def rotation_vs_procedure(a1):
    """Paired: for each fold and arm, how the rotation-level null would have judged the same draws.

    The rotation statistic under a permutation is the maximum confirmation delta that permutation's
    search produced in any single rotation — the quantity the old runner compared against a single
    real rotation delta. Comparing it here against the same real rotations, on the same
    permutations, isolates the effect of aggregation alone.
    """
    real = read_json(CROSSFIT)
    out = []
    for (ticker, ofold, arm), v in sorted(a1.items()):
        f = v["_fold"]
        draws = f.get("rotation_level_diagnostic", {}).get("null_deltas_per_permutation") or []
        col = 0 if arm == "flat" else 1
        rot_null = [max((p[col] for p in perm if p and p[col] is not None), default=None)
                    for perm in draws if perm]
        rot_null = [x for x in rot_null if x is not None]
        if not rot_null:
            continue
        rec = [x for x in real["tables"][ticker]["folds"] if x["outer_fold"] == ofold][0]
        key = ACC.unit_key(arm)
        real_rot = [r["arms"][arm]["confirm_delta"] for r in rec["rotations"]
                    if r["arms"][arm][key] == v["unit"] and r["arms"][arm]["confirm_delta"] is not None]
        if not real_rot:
            continue
        r_stat = max(real_rot)
        b_rot = sum(1 for x in rot_null if x >= r_stat)
        out.append({
            "ticker": ticker, "outer_fold": ofold, "arm": arm, "unit": v["unit"],
            "procedure": {"T_real": v["real_statistic"], "exceedances": v["exceedances"],
                          "verdict": v["verdict"], "permutations": v["permutations_executed"]},
            "rotation": {"real_max_delta": round(float(r_stat), 6), "exceedances": b_rot,
                         "permutations": len(rot_null),
                         "verdict": "rejected_early" if b_rot >= 5 else "passed"},
            "conclusion_changes": bool((v["verdict"] == "passed") != (b_rot < 5)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=str(DATA / "rung5_summary.json"))
    args = ap.parse_args()

    got = {k: load(k) for k in NULLS}
    a1 = got["a1"]
    if not a1:
        raise SystemExit("brak procedure_null_a1.json — nie ma czego podsumować")

    rows = []
    for key in sorted(a1):
        ticker, ofold, arm = key
        row = {"ticker": ticker, "outer_fold": ofold, "arm": arm,
               "unit": a1[key]["unit"], "T_real": a1[key]["real_statistic"], "nulls": {}}
        for kind in ("a1", "a2", "b"):
            v = got[kind].get(key)
            if not v:
                row["nulls"][kind] = {"verdict": "not_run"}
                continue
            cell = {"verdict": v["verdict"], "exceedances": v["exceedances"],
                    "permutations_executed": v["permutations_executed"]}
            if v["verdict"] == "passed":
                cell["p_mc"] = v.get("p_mc")
            else:
                cell["final_p_lower_bound"] = v.get("final_p_lower_bound")
                cell["_p_note"] = ("lower bound — a test that chose when to stop does not get a "
                                   "fixed-budget p-value")
            row["nulls"][kind] = cell
        rows.append(row)

    cross = {}
    for r in rows:
        va, vb = r["nulls"]["a1"]["verdict"], r["nulls"]["b"]["verdict"]
        if vb == "not_run":
            continue
        cross.setdefault(f"a1={va} b={vb}", []).append(f"{r['ticker']}/{r['outer_fold']}/{r['arm']}")

    comp = rotation_vs_procedure(a1)
    changed = [c for c in comp if c["conclusion_changes"]]
    summary = {
        "_status": "Development Candidate — a human reads this and writes the verdict",
        "scope": {"outer_fold_arm_pairs": len(rows),
                  "tickers": len({r["ticker"] for r in rows}),
                  "outer_folds": len({(r["ticker"], r["outer_fold"]) for r in rows})},
        "counts": {kind: {v: sum(1 for r in rows if r["nulls"][kind]["verdict"] == v)
                          for v in ("passed", "rejected_early", "incomplete", "not_run")}
                   for kind in ("a1", "a2", "b")},
        "verdicts": rows,
        "null_a_vs_null_b": {
            "_interpretation": {
                "a1=rejected_early b=rejected_early": "no advantage; strong",
                "a1=passed b=passed": "stable survivor",
                "a1=passed b=rejected_early": "the A result probably came from breaking "
                                              "optional <-> core rather than from information",
                "a1=rejected_early b=passed": "A was too conservative, or g is defective"},
            "cells": cross},
        "rotation_vs_procedure": {
            "_role": "paired — same permutations, so this isolates the effect of aggregating the "
                     "whole acceptance rule rather than testing rotation by rotation",
            "conclusions_changed": len(changed),
            "of_total": len(comp),
            "median_rotation_exceedances": (st.median([c["rotation"]["exceedances"] for c in comp])
                                            if comp else None),
            "median_procedure_exceedances": (st.median([c["procedure"]["exceedances"] for c in comp])
                                             if comp else None),
            "rows": comp},
    }
    sha = write_json_atomic(args.out, summary)

    print(f"\nRung 5 — podsumowanie ({len(rows)} par outer_fold × ramię)\n")
    print(f"{'ticker':<7}{'fold':>5} {'arm':<14}{'jednostka':<17}{'T_real':>9}"
          f"{'A1':>16}{'A2':>16}{'B':>16}")
    for r in rows:
        def cell(k):
            v = r["nulls"][k]
            if v["verdict"] == "not_run":
                return "—"
            mark = "PRZESZŁO" if v["verdict"] == "passed" else "odrzucone"
            return f"{mark} b={v['exceedances']}"
        print(f"{r['ticker']:<7}{r['outer_fold']:>5} {r['arm']:<14}{str(r['unit']):<17}"
              f"{r['T_real']:>+9.4f}{cell('a1'):>16}{cell('a2'):>16}{cell('b'):>16}")
    print(f"\nliczności: {json.dumps(summary['counts'], ensure_ascii=False)}")
    print(f"rotacja kontra procedura: {len(changed)}/{len(comp)} wniosków zmienia się przy "
          f"agregacji całej reguły")
    print(f"\nwrote {args.out}  sha256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
