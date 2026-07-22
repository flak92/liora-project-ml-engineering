#!/usr/bin/env python3
"""The report — the funnel and the per-asset descriptions, derived from artifacts, never hand-written.

The funnel `provisional -> A1 survivors -> A1xA2xB stable -> Rung 6 retained` is COMPUTED from the
result artifacts every time. The known development-panel numbers (26 -> 11 -> 9 -> 4) are not baked
in anywhere; they are only a snapshot-parity check — proof that the branch reads the frozen artifacts
correctly. A fresh panel is free to produce 30 -> 7 -> 2 -> 0 and be just as correct, because the
funnel is a property of the data, not a success condition.

Two sources, one code path:

    presentation   results/methodology_snapshot/   frozen dev-panel verdicts, read in a blink
    reproduction   runs/<id>/results/panels/       a run the engine just produced

    python3 engine/report.py --snapshot                     # presentation
    python3 engine/report.py --run-dir runs/<id>            # reproduction
    python3 engine/report.py --snapshot --parity 26 11 9 4  # assert the known funnel
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from artifact_io import read_json, write_json_atomic                       # noqa: E402

SNAPSHOT = ROOT / "results" / "methodology_snapshot"


def _load(source, name):
    return read_json(Path(source) / name)


def _accepted_arms(crossfit):
    n = 0
    if crossfit:
        for rec in crossfit["tables"].values():
            for f in rec["folds"]:
                for arm in ("flat", "hierarchical"):
                    if f["verdict"][arm].get("accepted"):
                        n += 1
    return n


def _passed(null_doc):
    keys = set()
    if null_doc:
        for t, rec in null_doc["tables"].items():
            for f in rec["folds"]:
                for arm, v in f["arms"].items():
                    if v.get("verdict") == "passed":
                        keys.add((t, f["outer_fold"], arm))
    return keys


def funnel(source):
    """Every number derived from the artifacts in `source`. No constant is assumed."""
    crossfit = _load(source, "crossfit_selection.json")
    a1 = _load(source, "procedure_null_a1.json")
    a2 = _load(source, "procedure_null_a2.json")
    b = _load(source, "procedure_null_b.json")
    r6 = _load(source, "rung6_survivor_hpo.json")

    prov = _accepted_arms(crossfit)
    pa1, pa2, pb = _passed(a1), _passed(a2), _passed(b)
    stable = pa1 & (pa2 if a2 else pa1) & (pb if b else pa1)
    retained = sum(1 for r in (r6 or {}).get("results", []) if r.get("verdict") == "retained")

    return {
        "provisional_crossfit": prov,
        "passed_a1_marginal": len(pa1),
        "stable_a1_a2_b": len(stable),
        "retained_rung6": retained,
        "_note": "derived from artifacts; a fresh panel may differ and still be correct",
        "stable_units": sorted(f"{t}/{o}/{a}" for t, o, a in stable),
    }


def compiled(source):
    """Per-asset descriptions — from the snapshot's compiled/ dir, or (reproduction) rebuilt."""
    d = Path(source) / "compiled"
    if d.is_dir():
        return {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted(d.glob("*.json"))}
    return {}


def build(source, out=None):
    fn = funnel(source)
    comp = compiled(source)
    resolved = sum(1 for c in comp.values() if c.get("status") == "resolved")
    conditional = sum(1 for c in comp.values() if c.get("status") == "resolved_conditional")
    empty = sum(1 for c in comp.values() if c.get("status") == "resolved_empty")
    report = {"source": str(source), "funnel": fn,
              "compiled_counts": {"resolved": resolved, "resolved_conditional": conditional,
                                  "resolved_empty": empty, "total": len(comp)},
              "assets": {a: {"status": c.get("status"),
                             "selected_features": c.get("selected_features", [])}
                         for a, c in comp.items()}}
    if out:
        write_json_atomic(out, report)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--parity", nargs=4, type=int, default=None,
                    help="assert funnel == these four numbers (snapshot-parity test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    source = SNAPSHOT if args.snapshot or not args.run_dir else Path(args.run_dir) / "results" / "panels"
    rep = build(source, args.out)
    fn = rep["funnel"]
    print(f"Lejek (wyprowadzony z artefaktów w {source.name if hasattr(source,'name') else source}):\n")
    print(f"  provisional (cross-fit)        {fn['provisional_crossfit']}")
    print(f"  → passed A1 (marginalny)       {fn['passed_a1_marginal']}")
    print(f"  → stabilne A1×A2×B             {fn['stable_a1_a2_b']}")
    print(f"  → retained Rung 6              {fn['retained_rung6']}")
    print(f"\n  compiler: {rep['compiled_counts']}")

    if args.parity:
        got = [fn["provisional_crossfit"], fn["passed_a1_marginal"],
               fn["stable_a1_a2_b"], fn["retained_rung6"]]
        ok = got == args.parity
        print(f"\n  snapshot-parity {args.parity}: {'ZGODNE' if ok else f'ROZJAZD {got}'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
