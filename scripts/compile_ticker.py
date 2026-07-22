#!/usr/bin/env python3
"""The Feature Discovery Compiler — one resolved verdict per ticker, assembled, never recomputed.

The ladder writes many artifacts, each answering one rung's question. A person asking "so what did
the method decide for AZO?" should not have to read five of them. This assembles the answer the
owner specified in the vision document: for each ticker, the features that survived the whole ladder
(or the honest statement that none did), with the numbers that justify it and the reason the search
stopped.

Every field is copied from an artifact that already computed it — this file trains nothing and runs
no model:

  status              resolved when at least one unit passed the procedure-level null; resolved_empty
                      when the ladder confirmed nothing. Both are correct outcomes.
  selected_features   the units (features for flat, families for hierarchical) that passed the null
  rejected_features   how many candidates did not survive
  max_null_p          per surviving unit: p_mc if it passed, or the lower bound if it was rejected
  confirmation_win_rate, outer_delta   the cross-fit evidence behind each unit
  compute_seconds     measured, from the run manifests and per-ticker artifact timings
  stop_reason         why the search ended for this ticker

J is attached as a REPORTING scalar, never a gate. The gates and the max-null already decided what is
admissible; J only ranks the survivors and books the cost, so a reader can order them. Its weights
are illustrative reporting weights, not calibrated coefficients, and a feature with a bad J that
nonetheless passed every gate is still selected — J cannot overturn a gate, by construction.

    python3 scripts/compile_ticker.py                 # every ticker, to xgb/data/compiled/
    python3 scripts/compile_ticker.py AZO --print     # one ticker to stdout
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(XGB / "tools"))
sys.path.insert(0, str(XGB / "src"))
import runtime_init  # noqa: E402,F401
runtime_init.apply()
from artifact_io import read_json, write_json_atomic                       # noqa: E402

DATA = XGB / "data"
CROSSFIT = DATA / "crossfit_selection.json"
NULLS = {"a1": DATA / "procedure_null_a1.json",
         "a1_smoke": DATA / "procedure_null_a1_smoke.json",
         "a2": DATA / "procedure_null_a2.json",
         "b": DATA / "procedure_null_b.json"}
OUTDIR = DATA / "compiled"

# Reporting weights for J. NOT calibrated coefficients — they order survivors and nothing more.
J_WEIGHTS = {"optimism": 1.0, "instability": 0.5, "complexity": 0.004, "cost": 1e-5}


def _null_index(kind):
    """{(ticker, outer_fold, arm): verdict} for whichever null artifact exists."""
    doc = read_json(NULLS[kind])
    if doc is None:
        return None
    return {(f["ticker"], f["outer_fold"], a): v
            for t in doc["tables"].values() for f in t["folds"]
            for a, v in f["arms"].items()}


def _candidate_count():
    try:
        import feature_search as FS
        return len(FS.candidate_ids())
    except Exception:                                              # noqa: BLE001
        return None


def _null_p(v):
    """The p-value to report, faithful to how the test ended."""
    if v["verdict"] == "passed":
        return {"p_mc": v.get("p_mc"), "basis": "fixed budget, 50 permutations"}
    if v["verdict"] == "rejected_early":
        return {"p_lower_bound": v.get("final_p_lower_bound"),
                "basis": "futility stop — a lower bound, not a fixed-budget p-value"}
    return {"p": None, "basis": v["verdict"]}


def compile_ticker(ticker, crossfit, nulls, n_candidates):
    """One resolved record. `nulls` maps kind -> index (or None); the strongest available null decides."""
    rec = crossfit["tables"].get(ticker)
    if rec is None:
        return {"ticker": ticker, "model": "xgb", "status": "absent",
                "stop_reason": "no cross-fit record for this ticker"}

    # Which null is the authority: the full A1 if present, else the smoke, else screening only.
    null_kind = next((k for k in ("a1", "a1_smoke") if nulls.get(k) is not None), None)
    null_idx = nulls.get(null_kind) if null_kind else None

    seconds = float(rec.get("seconds", 0.0))
    accepted, rejected_by_null, screening_only = [], [], []
    for f in rec["folds"]:
        for arm in ("flat", "hierarchical"):
            v = f["verdict"][arm]
            if not v.get("accepted"):
                continue                                          # never cleared cross-fit — not a survivor
            entry = {"outer_fold": f["outer_fold"], "arm": arm, "unit": v["unit"],
                     "representatives": v.get("representatives", [v["unit"]]),
                     "confirmation_win_rate": round(v["wins"] / v["n_deltas"], 3),
                     "outer_delta": round(v["median_delta"], 6), "T_real": v["T"]}
            nv = null_idx.get((ticker, f["outer_fold"], arm)) if null_idx else None
            if nv is None:
                entry["max_null"] = {"basis": "screening only — procedure null not yet run here"}
                screening_only.append(entry)
            elif nv["verdict"] == "passed":
                entry["max_null"] = _null_p(nv)
                accepted.append(entry)
            else:
                entry["max_null"] = _null_p(nv)
                rejected_by_null.append(entry)

    units = sorted({e["unit"] for e in accepted}, key=str)
    if accepted:
        status, stop = "resolved", "survived the procedure-level max-null"
    elif screening_only and not null_idx:
        status, stop = "screening_only", "cross-fit accepted; procedure null not yet run"
    elif rejected_by_null or screening_only:
        status, stop = "resolved_empty", "no unit exceeded the procedure-level max-null"
    else:
        status, stop = "resolved_empty", "cross-fit accepted nothing (no provisional survivor)"

    out = {
        "ticker": ticker, "model": "xgb",
        "status": status,
        "null_authority": null_kind or "none",
        "selection_mode": (sorted({e["arm"] for e in accepted})[0] if accepted else None),
        "selected_features": units,
        "n_selected": len(units),
        "rejected_features": (n_candidates - len(units)) if n_candidates else None,
        "confirmation_win_rate": (round(sum(e["confirmation_win_rate"] for e in accepted)
                                        / len(accepted), 3) if accepted else None),
        "max_null_p": [{"unit": e["unit"], "arm": e["arm"], **e["max_null"]} for e in accepted],
        "outer_delta": (round(max(e["outer_delta"] for e in accepted), 6) if accepted else None),
        "compute_seconds": round(seconds, 1),
        "stop_reason": stop,
        "evidence": {"accepted": accepted, "rejected_by_null": rejected_by_null,
                     "screening_only": screening_only},
    }
    out["J_report"] = _objective_report(accepted, rec, seconds)
    return out


def _objective_report(accepted, rec, seconds):
    """J = U_outer − λ₁·optimism − λ₂·instability − λ₃·complexity − λ₄·cost, as a RANKING report.

    Documented as a report so no reader mistakes it for the decision: the decision was the gates and
    the null. Optimism here is discovery_gain − confirm_delta averaged over the surviving arms'
    rotations; instability is 1 − win_rate; complexity is the subset size; cost is core-seconds.
    """
    if not accepted:
        return {"J": None, "_note": "no survivor to rank"}
    # optimism from the rotations that chose each surviving unit
    opt = []
    for f in rec["folds"]:
        for e in accepted:
            if f["outer_fold"] != e["outer_fold"]:
                continue
            for r in f["rotations"]:
                a = r["arms"][e["arm"]]
                key = "picked" if e["arm"] == "flat" else "family"
                if a.get(key) == e["unit"] and a.get("confirm_delta") is not None \
                        and a.get("discovery_gain") is not None:
                    opt.append(a["discovery_gain"] - a["confirm_delta"])
    u_outer = max(e["outer_delta"] for e in accepted)
    optimism = (sum(opt) / len(opt)) if opt else 0.0
    instability = 1.0 - (sum(e["confirmation_win_rate"] for e in accepted) / len(accepted))
    complexity = len({e["unit"] for e in accepted})
    w = J_WEIGHTS
    j = (u_outer - w["optimism"] * optimism - w["instability"] * instability
         - w["complexity"] * complexity - w["cost"] * seconds)
    return {"J": round(j, 6),
            "components": {"U_outer": round(u_outer, 6), "optimism": round(optimism, 6),
                           "instability": round(instability, 6), "complexity": complexity,
                           "cost_core_seconds": round(seconds, 1)},
            "weights": w,
            "_status": "REPORTING scalar — ranks survivors and books cost; never a gate; cannot "
                       "overturn the max-null or the acceptance contract"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--print", action="store_true", dest="to_stdout")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    crossfit = read_json(CROSSFIT)
    if crossfit is None:
        sys.exit(f"brak {CROSSFIT}")
    nulls = {k: _null_index(k) for k in NULLS}
    n_candidates = _candidate_count()
    tickers = args.tickers or sorted(crossfit["tables"])

    outdir = Path(args.outdir)
    records = []
    for t in tickers:
        rec = compile_ticker(t, crossfit, nulls, n_candidates)
        records.append(rec)
        if not args.to_stdout:
            write_json_atomic(outdir / f"{t}.json", rec)

    resolved = [r for r in records if r["status"] == "resolved"]
    empty = [r for r in records if r["status"] == "resolved_empty"]
    print(f"Feature Discovery Compiler — {len(tickers)} tabel  "
          f"(null: {records[0].get('null_authority')})\n")
    print(f"{'ticker':<7}{'status':<16}{'tryb':<14}{'wybrane':<28}{'win':>6}{'J':>9}")
    for r in records:
        feats = ",".join(map(str, r["selected_features"])) or "—"
        wr = f"{r['confirmation_win_rate']:.2f}" if r["confirmation_win_rate"] is not None else "—"
        j = r.get("J_report", {}).get("J")
        js = f"{j:+.4f}" if j is not None else "—"
        print(f"{r['ticker']:<7}{r['status']:<16}{str(r['selection_mode'] or '—'):<14}"
              f"{feats[:26]:<28}{wr:>6}{js:>9}")
    print(f"\n  resolved: {len(resolved)}   resolved_empty: {len(empty)}   "
          f"pozostałe: {len(records) - len(resolved) - len(empty)}")

    if args.to_stdout:
        print("\n" + json.dumps(records[0] if len(records) == 1 else records, indent=1,
                                ensure_ascii=False))
    else:
        print(f"\nwrote {outdir}/<ticker>.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
