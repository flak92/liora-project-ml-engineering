#!/usr/bin/env python3
"""Rung 8 — the cross-asset matrix. Which OHLCV relationships travel, and which are one asset's quirk?

The per-ticker ladder answers "what works for AZO". This asks the transfer question: a family that
survives on one asset might be a universal edge, a regime-conditional one, or an artefact of that
single table. The only honest way to tell is to look across the panel — so this aggregates the
per-ticker artifacts into a ticker x family matrix and classifies each family by how widely it
recurs.

The unit is the FAMILY, not the feature, because that is the ladder's stability unit: the same
OHLCV relationship recurring across assets counts even when a different variant of it wins each
time. Flat-arm feature ids are mapped back to their family so both arms speak the same language.

Classification, on the strongest null evidence available:
  universal        confirmed on a majority of the tickers where it was even a provisional candidate
  conditional      confirmed on more than one ticker but not a majority — travels within a regime
  asset_specific   confirmed on exactly one ticker — needs local re-confirmation before transfer
  unconfirmed      provisionally accepted somewhere but never survived the null

This trains nothing and reads only artifacts already written. It degrades honestly: with only the
smoke null present it classifies on those folds and says so; with the full A1 it uses all sixteen.

The output is also a TRANSFER PRIOR: families ranked by cross-asset recurrence, the seed for
narrowing the candidate pool on a new ticker (search the widely-confirmed families first). The prior
orders effort; it never skips the ladder — a new asset still earns its own confirmation.

    python3 scripts/cross_asset_matrix.py
"""
import argparse
import json
import sys
from collections import defaultdict
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
NULLS = [("a1", DATA / "procedure_null_a1.json"),
         ("a1_smoke", DATA / "procedure_null_a1_smoke.json")]
OUT = DATA / "cross_asset_matrix.json"

MAJORITY = 0.5                     # "universal" = confirmed on strictly more than half its candidate tickers


def _families():
    import feature_search as FS
    import golden
    cands = FS.candidate_ids()
    fam_of = golden.load_families(FS.FAMILIES_PATH, cands)          # {feature_id: family}
    return fam_of


def _null_index():
    """(ticker, outer_fold, arm) -> verdict, from the strongest available null; plus its label."""
    for kind, path in NULLS:
        doc = read_json(path)
        if doc is not None:
            idx = {(f["ticker"], f["outer_fold"], a): v
                   for t in doc["tables"].values() for f in t["folds"]
                   for a, v in f["arms"].items()}
            return idx, kind
    return {}, "none"


def _family_of(unit, arm, fam_of):
    """Hierarchical units are already families; flat units are feature ids mapped to their family."""
    if arm == "hierarchical":
        return unit
    try:
        return fam_of.get(int(unit), f"feature:{unit}")
    except (TypeError, ValueError):
        return str(unit)


def build(crossfit, null_idx, null_kind, fam_of):
    # per (ticker, family): was it a provisional candidate, and did it pass the null, anywhere?
    cand = defaultdict(lambda: defaultdict(lambda: {"provisional": False, "null_passed": False,
                                                    "null_rejected": False, "screening": False,
                                                    "folds": [], "best_win": 0.0, "best_delta": None}))
    for ticker, rec in crossfit["tables"].items():
        for f in rec["folds"]:
            for arm in ("flat", "hierarchical"):
                v = f["verdict"][arm]
                if not v.get("accepted"):
                    continue
                fam = _family_of(v["unit"], arm, fam_of)
                cell = cand[fam][ticker]
                cell["provisional"] = True
                cell["folds"].append({"outer_fold": f["outer_fold"], "arm": arm})
                wr = v["wins"] / v["n_deltas"]
                cell["best_win"] = max(cell["best_win"], wr)
                d = v["median_delta"]
                cell["best_delta"] = d if cell["best_delta"] is None else max(cell["best_delta"], d)
                nv = null_idx.get((ticker, f["outer_fold"], arm))
                if nv is None:
                    cell["screening"] = True
                elif nv["verdict"] == "passed":
                    cell["null_passed"] = True
                else:
                    cell["null_rejected"] = True

    families = []
    for fam, tickers in sorted(cand.items()):
        prov_t = sorted(tickers)
        passed_t = sorted(t for t, c in tickers.items() if c["null_passed"])
        screening_t = sorted(t for t, c in tickers.items() if c["screening"] and not c["null_passed"])
        n_prov, n_pass = len(prov_t), len(passed_t)

        if n_pass == 0:
            cls = "unconfirmed" if not screening_t else "screening_only"
        elif n_pass == 1:
            cls = "asset_specific"
        elif n_pass > MAJORITY * n_prov:
            cls = "universal"
        else:
            cls = "conditional"

        families.append({
            "family": fam,
            "classification": cls,
            "provisional_tickers": prov_t,
            "confirmed_tickers": passed_t,
            "screening_tickers": screening_t,
            "n_provisional": n_prov, "n_confirmed": n_pass,
            "recurrence": round(n_pass / n_prov, 3) if n_prov else 0.0,
            "cells": {t: {"provisional": c["provisional"], "null_passed": c["null_passed"],
                          "null_rejected": c["null_rejected"], "screening": c["screening"],
                          "best_win_rate": round(c["best_win"], 3),
                          "best_outer_delta": (round(c["best_delta"], 6)
                                               if c["best_delta"] is not None else None),
                          "folds": c["folds"]}
                      for t, c in sorted(tickers.items())},
        })

    # transfer prior: confirmed families first (by recurrence, then reach), then screening, then the rest
    order = {"universal": 0, "conditional": 1, "asset_specific": 2, "screening_only": 3, "unconfirmed": 4}
    prior = sorted(families, key=lambda fam: (order[fam["classification"]],
                                              -fam["n_confirmed"], -fam["recurrence"], fam["family"]))
    return {
        "_what": "ticker x family recurrence of ladder-confirmed OHLCV relationships",
        "null_authority": null_kind,
        "_null_note": ("full 16-fold A1" if null_kind == "a1" else
                       "SMOKE null only (3 folds) — classification firms up when the full A1 lands"
                       if null_kind == "a1_smoke" else "no procedure null yet — cross-fit only"),
        "majority_threshold": MAJORITY,
        "counts": {cls: sum(1 for f in families if f["classification"] == cls)
                   for cls in ("universal", "conditional", "asset_specific",
                               "screening_only", "unconfirmed")},
        "families": families,
        "transfer_prior": [{"family": f["family"], "classification": f["classification"],
                            "confirmed_tickers": f["confirmed_tickers"],
                            "recurrence": f["recurrence"]} for f in prior],
        "_transfer_prior_use": "order in which to search families on a NEW ticker; it narrows effort "
                               "and never skips the ladder — a new asset earns its own confirmation",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    crossfit = read_json(CROSSFIT)
    if crossfit is None:
        sys.exit(f"brak {CROSSFIT}")
    null_idx, null_kind = _null_index()
    fam_of = _families()
    matrix = build(crossfit, null_idx, null_kind, fam_of)
    sha = write_json_atomic(args.out, matrix)

    print(f"Rung 8 — macierz cross-asset  (null: {null_kind} — {matrix['_null_note']})\n")
    print(f"{'klasa':<16}{'liczba':>7}")
    for cls, n in matrix["counts"].items():
        print(f"{cls:<16}{n:>7}")
    print(f"\n{'rodzina':<20}{'klasa':<16}{'potwierdzone tickery':<34}{'reach':>6}")
    for f in matrix["families"]:
        if f["n_confirmed"] > 0:
            tk = ",".join(f["confirmed_tickers"])
            print(f"{f['family']:<20}{f['classification']:<16}{tk[:32]:<34}"
                  f"{f['n_confirmed']}/{f['n_provisional']}")
    print(f"\nwrote {args.out}  sha256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
