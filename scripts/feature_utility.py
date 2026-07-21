#!/usr/bin/env python3
"""Does a feature that looks important INSIDE a model actually improve a model that can learn?

Stage 2 of the methodology. Stage 1 established that the sealed hyper-parameter space produced a
constant predictor in 63% of draws — under such a model every feature comparison compares
nothing. With the space rescaled to the data's effective hessian that rate is 8%, so the question
can finally be asked properly, and this is where it is asked.

Per ticker and per outer-train window, on that window only:

    one shared event set  ->  identical inner folds  ->  HPO v2 on CORE  ->  freeze
        ->  core | core + one feature | core + one whole family | greedy subset

Every configuration sees the same events, the same folds, the same seed, the same quantile
operating-point contract and the same execution costs. The only thing that varies is which
features are present, which is the only way the difference can be attributed to them.

HPO runs on CORE, not on the superset. The sealed procedure tunes on all 45 candidates at once
and then scores subsets at those parameters, which its own docstring admits inflates the reported
gain: the parameters are co-adapted to features the baseline does not have. Tuning on core and
freezing gives every configuration the same, honestly-earned protocol.

The viability gate — split_nodes >= 20 and pred_std >= 0.005, declared in
config/xgb_search_space_v2.json before any of this ran — decides only whether a configuration
TESTED the features, never which configuration wins. Ranking stays what it was: marginal
Train-OOF log-growth and the fraction of folds beaten. A gate that picked winners would be
tuning dressed as hygiene.

SHAP is carried as a diagnostic column and nothing more. Whether it agrees with the marginal
utility measured here is a question for later, and answering it early would turn a measurement
into a selector.

TRAIN ONLY. Outer validation windows are never read here — this stage produces the register and
the frozen state that stage 3 evaluates against them.

    python3 scripts/feature_utility.py NVDA
    python3 scripts/feature_utility.py --jobs 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
os.environ.setdefault("LIORA_EPOCH", "sealed")
sys.path.insert(0, str(XGB / "src"))
sys.path.insert(0, str(XGB / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

SAMPLE = ROOT / "config" / "sample_20.json"
SPACE_V2 = ROOT / "config" / "xgb_search_space_v2.json"
DEFAULT_OUT = XGB / "data" / "feature_utility.json"
HPO_TRIALS = 30
SEED = 42

# Carried from stage 1: this table produced a degenerate model in 47% of draws even under the
# rescaled space, where every other table produced none. Flagged rather than dropped, so stage 3
# can ask whether feature instability concentrates on tables whose models can barely learn.
HIGH_DEGENERACY = {"GWW"}


def _space():
    return json.loads(SPACE_V2.read_text(encoding="utf-8"))


def viability_floor():
    v = _space()["viability"]
    return int(v["min_split_nodes"]), float(v["min_pred_std"])


def hpo_core(dfx, dfb, tev, bnds, names_core, H, seed=SEED, trials=HPO_TRIALS):
    """Tune on core alone, keep the best VIABLE trial, and report what the search saw.

    A trial that fails the floor cannot win, because a constant predictor's score says nothing
    about the model — but it is still recorded, so the register shows how much of the space was
    usable for this window rather than only what survived.
    """
    import numpy as np
    import model_viability as MV

    space, rng = _space(), np.random.default_rng(seed)
    min_sn, min_sd = viability_floor()
    trials_log, best, best_params, best_rel = [], None, None, None
    for i in range(trials):
        absolute, relative = MV.draw(space, rng, H)
        r = MV.evaluate(dfx, dfb, tev, bnds, absolute, names_core, seed)
        if r is None:
            continue
        viable = r["split_nodes"] >= min_sn and r["pred_std"] >= min_sd
        trials_log.append({"trial": i, "viable": viable, "split_nodes": r["split_nodes"],
                           "pred_std": r["pred_std"], "oof_log_growth": r["oof_log_growth"],
                           "trades": r["trades"], "auc_pr": r["auc_pr"],
                           "gamma": round(absolute["gamma"], 6),
                           "gamma_rel": round(relative.get("gamma", 0.0), 8)})
        if viable and (best is None or r["oof_log_growth"] > best["oof_log_growth"]):
            best, best_params, best_rel = r, absolute, relative
    return {"params": best_params, "relative": best_rel, "baseline": best,
            "trials": trials_log,
            "viable_trials": sum(1 for t in trials_log if t["viable"]),
            "trials_run": len(trials_log)}


def fold_win(base, cur):
    pairs = [(b, c) for b, c in zip(base, cur) if b is not None and c is not None]
    return (sum(1 for b, c in pairs if c > b) / len(pairs)) if pairs else 0.0


def register_fold(dfx, dfb, tev, bnds_inner, ticker, cands, fam_of, seed=SEED):
    """The whole ladder for one outer-train window: core, every feature, every family, greedy."""
    import numpy as np
    import feature_search as FS
    import model_viability as MV
    import golden
    import pipeline as P

    core = FS._names(ticker, [])
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    inner = [j for j, x in enumerate(t0s) if x <= bnds_inner["train_end_idx"]]
    H = MV.hessian_total(y, w, np.asarray(inner))

    hpo = hpo_core(dfx, dfb, tev, bnds_inner, core, H, seed)
    if hpo["params"] is None:
        return {"stage": "no_viable_model", "hessian": round(H, 3),
                "viable_trials": 0, "trials_run": hpo["trials_run"]}

    params, base = hpo["params"], hpo["baseline"]
    min_sn, min_sd = viability_floor()

    def measure(names):
        r = MV.evaluate(dfx, dfb, tev, bnds_inner, params, names, seed)
        if r is None:
            return None
        r["viable"] = bool(r["split_nodes"] >= min_sn and r["pred_std"] >= min_sd)
        r["gain"] = round(r["oof_log_growth"] - base["oof_log_growth"], 8)
        r["fold_win"] = round(fold_win(base["fold_growth"], r["fold_growth"]), 6)
        return r

    def slim(r, **extra):
        # fold_growth is kept because the family policy computes its one-SE plateau from the
        # per-fold deltas, not from the mean — dropping it would silently change that rule.
        keep = ("gain", "fold_win", "viable", "split_nodes", "pred_std", "auc_pr",
                "oof_log_growth", "trades", "operating_point", "fold_growth")
        return {**{k: r[k] for k in keep}, **extra}

    singles = {}
    for cid in cands:
        r = measure(FS._names(ticker, [cid]))
        if r:
            singles[int(cid)] = slim(r, family=fam_of.get(cid))

    families = {}
    members = {}
    for cid, fam in fam_of.items():
        members.setdefault(fam, []).append(int(cid))
    for fam, ids in sorted(members.items()):
        r = measure(FS._names(ticker, ids))
        if r:
            best_member = max((i for i in ids if i in singles),
                              key=lambda i: singles[i]["gain"], default=None)
            families[fam] = slim(r, n_members=len(ids), members=sorted(ids),
                                 best_member=best_member,
                                 best_member_gain=(singles[best_member]["gain"]
                                                   if best_member is not None else None))

    # Greedy over the same gates the sealed search uses, on the survivors of the same prefilter.
    # The same prefilter the sealed search applies, with one condition added: a configuration
    # whose model could not differentiate has not tested its feature, so its gain is not evidence.
    survivors = [{"id": i, "mean": s["gain"],
                  "folds": [c - b for c, b in zip(s["fold_growth"], base["fold_growth"])]}
                 for i, s in singles.items()
                 if s["gain"] >= FS.MIN_FEATURE_GAIN and s["fold_win"] >= FS.MIN_FOLD_WIN_FRAC
                 and s["viable"]]

    state = {"selected": [], "best_pen": base["oof_log_growth"], "cur": base}
    greedy_log = []
    if survivors:
        simplicity = {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                      for ns, reg in P.FEATURE_REGISTRIES.items() if ns != "1h"
                      for f in reg["features"] if bool(f.get("implemented", True))}
        reps, fam_variants, _ = golden.family_stage_pool(survivors, fam_of, simplicity)

        def try_add(cid):
            r = measure(FS._names(ticker, state["selected"] + [cid]))
            if r is None:
                return False
            pen = r["oof_log_growth"] - FS.COMPLEXITY_PEN * (len(state["selected"]) + 1)
            ok = (pen > state["best_pen"] + 1e-6
                  and fold_win(state["cur"]["fold_growth"], r["fold_growth"]) >= FS.MIN_FOLD_WIN_FRAC
                  and r["viable"])
            greedy_log.append({"id": int(cid), "accepted": bool(ok),
                               "penalized": round(pen, 8), "viable": r["viable"]})
            if ok:
                state.update(selected=state["selected"] + [cid], best_pen=pen, cur=r)
            return ok

        accepted = []
        for cid in reps:
            if len(state["selected"]) >= FS.MAX_SELECT:
                break
            if try_add(cid):
                accepted.append(fam_of[cid])
        for fam in accepted:
            if len(state["selected"]) >= FS.MAX_SELECT:
                break
            for cid in fam_variants.get(fam, [])[:FS.FAMILY_CAP]:
                if len(state["selected"]) >= FS.MAX_SELECT:
                    break
                try_add(cid)

    fin = state["cur"]
    applied = bool(state["selected"]
                   and fin["oof_log_growth"] >= base["oof_log_growth"] + FS.MIN_GAIN)
    return {
        "stage": "auto_selected" if applied else "core_only",
        "hessian": round(H, 3),
        "viable_trials": hpo["viable_trials"], "trials_run": hpo["trials_run"],
        "frozen_params": {k: (round(v, 8) if isinstance(v, float) else v)
                          for k, v in params.items()},
        "gamma_rel": round(hpo["relative"].get("gamma", 0.0), 8),
        "core": slim(dict(base, gain=0.0, fold_win=0.0,
                          viable=base["split_nodes"] >= min_sn and base["pred_std"] >= min_sd)),
        "singles": singles, "families": families,
        "greedy": {"selected": [int(x) for x in (state["selected"] if applied else [])],
                   "penalized_gain": round(state["best_pen"] - base["oof_log_growth"], 8),
                   "n_features": len(state["selected"] if applied else []),
                   "log": greedy_log, "survivors": len(survivors)},
    }


def register_for(ticker, seed=SEED):
    import feature_search as FS
    import golden
    import nested_validation as NV
    import pipeline as P

    t0 = time.time()
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, FS.SCRATCH / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))   # ONE shared event set
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    fam_of = golden.load_families(FS.FAMILIES_PATH, cands)
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]

    folds = []
    for i, (tr, va, inner_end) in enumerate(NV.outer_folds(t0s, bnds)):
        r = register_fold(dfx, dfb, tev, dict(bnds, train_end_idx=int(inner_end)),
                          ticker, cands, fam_of, seed)
        r["outer_fold"] = i
        r["n_outer_train"], r["n_outer_val"] = len(tr), len(va)
        r["inner_train_end_idx"] = int(inner_end)
        folds.append(r)

    return {"ticker": ticker, "high_degeneracy_rate": ticker in HIGH_DEGENERACY,
            "outer_folds": len(folds), "seconds": round(time.time() - t0, 1), "folds": folds}


def _line(r, total, done):
    sel = [len(f.get("greedy", {}).get("selected", [])) for f in r["folds"]]
    vt = [f.get("viable_trials", 0) for f in r["folds"]]
    flag = "  [high_degeneracy_rate]" if r["high_degeneracy_rate"] else ""
    print(f"  [{done}/{total}] {r['ticker']:<6} wybranych/fold={sel}  "
          f"zdolnych trialów/fold={vt}  ({r['seconds']:.0f}s){flag}")


def diagnostics(path):
    """Add the SHAP column over the SAME 45 candidates the register measured.

    The earlier grid (scripts/shap_grid.py) ranked whatever the sealed model's manifest held —
    core plus the handful of optional features that ticker had selected — so it shares almost no
    features with this register, which measures the marginal utility of all 45 candidates. The
    ranks are only comparable when both are taken over the same set, so SHAP is recomputed here
    on a superset model carrying every candidate, at the parameters this register froze, out of
    fold. It remains a diagnostic: nothing selects on it.
    """
    import feature_search as FS
    import oof_shap as OS
    import pipeline as P
    import xgboost as xgb

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    for ticker, tab in doc["tables"].items():
        df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, FS.SCRATCH / f"{ticker}_1h.parquet")
        cands = FS.candidate_ids()
        rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
        dfb, bnds = rec["df_b"], rec["bounds"]
        names = FS._names(ticker, cands)
        for f in tab["folds"]:
            if f.get("stage") == "no_viable_model" or "frozen_params" not in f:
                continue
            bi = dict(bnds, train_end_idx=int(f["inner_train_end_idx"]))
            r = OS.rank_superset(P, xgb, dfb, bi, f["frozen_params"], names, SEED)
            f["shap"] = None if r is None else {"val_rows": r["val_rows"],
                                                "folds_used": r["folds_used"]}
        print(f"  {ticker}: SHAP na supersecie dołożony do {sum(1 for f in tab['folds'] if f.get('shap'))} foldów")
    Path(path).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--diagnostics", action="store_true",
                    help="augment an existing register with the SHAP column over the same candidates")
    args = ap.parse_args()

    if args.diagnostics:
        return diagnostics(args.out)

    tickers = args.tickers or json.loads(SAMPLE.read_text(encoding="utf-8"))["sample"]
    sn, sd = viability_floor()
    print(f"feature utility register — HPO v2 on core, frozen per outer fold; "
          f"viability floor split_nodes>={sn}, pred_std>={sd}\n  {len(tickers)} table(s), "
          f"{HPO_TRIALS} HPO trials, seed {SEED}\n")

    results = {}
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(register_for, tickers):
                results[r["ticker"]] = r
                _line(r, len(tickers), len(results))
    else:
        for i, t in enumerate(tickers, 1):
            r = register_for(t)
            results[t] = r
            _line(r, len(tickers), i)

    Path(args.out).write_text(json.dumps(
        {"hpo_trials": HPO_TRIALS, "seed": SEED,
         "viability": {"min_split_nodes": sn, "min_pred_std": sd},
         "operating_point": "quantile", "tables": results}, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
