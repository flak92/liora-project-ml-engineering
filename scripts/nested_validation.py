#!/usr/bin/env python3
"""Nested validation of the WHOLE selection procedure — HPO included, not held fixed.

The feature search reports a Train-OOF gain, and that number is the search's own objective:
it rises with how many subsets were tried, whether or not the chosen features generalise.
Reporting it as evidence of an edge measures search intensity. This runner asks the question
that number cannot answer — does the procedure that produced the subset generalise at all?

Per outer fold, on that fold's training part ONLY:

    inner HPO  ->  prefilter  ->  family policy  ->  greedy  ->  subset + operating point

and then one single evaluation on the outer validation fold, which the procedure never saw.

Why the HPO has to be inside. Freezing hyper-parameters tuned on the whole Train would leak
every outer validation fold into the procedure through the parameters, and the result would
measure only how well the *feature selector* generalises at pre-fixed settings. That is a
legitimate but different experiment, and it must then be called nested feature-selection
validation with fixed hyperparameters — not nested validation of the procedure.

The operating point is part of the procedure's output, so it is chosen on the inner folds and
applied to the outer fold as a fixed number. Letting op_select run on the outer fold would let
the evaluation pick its own theta on the data it is judging — the same leak one level down.
The core-only control arm goes through the identical treatment, so the comparison is paired.

Gates are imported from the search itself, never restated here: MIN_FEATURE_GAIN,
MIN_FOLD_WIN_FRAC, MIN_TRADES, COMPLEXITY_PEN, MAX_SELECT, FAMILY_CAP, MIN_GAIN, HPO_TRIALS.
Only the orchestration is written out, because search_ticker resolves its own data and cannot
be pointed at a sub-range; the loop below mirrors it step for step.

TRAIN ONLY. The OOS window is never read: outer folds are carved out of Train, and the purge
assertion inside eval_subset fires on every evaluation.

    python3 scripts/nested_validation.py NVDA                 # one table
    python3 scripts/nested_validation.py --jobs 3             # the whole sample
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

SAMPLE = ROOT / "config" / "sample_20.json"
DEFAULT_OUT = XGB / "data" / "nested_validation.json"
K_OUT = 4

# The operating point, in two mechanisms that differ in one thing only.
#
#   absolute  the probability itself is the cut, over OPERATING_SPACE.theta_scoring. This is what
#             the sealed pipeline does, and it is why 32 of 80 outer folds traded nothing: the
#             model's predictions concentrate around the base rate and their spread narrows as the
#             training set grows (NVDA: max 0.443 -> 0.422 -> 0.412 -> 0.407), so a cut fixed on
#             the smaller inner folds can sit above everything the outer model produces.
#
#   quantile  the rule is "fire the top q of signals", and the cut is that quantile of the model's
#             predictions ON ITS OWN TRAINING ROWS. Measured against the alternatives on MTD's
#             first outer fold: the inner-OOF distribution puts q85 at 0.457 while the outer fold
#             wants 0.512, because models trained on less data predict wider; the model's own
#             training distribution puts it at 0.516. Causal — the cut exists before the window is
#             traded — and it needs nothing the procedure is not allowed to see.
#
# Declared before the run and not tuned afterwards. Six points, matching the six the absolute
# grid has, spanning "top quarter" to "top hundredth".
Q_GRID = [0.75, 0.82, 0.88, 0.92, 0.96, 0.99]


def _need_research_tree():
    missing = [p for p in (XGB / "src" / "pipeline.py", XGB / "tools" / "feature_search.py")
               if not p.exists()]
    if missing:
        sys.exit("research tree absent — missing " + ", ".join(str(p) for p in missing))


def outer_folds(t0s, bounds, k=K_OUT):
    """Walk-forward outer split of Train, purged and embargoed exactly like the inner one.

    Returns (outer_train_idx, outer_val_idx, inner_train_end_idx) per fold. The last element is
    where the inner folds must stop so that nothing inside the procedure can touch the outer
    validation window, not even through a label horizon.
    """
    import numpy as np
    import pipeline as P

    H, emb = P.PIPELINE_PARAMETERS["H"], P.embargo_candles()
    ts, te = bounds["train_start_idx"], bounds["train_end_idx"]
    edges = np.linspace(ts, te + 1, k + 2, dtype=int)
    out = []
    for i in range(1, k + 1):
        vlo, vhi = int(edges[i]), int(edges[i + 1])
        va = [j for j, x in enumerate(t0s) if vlo <= x < vhi]
        tr = [j for j, x in enumerate(t0s) if x + H < vlo - emb]
        if len(va) >= 5 and len(tr) >= 10:
            out.append((tr, va, vlo - emb - H - 1))
    return out


def _fold_matrices(dfb, tev, names):
    import numpy as np
    X = dfb[names].to_numpy(float)
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    ticker = str(dfb["asset_id"].iloc[0])
    by_sid = {f"{ticker}:{s['t0']}:{s['direction']}": s for s in tev}
    return X, y, w, t0s, by_sid


def choose_operating_point(P, dfx, fold_data, mode):
    """Pick ONE operating point on the accumulated folds — never per fold (the fold-oracle bias).

    Both mechanisms end in the same op_select call, so the plateau rule, the trade floor and the
    deterministic tie-break are shared; only what the grid ranges over differs. The quantile grid
    carries q in the "theta" slot because op_select orders and detects boundaries by that key, and
    q is monotone in selectivity exactly as theta is.
    """
    import math
    import numpy as np
    import op_select

    os_ = P.operating_space()
    if mode == "absolute":
        return P.score_shared_operating_point(dfx, [(sc, lo, hi) for sc, lo, hi, _ in fold_data])

    E0 = P.PIPELINE_PARAMETERS["INITIAL_CAPITAL_USD"]
    grid = []
    for q in Q_GRID:
        gs, ns, cuts = [], [], []
        for sc, lo, hi, p_train in fold_data:
            cut = float(np.quantile(p_train, q))
            summ = P.run_engine(dfx, sc, lo, hi, cut, kelly_fraction=None)[0]
            gs.append(math.log(max(summ["end_capital"], P.EPS) / E0))
            ns.append(int(summ["trades"]))
            cuts.append(cut)
        grid.append({"theta": float(q), "lambda": None, "fold_growth": gs,
                     "fold_trades": ns, "cuts": cuts})
    return op_select.select_operating_point(grid, min_oof_trades=int(os_["min_oof_trades"]),
                                            theta_spectrum=[float(q) for q in Q_GRID])


def eval_subset(dfx, dfb, tev, bnds, params, names, seed, mode):
    """FS.eval_subset, with the operating-point mechanism made a parameter.

    Rewritten here rather than imported because the sealed version hardcodes the absolute
    mechanism, and changing it there would move HPO, the search and the calibration at once. The
    body is otherwise line for line the same — including the purge assertion, which is the reason
    it is repeated rather than trusted. The one addition is the model's predictions on its own
    training rows, which the quantile mechanism needs and the absolute one ignores.
    """
    import numpy as np
    import xgboost as xgb
    import pipeline as P

    X, y, w, t0s, by_sid = _fold_matrices(dfb, tev, names)
    folds = P.purged_wf_folds(t0s, bnds["train_start_idx"], bnds["train_end_idx"])
    assert all(max(t0s[j] for j in va) < bnds["oos_start_idx"] for _, va in folds if va), \
        "nested eval_subset: a CV val fold reaches OOS (purge invariant violated)"

    growth = [None] * len(folds)
    valid, fold_data = [], []
    for pos, (tr, va) in enumerate(folds):
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
        p = bst.predict(xgb.DMatrix(X[va], feature_names=names))
        p_train = bst.predict(xgb.DMatrix(X[tr], feature_names=names))
        scored = [(by_sid[dfb["setup_id"].iloc[j]], float(p[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        fold_data.append((scored, min(vt0), max(vt0), p_train))
        valid.append(pos)
    if not fold_data:
        return growth, 0, None

    sel = choose_operating_point(P, dfx, fold_data, mode)
    for k, pos in enumerate(valid):
        growth[pos] = float(sel["fold_growth"][k])
    return growth, int(sel["trades"]), sel


def run_procedure(dfx, dfb, tev, bnds_inner, ticker, cands, seed, mode):
    """The search, run end to end on one outer fold's training part. Mirrors search_ticker."""
    import feature_search as FS
    import golden
    import pipeline as P

    best_params, _ap, _nf = P.layer7_optuna(dfx, dfb, tev, bnds_inner, seed,
                                            FS._manifest(ticker, cands), trials=FS.HPO_TRIALS)
    base_folds, base_trades, _ = eval_subset(dfx, dfb, tev, bnds_inner, best_params,
                                             FS._names(ticker, []), seed, mode)
    base_mean = FS._fmean(base_folds)
    if base_mean is None or base_trades < FS.MIN_TRADES:
        return {"selected": [], "params": best_params, "base_mean": base_mean,
                "stage": "thin_no_trades" if base_mean is not None else "no_valid_folds"}

    survivors = []
    for cid in cands:
        folds, _, _ = eval_subset(dfx, dfb, tev, bnds_inner, best_params,
                                  FS._names(ticker, [cid]), seed, mode)
        mean = FS._fmean(folds)
        if mean is None:
            continue
        gain, wf = mean - base_mean, FS._fold_win_frac(base_folds, folds)
        if gain >= FS.MIN_FEATURE_GAIN and wf >= FS.MIN_FOLD_WIN_FRAC:
            survivors.append({"id": cid, "mean": gain,
                              "folds": [(c - b) if (c is not None and b is not None) else None
                                        for c, b in zip(folds, base_folds)]})

    fam_of = golden.load_families(FS.FAMILIES_PATH, cands)
    simplicity = {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                  for ns, reg in P.FEATURE_REGISTRIES.items() if ns != "1h"
                  for f in reg["features"] if bool(f.get("implemented", True))}
    reps, fam_variants, _rep = (golden.family_stage_pool(survivors, fam_of, simplicity)
                                if survivors else ([], {}, {}))

    state = {"selected": [], "best_pen": base_mean - 0.0, "cur_folds": base_folds}

    def try_add(cid):
        folds, _, _ = eval_subset(dfx, dfb, tev, bnds_inner, best_params,
                                  FS._names(ticker, state["selected"] + [cid]), seed, mode)
        mean = FS._fmean(folds)
        if mean is None:
            return False
        pen = mean - FS.COMPLEXITY_PEN * (len(state["selected"]) + 1)
        if pen > state["best_pen"] + 1e-6 and \
                FS._fold_win_frac(state["cur_folds"], folds) >= FS.MIN_FOLD_WIN_FRAC:
            state.update(selected=state["selected"] + [cid], best_pen=pen, cur_folds=folds)
            return True
        return False

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

    fin_mean = FS._fmean(state["cur_folds"]) if state["selected"] else base_mean
    applied = bool(state["selected"]) and fin_mean is not None and \
        (fin_mean >= base_mean + FS.MIN_GAIN)
    return {"selected": state["selected"] if applied else [], "params": best_params,
            "base_mean": base_mean, "inner_mean": fin_mean,
            "stage": "auto_selected" if applied else "core_only"}


def operating_point_on_inner(dfx, dfb, tev, bnds_inner, params, names, seed, mode):
    """The operating point the procedure would ship — chosen on inner folds, nothing else.

    Returns an absolute probability in the absolute arm and a quantile level in the quantile arm.
    Both come out of the same op_select accumulation, so neither arm gets a per-fold best.
    """
    _g, _n, sel = eval_subset(dfx, dfb, tev, bnds_inner, params, names, seed, mode)
    return None if sel is None else float(sel["theta"])


def score_outer(dfx, dfb, tev, names, params, point, tr, va, seed, mode):
    """One evaluation on the untouched outer fold, at the point fixed by the inner run.

    In the quantile arm `point` is a level, and the cut it becomes is read off the model's own
    training rows — never off `va`. Reading it off `va` would let the evaluation set its own
    threshold from the data it is judging, which is the leak this whole runner exists to avoid.

    Also returns what the fold needs to be interpreted rather than merely scored: how many split
    nodes the model actually has, and the spread of its predictions. A booster with no splits is a
    constant, and no threshold mechanism of any kind can select within a constant — such folds say
    nothing about either arm and must be counted separately.
    """
    import numpy as np
    import xgboost as xgb
    import pipeline as P

    X, y, w, t0s, by_sid = _fold_matrices(dfb, tev, names)
    if len(np.unique(y[tr])) < 2:
        return None, 0, {}
    bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
    p = bst.predict(xgb.DMatrix(X[va], feature_names=names))
    p_train = bst.predict(xgb.DMatrix(X[tr], feature_names=names))
    cut = float(point) if mode == "absolute" else float(np.quantile(p_train, float(point)))

    td = bst.trees_to_dataframe()
    diag = {"split_nodes": int((td["Feature"] != "Leaf").sum()),
            "trees": int(td["Tree"].nunique()),
            "cut": round(cut, 6),
            "val_p_min": round(float(p.min()), 6), "val_p_max": round(float(p.max()), 6),
            "val_p_std": round(float(p.std()), 8),
            "val_share_above_cut": round(float((p >= cut).mean()), 6),
            "train_p_max": round(float(p_train.max()), 6),
            "train_p_std": round(float(p_train.std()), 8)}

    scored = [(by_sid[dfb["setup_id"].iloc[j]], float(p[k])) for k, j in enumerate(va)]
    vt0 = [t0s[j] for j in va]
    # op_grid_scores returns the raw per-fold vectors; log_growth and trades are added later by
    # op_select's accumulation step, which is deliberately NOT called here — selecting anything
    # on this fold would be the evaluation choosing its own parameter on the data it judges.
    grid = P.op_grid_scores(dfx, [(scored, min(vt0), max(vt0))], [cut], P.op_lambdas())
    if not grid:
        return None, 0, diag
    g = grid[0]
    return float(sum(g["fold_growth"])), int(sum(g["fold_trades"])), diag


def nested_table(ticker, seed=42, mode="absolute"):
    import feature_search as FS
    import pipeline as P

    t0 = time.time()
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, FS.SCRATCH / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]

    folds, out = outer_folds(t0s, bnds), []
    for i, (tr, va, inner_end) in enumerate(folds):
        bnds_inner = dict(bnds, train_end_idx=int(inner_end))
        proc = run_procedure(dfx, dfb, tev, bnds_inner, ticker, cands, seed, mode)
        core = FS._names(ticker, [])
        picked = FS._names(ticker, proc["selected"]) if proc["selected"] else core

        th_core = operating_point_on_inner(dfx, dfb, tev, bnds_inner, proc["params"], core, seed, mode)
        th_pick = (operating_point_on_inner(dfx, dfb, tev, bnds_inner, proc["params"], picked, seed, mode)
                   if proc["selected"] else th_core)
        g_core, n_core, d_core = ((None, 0, {}) if th_core is None else
                                  score_outer(dfx, dfb, tev, core, proc["params"], th_core,
                                              tr, va, seed, mode))
        g_pick, n_pick, d_pick = ((None, 0, {}) if th_pick is None else
                                  score_outer(dfx, dfb, tev, picked, proc["params"], th_pick,
                                              tr, va, seed, mode))

        # Each arm carries its own operating point, because the procedure's output is the pair
        # (features, theta) — that is the honest end-to-end comparison. It does conflate two
        # effects: on one probe the selected arm lost 0.456 while trading 4 times against the
        # core's 178, almost all of it the theta moving rather than the features. So the subset
        # is scored a second time at the CORE's theta, which isolates what the features did.
        # Diagnostic only: this theta is not what the procedure would have shipped.
        g_pick_ct, n_pick_ct, _ = ((None, 0, {}) if (th_core is None or not proc["selected"]) else
                                   score_outer(dfx, dfb, tev, picked, proc["params"], th_core,
                                               tr, va, seed, mode))

        out.append({"outer_fold": i, "stage": proc["stage"],
                    "selected": [int(x) for x in proc["selected"]],
                    "n_outer_train": len(tr), "n_outer_val": len(va),
                    "theta_core": th_core, "theta_selected": th_pick,
                    "outer_core": g_core, "outer_selected": g_pick,
                    "outer_trades_core": n_core, "outer_trades_selected": n_pick,
                    "outer_selected_at_core_theta": g_pick_ct,
                    "outer_trades_selected_at_core_theta": n_pick_ct,
                    "delta": (None if (g_core is None or g_pick is None) else round(g_pick - g_core, 8)),
                    "delta_features_only": (None if (g_core is None or g_pick_ct is None)
                                            else round(g_pick_ct - g_core, 8)),
                    # A fold where neither arm traded carries no information; it must not be
                    # counted as a tie when the win rate is computed.
                    "informative": bool(proc["selected"] and (n_core > 0 or n_pick > 0)),
                    "diag_core": d_core, "diag_selected": d_pick,
                    # A booster with no split nodes is a constant: it cannot depend on any feature,
                    # so neither threshold mechanism has anything to select within.
                    "degenerate": bool(d_core.get("split_nodes", 0) == 0),
                    "inner_gain": (None if proc.get("inner_mean") is None or proc.get("base_mean") is None
                                   else round(proc["inner_mean"] - proc["base_mean"], 8))})
    return {"ticker": ticker, "mode": mode, "outer_folds": len(folds),
            "seconds": round(time.time() - t0, 1), "folds": out}


def main():
    _need_research_tree()
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--mode", choices=("absolute", "quantile"), default="absolute",
                    help="operating-point mechanism; everything else is identical between the two")
    args = ap.parse_args()

    tickers = args.tickers or json.loads(SAMPLE.read_text(encoding="utf-8"))["sample"]
    grid = ("OPERATING_SPACE.theta_scoring" if args.mode == "absolute" else f"q in {Q_GRID}")
    print(f"nested validation of the whole procedure (inner HPO per outer fold), "
          f"k_out={K_OUT}, {len(tickers)} table(s)\n  operating point: {args.mode} — {grid}\n")

    from functools import partial
    run = partial(nested_table, mode=args.mode)
    results = {}
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(run, tickers):
                results[r["ticker"]] = r
                _line(r, len(tickers), len(results))
    else:
        for i, t in enumerate(tickers, 1):
            r = run(t)
            results[t] = r
            _line(r, len(tickers), i)

    Path(args.out).write_text(json.dumps({"k_out": K_OUT, "mode": args.mode,
                                          "q_grid": Q_GRID if args.mode == "quantile" else None,
                                          "tables": results}, indent=1) + "\n",
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


def _line(r, total, done):
    d = [f["delta"] for f in r["folds"] if f["delta"] is not None]
    wins = sum(1 for x in d if x > 0)
    print(f"  [{done}/{total}] {r['ticker']:<6} outer={r['outer_folds']} "
          f"wins={wins}/{len(d)} "
          f"delty={[round(x, 3) for x in d]} ({r['seconds']:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
