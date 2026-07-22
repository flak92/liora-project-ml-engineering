#!/usr/bin/env python3
"""Can the model learn at all? — a register of what each hyper-parameter draw actually produces.

The nested validation found 33 of 80 outer folds whose booster had NOT ONE split node, and a
median of four across all of them. A tree with no splits is a constant: it cannot use a feature,
so comparing feature subsets under it compares nothing. Before asking again whether the 45
candidates carry signal, the model has to be capable of testing them.

This script does not tune anything towards a better result. It records, for every trial, what
that draw did to the model — split nodes, leaves, prediction spread, AUC-PR, out-of-fold trading
log-growth, trades — on the CORE feature set only, so nothing here depends on which optional
features exist. The register is the deliverable; the viability thresholds are read off it
afterwards and declared before they are used.

Why the space is expressed relative to the hessian. For binary:logistic, `gamma` and
`min_child_weight` are thresholds in units of summed hessian, where a row contributes
w_i * p(1-p). The label_uniqueness_weight shrinks the effective sample about eighteen-fold —
9629 rows carry 524.7 of weight — so a fold's total hessian lands near 50-100 where unweighted
data would give ~1400. The sealed space samples gamma uniformly from [0, 5]; measured on MTD at
hessian 75, gamma 1.0 leaves 29 split nodes and gamma 5.0 leaves one. Scaling by the fold's own
hessian makes one number mean one thing on every asset — which is the point: the space calibrates
itself from each asset's data instead of being imposed on all of them alike.

The sealed space and the sealed pipeline are never touched. This runner builds its own trial loop
out of P._xgb_train and P.purged_wf_folds rather than rebinding anything inside pipeline.py.

TRAIN ONLY — the purge assertion fires on every trial.

    python3 scripts/model_viability.py NVDA --trials 30
    python3 scripts/model_viability.py --jobs 3 --trials 30      # the whole sample, both spaces
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
DATA_DIR = Path(os.environ.get("LIORA_RESEARCH_DATA_DIR") or str(XGB / "data"))  # run-scoped przez engine, domyślnie kanoniczne
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init  # noqa: E402,F401 — caps BLAS/OpenMP pools before anything numeric loads
runtime_init.apply()
sys.path.insert(0, str(XGB / "src"))
sys.path.insert(0, str(XGB / "tools"))

SAMPLE = ROOT / "config" / "sample_20.json"
SPACE_V2 = ROOT / "config" / "xgb_search_space_v2.json"
SPACE_V1 = ROOT / "config" / "xgboost_optuna_search_space.json"
DEFAULT_OUT = DATA_DIR / "model_viability.json"


def hessian_total(y, w, idx):
    """Sum of hessians at the base rate — the scale gamma and min_child_weight are measured in.

    Computed at p = base rate rather than per boosting round: it is the scale the FIRST split
    faces, and the first split is the one that either happens or does not.
    """
    p = float(y[idx].mean())
    return float(w[idx].sum() * p * (1.0 - p))


def draw(space, rng, H):
    """One parameter draw. Hessian-relative entries are multiplied out here, so what reaches
    XGBoost is always absolute and what is recorded is both."""
    rel = set(space.get("hessian_relative", []))
    absolute, relative = {}, {}
    for sp in space["parameters"]:
        n, lo, hi = sp["name"], sp["low"], sp["high"]
        if sp["suggest"] == "int":
            v = int(rng.integers(int(lo), int(hi) + 1))
        elif sp.get("log"):
            import math
            v = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        else:
            v = float(rng.uniform(lo, hi))
        if n in rel:
            relative[n] = v
            v = v * H
        absolute[n] = v
    return absolute, relative


def evaluate(dfx, dfb, tev, bnds, params, names, seed, mode="quantile"):
    """One trial across the purged folds: what the model became, and what it scored."""
    import numpy as np
    import xgboost as xgb
    import pipeline as P
    sys.path.insert(0, str(ROOT / "scripts"))
    import nested_validation as NV

    X = dfb[names].to_numpy(float)
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    ticker = str(dfb["asset_id"].iloc[0])
    by_sid = {f"{ticker}:{s['t0']}:{s['direction']}": s for s in tev}
    folds = P.purged_wf_folds(t0s, bnds["train_start_idx"], bnds["train_end_idx"])
    assert all(max(t0s[j] for j in va) < bnds["oos_start_idx"] for _, va in folds if va), \
        "model_viability: a CV val fold reaches OOS (purge invariant violated)"

    from sklearn.metrics import average_precision_score
    splits, leaves, stds, aps, fold_data = [], [], [], [], []
    pmin, pmax = 1.0, 0.0
    for tr, va in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
        td = bst.trees_to_dataframe()
        splits.append(int((td["Feature"] != "Leaf").sum()))
        leaves.append(int((td["Feature"] == "Leaf").sum()))
        p = bst.predict(xgb.DMatrix(X[va], feature_names=names))
        p_train = bst.predict(xgb.DMatrix(X[tr], feature_names=names))
        stds.append(float(p.std()))
        pmin, pmax = min(pmin, float(p.min())), max(pmax, float(p.max()))
        if len(np.unique(y[va])) > 1:
            aps.append(float(average_precision_score(y[va], p)))
        scored = [(by_sid[dfb["setup_id"].iloc[j]], float(p[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        fold_data.append((scored, min(vt0), max(vt0), p_train))

    if not fold_data:
        return None
    sel = NV.choose_operating_point(P, dfx, fold_data, mode)
    return {"split_nodes": int(sum(splits)), "split_nodes_per_fold": splits,
            "fold_growth": [float(g) for g in sel["fold_growth"]],
            "fold_trades": [int(n) for n in sel["fold_trades"]],
            "leaves": int(sum(leaves)),
            "pred_std": round(float(sum(stds) / len(stds)), 6),
            "pred_min": round(pmin, 6), "pred_max": round(pmax, 6),
            "auc_pr": round(float(sum(aps) / len(aps)), 6) if aps else None,
            "oof_log_growth": round(float(sum(sel["fold_growth"])), 6),
            "trades": int(sel["trades"]), "operating_point": float(sel["theta"]),
            "folds_used": len(fold_data)}


def register_for(ticker, trials=30, seed=42):
    """Both spaces, the same draws' worth of budget, on the same events and folds."""
    import numpy as np
    import feature_search as FS
    import pipeline as P

    t0 = time.time()
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, FS.SCRATCH / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    names = FS._names(ticker, [])                      # CORE ONLY — no optional feature involved
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    H = hessian_total(y, w, np.arange(len(dfb)))

    out = {"ticker": ticker, "hessian": round(H, 3), "rows": int(len(dfb)),
           "weight_sum": round(float(w.sum()), 3), "core_features": len(names), "spaces": {}}
    for label, path in (("v1_absolute", SPACE_V1), ("v2_hessian_relative", SPACE_V2)):
        space = json.loads(path.read_text(encoding="utf-8"))
        rng = np.random.default_rng(seed)
        recs = []
        for i in range(trials):
            absolute, relative = draw(space, rng, H)
            r = evaluate(dfx, dfb, tev, bnds, absolute, names, seed)
            if r is None:
                continue
            r["trial"] = i
            r["gamma"] = round(absolute["gamma"], 6)
            r["min_child_weight"] = round(absolute["min_child_weight"], 6)
            r["gamma_rel"] = round(relative.get("gamma", absolute["gamma"] / H), 8)
            recs.append(r)
        out["spaces"][label] = recs
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _line(r, total, done):
    for label, recs in r["spaces"].items():
        sn = sorted(x["split_nodes"] for x in recs)
        dead = sum(1 for x in sn if x == 0)
        med = sn[len(sn) // 2] if sn else 0
        print(f"  [{done}/{total}] {r['ticker']:<6} H={r['hessian']:>6.1f}  {label:<20} "
              f"trials={len(recs):>3}  bez podziałów={dead:>3}  mediana węzłów={med:>5}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    tickers = args.tickers or json.loads(SAMPLE.read_text(encoding="utf-8"))["sample"]
    print(f"model viability register — core features only, {args.trials} draws per space, "
          f"{len(tickers)} table(s)\n")

    from functools import partial
    run = partial(register_for, trials=args.trials)
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

    Path(args.out).write_text(json.dumps({"trials": args.trials, "tables": results}, indent=1) + "\n",
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
