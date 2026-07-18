#!/usr/bin/env python3
"""Train-CV measurement for the XGB pipeline (development only — NEVER reads OOS).

The XGB model itself (training / HPO / trade engine) lives in pipeline.py, layers L7-L8;
this file is the Train-CV measurement harness around it.

Reproduces the notebook's L1-L7/L8 up to the operating point (load bars -> derive_output_b [L4-L6
features + Train candidates + Triple-Barrier labels, purged] -> L7 profit HPO -> L8 Kelly), then
reports the Train-side success metrics used to A/B strategy changes:

  train_cv_oof_log_growth : Sum_fold log(end_capital/E0) at (THRESHOLD_ENTRY, calibrated lambda) over
                            the purged walk-forward OOF folds — the geometric-growth objective.
  train_cv_pf             : profit factor over the concatenated out-of-fold trade ledger.

Everything is on the Train window; the OOS window is never generated here. Deterministic.
  python3 src/xgb/train_cv_eval.py TICKER=AAPL
  python3 src/xgb/train_cv_eval.py AAPL KO XOM
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P


def train_cv_metrics(ticker):
    import xgboost as xgb
    P.seed_everything()
    SEED = int(P.PIPELINE_PARAMETERS["RANDOM_SEED"])
    manifest = P.resolve_feature_manifest(ticker)
    with tempfile.TemporaryDirectory() as td:
        df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, str(Path(td) / "b.parquet"))
    REC = P.derive_output_b(df, ticker, manifest)
    df_b, tev, bounds = REC["df_b"], REC["train_events"], REC["bounds"]
    n_candidates = REC["audit"]["candidates"]["candidates"]
    if len(df_b) < 50:
        return {"ticker": ticker, "thin": True, "n_events": int(len(df_b)), "n_candidates": int(n_candidates)}

    best, cv_ap, n_folds = P.layer7_optuna(REC["df"], df_b, tev, bounds, SEED, manifest)
    cal = P.calibrate_kelly(REC["df"], df_b, tev, bounds, best, SEED, manifest)
    kf, theta = cal["kelly_fraction"], cal["theta_entry"]

    # Train-CV PF/log-growth at the deployed operating point (calibrated theta + lambda), replayed on
    # each fold's out-of-fold predictions — strictly inside Train (folds are purged pre-OOS).
    E0 = P.PIPELINE_PARAMETERS["INITIAL_CAPITAL_USD"]
    names = P.feature_names_of(manifest)
    X = df_b[names].to_numpy(float)
    y = df_b["Y_outcome"].to_numpy(int)
    w = df_b["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(sid.split(":")[1]) for sid in df_b["setup_id"]]
    folds = P.purged_wf_folds(t0s, bounds["train_start_idx"], bounds["train_end_idx"])
    tk = str(df_b["asset_id"].iloc[0])
    by_sid = {f"{tk}:{s['t0']}:{s['direction']}": s for s in tev}
    nets, lg, rets = [], 0.0, []
    for tr, va in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], best, SEED, feature_names=names)
        oof = bst.predict(xgb.DMatrix(X[va], feature_names=names))
        scored = [(by_sid[df_b["setup_id"].iloc[j]], float(oof[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        summ, ledger, _ = P.run_engine(REC["df"], scored, min(vt0), max(vt0), theta, kelly_fraction=kf)
        lg += math.log(max(summ["end_capital"], P.EPS) / E0)
        rets.append(summ["return_pct"])
        nets += [t["account_net_pnl_usd"] for t in ledger]
    nets = np.array(nets)
    gp = float(nets[nets > 0].sum()) if len(nets) else 0.0
    gl = float(-nets[nets < 0].sum()) if len(nets) else 0.0
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    return {"ticker": ticker, "thin": False, "n_candidates": int(n_candidates), "n_events": int(len(df_b)),
            "n_folds": int(n_folds), "train_cv_oof_log_growth": round(float(lg), 6),
            "train_cv_pf": round(float(pf), 4) if math.isfinite(pf) else None,
            "train_cv_return_pct": round(float(np.mean(rets)), 4) if rets else 0.0,
            "train_cv_trades": int(len(nets)), "cv_auc_pr": round(float(cv_ap), 4),
            "theta": round(float(theta), 3), "kelly": round(float(kf), 4),
            "reward_risk_b": round(float(P.PIPELINE_PARAMETERS["TB_ATR_TP"]) / float(P.PIPELINE_PARAMETERS["TB_ATR_SL"]), 3)}


def main():
    args = [a.split("=", 1)[1] if a.startswith("TICKER=") else a for a in sys.argv[1:]]
    for t in ([x.upper() for x in args] or ["AAPL"]):
        try:
            print(json.dumps(train_cv_metrics(t)), flush=True)
        except Exception as e:
            print(json.dumps({"ticker": t, "error": repr(e)[:200]}), flush=True)


if __name__ == "__main__":
    main()
