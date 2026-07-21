#!/usr/bin/env python3
"""Stage 3 — does a choice made only on outer-train improve outer-validation over a capable core?

Everything this runs on was decided in stage 2 and is read back verbatim: the hyper-parameters
that stage's HPO froze for each outer window, and the subset its greedy selected there. Nothing
is tuned again. There is no HPO here, no feature selection, no threshold moved, no SHAP consulted,
and the sealed OOS window is not read — outer folds are carved out of Train exactly as before.

    outer-train  ->  frozen params  ->  frozen subset  ->  one refit  ->  outer-validation

Three configurations are carried, all through the identical treatment: the core alone, the greedy
subset, and — as a diagnostic midpoint, never as a candidate — the single feature that scored best
in that window's inner search. If the subset only matches its own best single feature, the greedy
step is buying nothing.

One value has to be recomputed rather than read: stage 2 stored the operating point for core, for
each single and for each family, but not for the final greedy subset. It is recovered by replaying
that subset over the same inner folds at the same frozen parameters and the same seed, which is
deterministic and reproduces what stage 2 itself computed. Recovering a number is not reselecting:
the subset is fixed before this file runs.

The operating point always comes from the inner folds and is applied to the outer fold as a fixed
quantile level. Letting op_select run on the outer fold would let the evaluation choose its own
threshold on the data it is judging.

    python3 scripts/nested_outer.py --jobs 3
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init  # noqa: E402,F401 — caps BLAS/OpenMP pools before anything numeric loads
runtime_init.apply()
sys.path.insert(0, str(XGB / "src"))
sys.path.insert(0, str(XGB / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

REGISTER = XGB / "data" / "feature_utility.json"
DEFAULT_OUT = XGB / "data" / "nested_outer.json"
MODE = "quantile"
SEED = 42


def best_single(fold):
    """The strongest single feature of that window's inner search — diagnostic, not a candidate."""
    ok = [(int(k), v) for k, v in fold.get("singles", {}).items() if v.get("viable")]
    if not ok:
        return None
    fid, v = max(ok, key=lambda kv: kv[1]["gain"])
    return {"id": fid, "inner_gain": v["gain"], "inner_fold_win": v["fold_win"]}


def evaluate_table(ticker):
    import feature_search as FS
    import model_viability as MV
    import nested_validation as NV
    import pipeline as P

    t0 = time.time()
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))["tables"][ticker]
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, FS.SCRATCH / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    outer = NV.outer_folds(t0s, bnds)

    out = []
    for f in reg["folds"]:
        i = f["outer_fold"]
        if i >= len(outer) or f.get("stage") == "no_viable_model":
            continue
        tr, va, inner_end = outer[i]
        bi = dict(bnds, train_end_idx=int(inner_end))
        params = f["frozen_params"]

        configs = {"core": []}
        bs = best_single(f)
        if bs:
            configs["best_single"] = [bs["id"]]
        if f["greedy"]["selected"]:
            configs["greedy"] = list(f["greedy"]["selected"])

        row = {"outer_fold": i, "n_outer_train": len(tr), "n_outer_val": len(va),
               "selected": list(f["greedy"]["selected"]),
               "inner_penalized_gain": f["greedy"]["penalized_gain"],
               "best_single": bs, "arms": {}}
        for name, ids in configs.items():
            names = FS._names(ticker, ids)
            inner = MV.evaluate(dfx, dfb, tev, bi, params, names, SEED, mode=MODE)
            if inner is None:
                continue
            q = float(inner["operating_point"])
            g, n, diag = NV.score_outer(dfx, dfb, tev, names, params, q, tr, va, SEED, MODE)
            row["arms"][name] = {"n_features": len(ids), "q_inner": q,
                                 "inner_log_growth": inner["oof_log_growth"],
                                 "outer_log_growth": g, "outer_trades": n,
                                 "outer_split_nodes": diag.get("split_nodes"),
                                 "outer_pred_std": diag.get("val_p_std"),
                                 "outer_cut": diag.get("cut"),
                                 "outer_share_fired": diag.get("val_share_above_cut")}
        a = row["arms"]
        if "core" in a and "greedy" in a and a["core"]["outer_log_growth"] is not None \
                and a["greedy"]["outer_log_growth"] is not None:
            row["delta"] = round(a["greedy"]["outer_log_growth"] - a["core"]["outer_log_growth"], 8)
            row["optimism_gap"] = round(row["inner_penalized_gain"] - row["delta"], 8)
            row["informative"] = bool(a["core"]["outer_trades"] > 0 or a["greedy"]["outer_trades"] > 0)
        else:
            row["delta"] = None
            row["optimism_gap"] = None
            row["informative"] = False
        out.append(row)

    return {"ticker": ticker, "high_degeneracy_rate": reg.get("high_degeneracy_rate", False),
            "seconds": round(time.time() - t0, 1), "folds": out}


def _line(r, total, done):
    d = [f["delta"] for f in r["folds"] if f["delta"] is not None]
    w = sum(1 for x in d if x > 0)
    flag = "  [high_degeneracy_rate]" if r["high_degeneracy_rate"] else ""
    print(f"  [{done}/{total}] {r['ticker']:<6} porównywalnych={len(d)}  wygrane={w}  "
          f"delty={[round(x, 3) for x in d]}  ({r['seconds']:.0f}s){flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not REGISTER.exists():
        sys.exit(f"stage 2 register missing: {REGISTER}")
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))
    tickers = args.tickers or list(reg["tables"])
    print(f"stage 3 — frozen params and frozen subsets replayed onto outer validation\n"
          f"  {len(tickers)} table(s), operating point: {MODE}, no HPO, no reselection\n")

    results = {}
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(evaluate_table, tickers):
                results[r["ticker"]] = r
                _line(r, len(tickers), len(results))
    else:
        for i, t in enumerate(tickers, 1):
            r = evaluate_table(t)
            results[t] = r
            _line(r, len(tickers), i)

    Path(args.out).write_text(json.dumps({"mode": MODE, "seed": SEED, "tables": results},
                                         indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
