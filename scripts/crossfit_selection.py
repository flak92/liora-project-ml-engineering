#!/usr/bin/env python3
"""Stage 3B — pick on folds you do not judge on.

Stage 3 located the failure precisely: the procedure takes the maximum of 45 noisy estimates of
marginal gain and then scores that maximum on the very folds that produced it. Inner promised
+0.1378, outer delivered -0.0337, and the best single feature did no better than the greedy
subset — so the damage is done at the moment of picking a winner, not when features are combined.

The fix splits the inner folds by role. Within one outer-train window:

    discovery folds  ->  rank the candidates  ->  pick one
    confirmation fold ->  measure that pick against core, having played no part in choosing it

The confirmation fold rotates, so each inner fold serves once as untouched validation. A feature
is judged by what confirmation says, never by the score that selected it.

Two arms, both declared before this ran, neither a repair fitted to stage 3's results:

    flat          45 candidates -> top single feature -> confirmation
    hierarchical  12 families -> strongest family -> its representative -> confirmation

The hierarchical arm exists because stage 3 hinted that survival differs by family, and the only
honest way to use a hint is to test it as its own arm rather than to fold it into the rule.
Stability there is judged on the FAMILY first: the same OHLCV relationship recurring across
rotations counts, even if a different variant of it wins each time.

The operating point always comes from the discovery folds and is applied to confirmation as a
fixed quantile level. Choosing it on the confirmation fold would reintroduce, one level down,
exactly the leak this file exists to close.

Nothing else moves: the 45 candidates, the families, the frozen HPO v2 parameters from stage 2,
the viability floor, the quantile operating point, the outer folds and the execution contract are
all read back unchanged. The sealed OOS window is never touched.

An empty subset is a correct outcome. This does not pick the best available feature; it picks
only a feature that survived being chosen and judged by different data.

    python3 scripts/crossfit_selection.py --jobs 3
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
sys.path.insert(0, str(ROOT / "scripts"))

REGISTER = DATA_DIR / "feature_utility.json"
DEFAULT_OUT = DATA_DIR / "crossfit_selection.json"
MODE = "quantile"
SEED = int(os.environ.get("RESEARCH_SEED", "42"))   # run-scoped przez engine (dispatch), domyślnie 42

# The acceptance contract now lives in `acceptance.py`, imported here and by the null runner so
# that both sides of the permutation test compute the statistic with literally the same code. Every
# clause must hold; failing any of them means the evidence is insufficient, and insufficient
# evidence returns nothing rather than the best of what was on offer.
import acceptance as ACC                                                   # noqa: E402

MIN_ROTATIONS = ACC.MIN_ROTATIONS       # picked in at least 2 of the 4 rotations (which overlap in
                                        # their discovery sets — "at least 2", not "independent")
MIN_MEDIAN_DELTA = ACC.COMPLEXITY_PEN   # the same complexity cost the search has always charged
MAJORITY = ACC.MAJORITY                 # strictly more than half the comparable confirmations positive


def _fold_data(dfx, dfb, tev, folds, params, names, seed):
    """Train on each fold's training rows, score its validation rows. Nothing global is touched."""
    import numpy as np
    import xgboost as xgb
    import pipeline as P

    X = dfb[names].to_numpy(float)
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    ticker = str(dfb["asset_id"].iloc[0])
    by_sid = {f"{ticker}:{s['t0']}:{s['direction']}": s for s in tev}
    out = []
    for tr, va in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
        p = bst.predict(xgb.DMatrix(X[va], feature_names=names))
        p_train = bst.predict(xgb.DMatrix(X[tr], feature_names=names))
        scored = [(by_sid[dfb["setup_id"].iloc[j]], float(p[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        out.append((scored, min(vt0), max(vt0), p_train))
    return out


def discover(dfx, dfb, tev, folds, params, names, seed):
    """Score a configuration on the discovery folds and return its per-fold growth and its q."""
    import nested_validation as NV
    import pipeline as P

    fd = _fold_data(dfx, dfb, tev, folds, params, names, seed)
    if not fd:
        return None
    sel = NV.choose_operating_point(P, dfx, fd, MODE)
    return {"fold_growth": [float(g) for g in sel["fold_growth"]],
            "mean": float(sum(sel["fold_growth"]) / len(sel["fold_growth"])),
            "q": float(sel["theta"]), "trades": int(sel["trades"])}


def confirm(dfx, dfb, tev, fold, params, names, seed, q):
    """One measurement on the confirmation fold at the operating point discovery had already fixed."""
    import math
    import numpy as np
    import pipeline as P

    fd = _fold_data(dfx, dfb, tev, [fold], params, names, seed)
    if not fd:
        return None
    scored, lo, hi, p_train = fd[0]
    cut = float(np.quantile(p_train, q))
    summ = P.run_engine(dfx, scored, lo, hi, cut, kelly_fraction=None)[0]
    E0 = P.PIPELINE_PARAMETERS["INITIAL_CAPITAL_USD"]
    return {"growth": math.log(max(summ["end_capital"], P.EPS) / E0),
            "trades": int(summ["trades"]), "cut": round(cut, 6)}


def rotation(dfx, dfb, tev, inner, r, params, ticker, cands, fam_of, simplicity, seed):
    """One rotation: fold r is held out for confirmation, the rest discover."""
    import feature_search as FS
    import golden

    disc = [f for i, f in enumerate(inner) if i != r]
    conf = inner[r]
    base = discover(dfx, dfb, tev, disc, params, FS._names(ticker, []), seed)
    if base is None:
        return None
    base_conf = confirm(dfx, dfb, tev, conf, params, FS._names(ticker, []), seed, base["q"])
    if base_conf is None:
        return None

    scores = {}
    for cid in cands:
        d = discover(dfx, dfb, tev, disc, params, FS._names(ticker, [cid]), seed)
        if d is None:
            continue
        scores[int(cid)] = {"gain": d["mean"] - base["mean"], "q": d["q"],
                            "folds": [c - b for c, b in zip(d["fold_growth"], base["fold_growth"])]}
    if not scores:
        return None

    out = {"confirmation_fold": r, "core_confirm_growth": base_conf["growth"],
           "core_confirm_trades": base_conf["trades"], "arms": {}}

    # FLAT — the single strongest candidate on discovery.
    top = max(scores, key=lambda i: scores[i]["gain"])
    c = confirm(dfx, dfb, tev, conf, params, FS._names(ticker, [top]), seed, scores[top]["q"])
    out["arms"]["flat"] = {"picked": top, "family": fam_of.get(top),
                           "discovery_gain": round(scores[top]["gain"], 8),
                           "confirm_delta": (None if c is None else
                                             round(c["growth"] - base_conf["growth"], 8)),
                           "confirm_trades": (None if c is None else c["trades"])}

    # HIERARCHICAL — the strongest family, then that family's representative: the simplest member
    # of its one-SE plateau, which is the rule the sealed policy already uses.
    byfam = {}
    for i, s in scores.items():
        byfam.setdefault(fam_of[i], []).append({"id": i, "mean": s["gain"], "folds": s["folds"]})
    fam_best = {f: max(v, key=lambda x: x["mean"])["mean"] for f, v in byfam.items()}
    fam = max(fam_best, key=fam_best.get)
    plateau, best_id, se = golden.one_se_set(byfam[fam])
    rep = golden.pick_representative(plateau, simplicity) or best_id
    c = confirm(dfx, dfb, tev, conf, params, FS._names(ticker, [rep]), seed, scores[rep]["q"])
    out["arms"]["hierarchical"] = {"family": fam, "picked": int(rep), "best_in_family": int(best_id),
                                   "plateau": [int(x) for x in plateau], "se": round(float(se), 8),
                                   "discovery_gain": round(scores[rep]["gain"], 8),
                                   "family_discovery_gain": round(fam_best[fam], 8),
                                   "confirm_delta": (None if c is None else
                                                     round(c["growth"] - base_conf["growth"], 8)),
                                   "confirm_trades": (None if c is None else c["trades"])}
    return out


def accept(rots):
    """Apply the frozen contract — delegated, so the null runs this exact code.

    The previous implementation examined only the modal pick (`Counter.most_common(1)`). On NOW's
    outer fold 0 two families were each chosen twice and the tie was broken by insertion order,
    handing the verdict to `price_distance` (positive in 1 of 2, median -0.0070) while `volume`
    sat there having won both of its rotations with a median of +0.0298. The fold was rejected on
    an ordering accident. Taking the maximum over everything eligible is what a search actually
    does, it is reproducible across Python versions, and — the reason it is mandatory rather than
    merely better — a permutation cannot honestly reproduce "whichever unit Counter returned
    first".
    """
    return ACC.verdict(rots)


def table(ticker):
    import feature_search as FS
    import golden
    import nested_validation as NV
    import pipeline as P

    t0 = time.time()
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))["tables"][ticker]
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, FS.SCRATCH / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    fam_of = golden.load_families(FS.FAMILIES_PATH, cands)
    simplicity = {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                  for ns, r in P.FEATURE_REGISTRIES.items() if ns != "1h"
                  for f in r["features"] if bool(f.get("implemented", True))}
    outer = NV.outer_folds(t0s, bnds)

    folds_out = []
    for f in reg["folds"]:
        i = f["outer_fold"]
        if i >= len(outer) or f.get("stage") == "no_viable_model":
            continue
        bi = dict(bnds, train_end_idx=int(f["inner_train_end_idx"]))
        inner = P.purged_wf_folds(t0s, bi["train_start_idx"], bi["train_end_idx"])
        rots = [rotation(dfx, dfb, tev, inner, r, f["frozen_params"], ticker, cands,
                         fam_of, simplicity, SEED) for r in range(len(inner))]
        rots = [x for x in rots if x]
        folds_out.append({"outer_fold": i, "n_inner_folds": len(inner),
                          "rotations": rots, "verdict": accept(rots)})

    return {"ticker": ticker, "high_degeneracy_rate": reg.get("high_degeneracy_rate", False),
            "seconds": round(time.time() - t0, 1), "folds": folds_out}


def _line(r, total, done):
    acc = {a: sum(1 for f in r["folds"] if f["verdict"].get(a, {}).get("accepted"))
           for a in ("flat", "hierarchical")}
    flag = "  [high_degeneracy_rate]" if r["high_degeneracy_rate"] else ""
    print(f"  [{done}/{total}] {r['ticker']:<6} przyjęte flat={acc['flat']}/{len(r['folds'])}  "
          f"hierarchical={acc['hierarchical']}/{len(r['folds'])}  ({r['seconds']:.0f}s){flag}")


def recompute_verdicts(path):
    """Re-apply the acceptance rule to rotations already on disk — no model is retrained.

    The rotations are the measurement; the verdict is a function of them. When the rule changes,
    replaying it costs a second and is exactly reproducible, whereas re-running 3B would cost two
    core-hours and, because it is deterministic, would produce the identical rotations anyway.
    """
    from artifact_io import read_json, write_json_atomic

    doc = read_json(path)
    if doc is None:
        raise SystemExit(f"brak lub uszkodzony artefakt: {path}")
    before = after = 0
    for rec in doc["tables"].values():
        for f in rec["folds"]:
            before += sum(1 for a in ("flat", "hierarchical")
                          if f.get("verdict", {}).get(a, {}).get("accepted"))
            f["verdict"] = ACC.verdict(f["rotations"])
            after += sum(1 for a in ("flat", "hierarchical") if f["verdict"][a]["accepted"])
    doc["contract"] = dict(doc.get("contract", {}),
                           rule="max-over-eligible", complexity_pen=ACC.COMPLEXITY_PEN,
                           min_rotations=ACC.MIN_ROTATIONS, majority=ACC.MAJORITY)
    sha = write_json_atomic(path, doc)
    print(f"werdykty przeliczone regułą max-over-eligible: {before} -> {after} przyjęć")
    print(f"  {path}  sha256 {sha[:16]}…")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--recompute-verdicts", action="store_true",
                    help="replay the acceptance rule over the stored rotations; trains nothing")
    args = ap.parse_args()

    if args.recompute_verdicts:
        recompute_verdicts(Path(args.out))
        return 0

    reg = json.loads(REGISTER.read_text(encoding="utf-8"))
    tickers = args.tickers or list(reg["tables"])
    print(f"stage 3B — discovery/confirmation cross-fitting, both arms\n"
          f"  contract: >={MIN_ROTATIONS} rotations, median confirmation delta > {MIN_MEDIAN_DELTA}, "
          f"majority positive\n  {len(tickers)} table(s), frozen params from stage 2, "
          f"operating point {MODE}\n")

    results = {}
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(table, tickers):
                results[r["ticker"]] = r
                _line(r, len(tickers), len(results))
    else:
        for i, t in enumerate(tickers, 1):
            r = table(t)
            results[t] = r
            _line(r, len(tickers), i)

    from artifact_io import write_json_atomic
    sha = write_json_atomic(args.out, {
        "contract": {"min_rotations": MIN_ROTATIONS, "min_median_delta": MIN_MEDIAN_DELTA,
                     "majority": MAJORITY, "rule": "max-over-eligible",
                     "complexity_pen": ACC.COMPLEXITY_PEN, "mode": MODE, "seed": SEED},
        "tables": results})
    print(f"\nwrote {args.out}  sha256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
