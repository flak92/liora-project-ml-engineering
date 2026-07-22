#!/usr/bin/env python3
"""Rung 6 — how much more is a survivor worth once the model is allowed to tune with it?

Every rung before this freezes the hyper-parameters on core, on purpose: tuning on the superset
co-adapts the parameters to features the baseline lacks and inflates every gain. That is right for
screening — cheaply killing the many — but it undersells the few that survive, because it never asks
what a confirmed feature is worth when the model tunes around it.

Rung 6 asks that, and only of Rung-5 survivors. For each survivor it tunes core alone at a budget B
and core+survivor at the SAME budget B, both on the discovery folds, then compares the two on the
confirmation fold that neither tuning ever saw. Equal budget is the fairness constraint: the
survivor competes against a core that had the same chance to improve, not against a core frozen at
the previous rung's parameters.

Tuning is itself a search, so it can manufacture an improvement the way any search manufactures a
maximum. Two guards:

  1. The confirmation fold is untouched by the tuning, so the reported delta is out-of-sample for
     the hyper-parameter search.
  2. Its OWN null. The survivor's column is block-permuted (the same L = 24 machinery as Rung 5) and
     core+permuted is tuned at the same budget B and confirmed on the same untouched fold. If a
     column with the feature's distribution but no alignment to the outcome buys the same tuned
     improvement, the improvement was a tuning artifact, not the feature. The real tuned delta must
     exceed that null.

Rung 6 can only DEMOTE. It may find a Rung-5 survivor adds nothing once core is retuned and strike
it; it can never resurrect anything Rung 5 rejected — it is only ever handed survivors. So a run
here can shrink the confirmed set, never grow it.

    python3 scripts/rung6_survivor_hpo.py --jobs 3
    python3 scripts/rung6_survivor_hpo.py --survivors-from xgb/data/procedure_null_a1.json
    python3 scripts/rung6_survivor_hpo.py --budget 20 --permutations 20   # B and the own-null size
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
import runtime_init  # noqa: E402,F401
runtime_init.apply()
sys.path.insert(0, str(XGB / "src"))
sys.path.insert(0, str(XGB / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

from artifact_io import read_json, write_json_atomic                       # noqa: E402
from ledger import Ledger                                                  # noqa: E402

DATA = DATA_DIR
CROSSFIT = DATA / "crossfit_selection.json"
REGISTER = DATA / "feature_utility.json"
DEFAULT_NULL = DATA / "procedure_null_a1.json"
OUT = DATA / "rung6_survivor_hpo.json"

BUDGET = 20                        # B: equal HPO budget for core and core+survivor (calibratable)
PERMUTATIONS = 20                  # own-null size (calibratable)
PASS_B = 4
FUTILITY_B = PASS_B + 1
SEED = 42
MODE = "quantile"


def survivors(null_path):
    """Rung-5 survivors: (ticker, outer_fold, arm) whose procedure-level null verdict == passed."""
    doc = read_json(null_path)
    if doc is None:
        return []
    out = []
    for t in doc["tables"].values():
        for f in t["folds"]:
            for arm, v in f["arms"].items():
                if v["verdict"] == "passed":
                    out.append({"ticker": f["ticker"], "outer_fold": f["outer_fold"],
                                "arm": arm, "unit": v["unit"]})
    return out


def _representative(crossfit, ticker, ofold, arm, unit):
    """The feature id Rung 6 adds. Flat's unit is already an id; hierarchical's is a family, whose
    representative id the cross-fit verdict recorded."""
    rec = [f for f in crossfit["tables"][ticker]["folds"] if f["outer_fold"] == ofold][0]
    v = rec["verdict"][arm]
    reps = v.get("representatives") or []
    if arm == "flat":
        return int(unit)
    return int(reps[0]) if reps else None


def _eval_folds(dfx, dfb, tev, folds, params, names, seed):
    """Train on each discovery fold, score its validation rows; return growth, q AND viability.

    Rung 6 needs the operating point and the growth (like cross-fit's discover) but also the split
    count, because a tuning trial that produced a constant model has not tested its features and
    cannot win. Built from the same atoms the frozen rungs use, so nothing about Rungs 1–5 changes.
    """
    import numpy as np
    import xgboost as xgb
    import pipeline as P
    import nested_validation as NV

    X = dfb[names].to_numpy(float)
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]
    ticker = str(dfb["asset_id"].iloc[0])
    by_sid = {f"{ticker}:{s['t0']}:{s['direction']}": s for s in tev}

    splits, stds, fold_data = [], [], []
    for tr, va in folds:
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
        td = bst.trees_to_dataframe()
        splits.append(int((td["Feature"] != "Leaf").sum()))
        p = bst.predict(xgb.DMatrix(X[va], feature_names=names))
        p_train = bst.predict(xgb.DMatrix(X[tr], feature_names=names))
        stds.append(float(p.std()))
        scored = [(by_sid[dfb["setup_id"].iloc[j]], float(p[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        fold_data.append((scored, min(vt0), max(vt0), p_train))
    if not fold_data:
        return None
    sel = NV.choose_operating_point(P, dfx, fold_data, MODE)
    return {"mean": float(sum(sel["fold_growth"]) / len(sel["fold_growth"])),
            "q": float(sel["theta"]), "split_nodes": int(sum(splits)),
            "pred_std": round(float(sum(stds) / len(stds)), 6)}


def _tune(dfx, dfb, tev, disc, names, H, budget, seed):
    """Best VIABLE parameter draw over `budget` trials, judged on discovery growth. Same draw
    sequence regardless of the feature set, so core and core+survivor and core+permuted differ only
    in their columns, never in which hyper-parameters they got to try."""
    import numpy as np
    import feature_utility as FU
    import model_viability as MV

    space, rng = FU._space(), np.random.default_rng(seed)
    min_sn, min_sd = FU.viability_floor()
    best = None
    for _ in range(budget):
        absolute, _ = MV.draw(space, rng, H)
        r = _eval_folds(dfx, dfb, tev, disc, absolute, names, seed)
        if r is None or r["split_nodes"] < min_sn or r["pred_std"] < min_sd:
            continue
        if best is None or r["mean"] > best["mean"]:
            best = {"params": absolute, "q": r["q"], "mean": r["mean"]}
    return best


def _tune_and_confirm(dfx, dfb, tev, disc, conf, names, H, budget, seed):
    """Tune one feature set at budget B on discovery, confirm on the untouched fold at its own q."""
    import crossfit_selection as CF
    b = _tune(dfx, dfb, tev, disc, names, H, budget, seed)
    if b is None:
        return None
    c = CF.confirm(dfx, dfb, tev, conf, b["params"], names, seed, b["q"])
    if c is None:
        return None
    return {"growth": c["growth"], "q": b["q"]}


def _tuned_delta(dfx, dfb, tev, disc, conf, core, plus, H, budget, seed, core_side=None):
    """(core+feature) minus core. The core side is invariant across the null — permuting the
    survivor's column leaves every core column untouched — so it is computed once by the caller and
    passed in; only the plus side is recomputed per permutation. This halves Rung 6's cost."""
    cc = core_side if core_side is not None else \
        _tune_and_confirm(dfx, dfb, tev, disc, conf, core, H, budget, seed)
    cp = _tune_and_confirm(dfx, dfb, tev, disc, conf, plus, H, budget, seed)
    if cc is None or cp is None:
        return None
    return {"delta": cp["growth"] - cc["growth"], "core_growth": cc["growth"],
            "plus_growth": cp["growth"], "core_q": cc["q"], "plus_q": cp["q"]}


def evaluate_survivor(job):
    """One survivor: real tuned delta, then its own permutation null; a verdict that can only demote."""
    s, budget, perms, run_id, ledger_path = job
    import numpy as np
    import feature_search as FS
    import model_viability as MV
    import pipeline as P
    import procedure_null as PN

    ticker, ofold, arm, unit = s["ticker"], s["outer_fold"], s["arm"], s["unit"]
    t0 = time.time()
    crossfit = read_json(CROSSFIT)
    rep = _representative(crossfit, ticker, ofold, arm, unit)
    if rep is None:
        return {**s, "status": "no_representative"}

    reg = json.loads(REGISTER.read_text(encoding="utf-8"))["tables"][ticker]
    fold = [f for f in reg["folds"] if f["outer_fold"] == ofold][0]
    scratch = runtime_init.scratch_dir(run_id, ticker)
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, scratch / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    recd = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = recd["df"], recd["df_b"], recd["train_events"], recd["bounds"]
    t0s = [int(x.split(":")[1]) for x in dfb["setup_id"]]

    bi = dict(bnds, train_end_idx=int(fold["inner_train_end_idx"]))
    inner = P.purged_wf_folds(t0s, bi["train_start_idx"], bi["train_end_idx"])
    assert all(max(t0s[j] for j in va) < bnds["oos_start_idx"] for _, va in inner if va), \
        "rung6: an inner val fold reaches OOS"
    # The rotation whose held-out fold this survivor was confirmed on is not singled out; Rung 6
    # holds out the LAST inner fold as its untouched confirmation and discovers on the rest — one
    # clean split, declared before the run rather than chosen to flatter the survivor.
    disc, conf = inner[:-1], inner[-1]

    core = FS._names(ticker, [])
    plus = FS._names(ticker, [rep])
    y = dfb["Y_outcome"].to_numpy(int)
    w = dfb["label_uniqueness_weight"].to_numpy(float)
    # The hessian MUST be measured over the inner-train rows only, exactly as stage 2 did
    # (feature_utility.register_fold), not over the whole frame. The space is hessian-relative, so a
    # different H rescales every gamma and would put Rung 6's tuning in a different parameter regime
    # than the one the frozen rungs calibrated — a larger H (all rows) inflates absolute gamma and
    # forbids nearly every split, which is why an earlier version found no viable draw at all.
    inner_rows = [j for j, x in enumerate(t0s) if x <= fold["inner_train_end_idx"]]
    H = MV.hessian_total(y, w, np.asarray(inner_rows))

    # Tune+confirm core ONCE. Permuting the survivor's column never touches a core column, so the
    # core side is identical on the real data and under every null permutation. Reused below.
    core_side = _tune_and_confirm(dfx, dfb, tev, disc, conf, core, H, budget, SEED)
    real = _tuned_delta(dfx, dfb, tev, disc, conf, core, plus, H, budget, SEED, core_side=core_side)
    if real is None:
        return {**s, "representative": rep, "status": "not_evaluable",
                "seconds": round(time.time() - t0, 1)}

    # Own null: permute the survivor's column with the Rung-5 block machinery, re-tune, re-confirm.
    col = [c for c in plus if c not in set(core)][0]
    rows = sorted({j for tr, va in inner for j in list(tr) + list(va)})
    blocks = PN.blocks_by_bar_time(t0s, rows, PN.L_BLOCK)
    base_col = dfb[[col]].to_numpy(float)
    led = Ledger(ledger_path)
    stage = "rung6"
    key = Ledger.key(s)
    cached = {Ledger.key(u): p for u, p in led.payloads(stage)}.get(key)
    null_deltas = list(cached["null_deltas"]) if cached else []

    rng = np.random.default_rng(SEED + 7 * ofold + hash(str(unit)) % 1000)
    exceed = sum(1 for d in null_deltas if d >= real["delta"])
    m = len(null_deltas)
    while m < perms and exceed < FUTILITY_B:
        order, _ = PN.draw_permutation(blocks, rng)
        permuted = PN.apply_block_order(base_col, blocks, order)
        dfp = dfb.copy(deep=False)
        dfp[col] = permuted[:, 0]
        nd = _tuned_delta(dfx, dfp, tev, disc, conf, core, plus, H, budget, SEED, core_side=core_side)
        if nd is None:
            continue
        null_deltas.append(round(nd["delta"], 8))
        if nd["delta"] >= real["delta"]:
            exceed += 1
        m += 1
    led.append(stage, s, "completed", payload={"null_deltas": null_deltas})

    b = exceed
    if b >= FUTILITY_B:
        verdict = {"verdict": "demoted_by_own_null", "exceedances": b,
                   "final_p_lower_bound": round((1 + b) / (perms + 1), 6)}
    elif real["delta"] <= 0:
        verdict = {"verdict": "demoted_no_tuned_gain", "exceedances": b}
    else:
        verdict = {"verdict": "retained", "exceedances": b, "p_mc": round((1 + b) / (perms + 1), 6)}
    return {**s, "representative": rep, "tuned_delta": round(real["delta"], 8),
            "core_growth": round(real["core_growth"], 6), "plus_growth": round(real["plus_growth"], 6),
            "permutations": m, "null_deltas": null_deltas, **verdict,
            "seconds": round(time.time() - t0, 1),
            "_can_only": "demote — Rung 6 never resurrects a Rung-5 rejection"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors-from", default=str(DEFAULT_NULL))
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    surv = survivors(Path(args.survivors_from))
    if not surv:
        write_json_atomic(args.out, {"survivors": 0, "results": [],
                                     "_note": "no Rung-5 survivor to extract — empty is a valid result"})
        print("Rung 6 — brak survivorów Rung 5 do wyciśnięcia (pusty subset to poprawny wynik)")
        return 0

    run_id = "rung6"
    run_dir = Path(args.run_dir) if args.run_dir else DATA / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "ledger.jsonl"
    Ledger(ledger_path).reconcile_orphans("rung6")

    print(f"Rung 6 — survivor-specific HPO\n  {len(surv)} survivor(ów), budżet B={args.budget}, "
          f"własny null {args.permutations} permutacji, futility b={FUTILITY_B}\n"
          f"  może TYLKO degradować — nigdy nie wskrzesza odrzucenia Rung 5\n")

    jobs = [(s, args.budget, args.permutations, run_id, ledger_path) for s in surv]
    results = []
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(evaluate_survivor, jobs):
                results.append(r)
    else:
        for j in jobs:
            results.append(evaluate_survivor(j))

    retained = [r for r in results if r.get("verdict") == "retained"]
    demoted = [r for r in results if str(r.get("verdict", "")).startswith("demoted")]
    sha = write_json_atomic(args.out, {
        "budget_B": args.budget, "permutations": args.permutations, "seed": SEED,
        "survivors": len(surv), "retained": len(retained), "demoted": len(demoted),
        "_invariant": "Rung 6 can only demote; retained <= Rung-5 survivors always",
        "results": results})

    print(f"{'ticker':<7}{'fold':>5} {'arm':<13}{'unit':<16}{'Δ strojone':>12}{'b':>4}  werdykt")
    for r in sorted(results, key=lambda x: (x["ticker"], x["outer_fold"])):
        td = r.get("tuned_delta")
        tds = f"{td:+.4f}" if td is not None else "—"
        print(f"{r['ticker']:<7}{r['outer_fold']:>5} {r['arm']:<13}{str(r['unit']):<16}"
              f"{tds:>12}{r.get('exceedances', 0):>4}  {r.get('verdict', r.get('status'))}")
    print(f"\n  retained: {len(retained)}   demoted: {len(demoted)}")
    print(f"\nwrote {args.out}  sha256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
