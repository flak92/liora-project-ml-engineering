#!/usr/bin/env python3
"""XGB per-asset feature search (Track A) — Train-CV only, ZERO OOS reads.

Implements the standard in `docs/METHODOLOGY.md`, reusing layer7's own OOF machinery.
Per ticker:

  1. Output-B is derived ONCE on the SUPERSET manifest (frozen 1h core + every optional/opt-in id),
     which fixes ONE common event set + purged-WF fold partition for every candidate subset — a
     subset eval is then a pure column-slice + XGBoost refit (no feature recompute, like the LSTM
     search's superset trick).
  2. Baseline = frozen 1h core only; ONE Train-only HPO (`layer7_optuna`), params FROZEN.
  3. Prefilter each candidate by its single-feature marginal Train-OOF log-growth gain over core,
     keeping only those with a robust fold-win edge (`min_feature_gain`, `min_fold_win_frac`).
     Every subset is scored at ONE shared floor-respecting operating point over
     OPERATING_SPACE.theta_scoring (op_select accumulate-then-select — never a per-fold theta).
  4. TWO-STAGE family-first greedy (golden-family-v2): stage 1 = single pass over the family
     REPRESENTATIVES (one-SE plateau -> simplest by complexity_score), each added only if it
     lifts the complexity-penalized score AND beats the current subset in a majority of folds;
     stage 2 = the remaining variants of ACCEPTED families only (same marginal rule, at most
     FAMILY_CAP per family). `max_select` caps additions.
  5. Apply the subset only if it beats core by `min_gain` AND the baseline produced at least
     `min_trades` Train-CV trades (the A3 rule — no economic power ⇒ no selection; checked right
     after the baseline eval, so thin tickers never pay for the prefilter). The 1h core is
     frozen (ids 1-99 can never enter an override; `_override_selection` fail-closes).

Known bias (accepted, guarded): HPO runs on the SUPERSET, so the frozen params are co-adapted to
features the core-only baseline lacks — reported gains are therefore UPPER-biased in magnitude.
The subset RANKING stays fair (identical df_b/events/folds/params for every subset), the
`MIN_GAIN` + `MIN_FOLD_WIN_FRAC` guards absorb the inflation, and the sealed run (`run_asset`)
re-runs HPO fresh on the chosen manifest, so nothing biased is ever deployed.

OOS is NEVER read here — selection is Train-CV only; the sealed run (`run_asset`) is the single OOS
read. Writes the per-asset overrides file ($PER_ASSET_OVERRIDES_PATH) atomically
under an flock. Resumable: a ticker already present with a matching `recipe_hash` is skipped unless
`--force`.

  .venv/bin/python3 src/xgb/feature_search.py [TICKER ...] [--out PATH] [--limit N] [--force]
  .venv/bin/python3 src/xgb/feature_search.py --status
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent                  # src/xgb — pipeline.py is a sibling
sys.path.insert(0, str(REPO))

import duckdb                                            # noqa: E402
import numpy as np                                      # noqa: E402
import xgboost as xgb                                   # noqa: E402

import pipeline as P                                    # noqa: E402

sys.path.insert(0, str(REPO.parent / "shared"))
import golden_calibration as golden                      # noqa: E402  (search POLICY only; gates stay here)

FAMILIES_PATH = REPO.parents[1] / "config" / "feature_families_xgb.json"
SEARCH_POLICY = "golden-family-v2"                       # part of the recipe identity

# Search knobs — mirror config/lstm.json FEATURE_SEARCH; env-overridable for experiments.
HPO_TRIALS        = int(os.environ.get("XGB_FS_HPO_TRIALS", "30"))
MIN_GAIN          = float(os.environ.get("XGB_FS_MIN_GAIN", "0.005"))
MAX_SELECT        = int(os.environ.get("XGB_FS_MAX_SELECT", "6"))
MIN_FEATURE_GAIN  = float(os.environ.get("XGB_FS_MIN_FEATURE_GAIN", "0.008"))
COMPLEXITY_PEN    = float(os.environ.get("XGB_FS_COMPLEXITY_PENALTY", "0.004"))
MIN_FOLD_WIN_FRAC = float(os.environ.get("XGB_FS_MIN_FOLD_WIN_FRAC", "0.5"))
MIN_TRADES        = int(os.environ.get("XGB_FS_MIN_TRADES", "8"))   # A3 economic-power floor (Train-CV)
FAMILY_CAP        = int(os.environ.get("XGB_FS_FAMILY_CAP", "3"))   # stage-2 variants per accepted family

DEFAULT_OUT = Path(os.environ.get("PER_ASSET_OVERRIDES_PATH",
                                  str(REPO / "data" / "per_asset_feature_overrides.json")))
SCRATCH = REPO / "tools" / ".search_scratch"


def _utcnow():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def candidate_ids():
    """Every implemented optional/opt-in id outside the frozen 1h core (ids 1-99 are never candidates)."""
    ids = []
    for ns, reg in P.FEATURE_REGISTRIES.items():
        if ns == "1h":
            continue
        ids += [int(f["id"]) for f in reg["features"] if bool(f.get("implemented", True))]
    return sorted(ids)


def recipe_hash(cands):
    """Identity of everything that shapes a search result. Resume skips a ticker only when its
    stored hash matches, so EVERY result-affecting input must be here: the candidate pool and
    non-1h registry formulas, the worker knobs, the FULL pipeline parameters (capital mode,
    barriers, costs, splits, entry gate — they define labels, events and the engine), the Optuna
    search space (drives best_params), and the frozen 1h core formulas (they ARE the baseline).
    A config edit therefore invalidates resume instead of silently mixing incomparable rows."""
    payload = {"cands": cands, "knobs": [HPO_TRIALS, MIN_GAIN, MAX_SELECT, MIN_FEATURE_GAIN,
                                         COMPLEXITY_PEN, MIN_FOLD_WIN_FRAC, MIN_TRADES, FAMILY_CAP],
               "feats": sorted((int(f["id"]), f["name"], f.get("formula", ""))
                               for ns, reg in P.FEATURE_REGISTRIES.items() if ns != "1h"
                               for f in reg["features"] if bool(f.get("implemented", True))),
               "core_1h": sorted((int(f["id"]), f["name"], f.get("formula", ""))
                                 for f in P.FEATURE_REGISTRIES["1h"]["features"]
                                 if bool(f.get("implemented", True))),
               "pipeline_parameters": P.PIPELINE_PARAMETERS,
               "optuna_space": P.XGBOOST_OPTUNA_SEARCH_SPACE,
               "search_policy": SEARCH_POLICY,
               "families": json.loads(FAMILIES_PATH.read_text(encoding="utf-8"))["families"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _manifest(ticker, ids):
    ov = {ticker: {"selected_optional_ids": sorted(int(i) for i in ids)}}
    return P.resolve_feature_manifest(ticker, overrides=ov)


def _names(ticker, ids):
    return P.feature_names_of(_manifest(ticker, ids))


def eval_subset(df, df_b, train_events, bounds, params, names, seed):
    """Train-OOF score of one subset at ONE shared floor-respecting operating point over
    OPERATING_SPACE.theta_scoring (accumulate-then-select via op_select — never a per-fold
    best theta, the fold-oracle bias #10; under all-in the point replays at f=1). Fail-closed:
    asserts every val fold stays strictly pre-OOS. Returns (per_fold_growth[list|None] AT the
    shared point, total Train-OOF trades at that point)."""
    X = df_b[names].to_numpy(float)
    y = df_b["Y_outcome"].to_numpy(int)
    w = df_b["label_uniqueness_weight"].to_numpy(float)
    t0s = [int(sid.split(":")[1]) for sid in df_b["setup_id"]]
    folds = P.purged_wf_folds(t0s, bounds["train_start_idx"], bounds["train_end_idx"])
    ticker = str(df_b["asset_id"].iloc[0])
    by_sid = {f"{ticker}:{s['t0']}:{s['direction']}": s for s in train_events}
    assert all(max(t0s[j] for j in va) < bounds["oos_start_idx"] for _, va in folds if va), \
        "xgb search: a CV val fold reaches OOS (purge invariant violated)"
    growth = [None] * len(folds)
    valid, fold_data = [], []
    for pos, (tr, va) in enumerate(folds):
        if len(np.unique(y[tr])) < 2:
            continue
        bst = P._xgb_train(X[tr], y[tr], w[tr], params, seed, feature_names=names)
        p = bst.predict(xgb.DMatrix(X[va], feature_names=names))
        scored = [(by_sid[df_b["setup_id"].iloc[j]], float(p[k])) for k, j in enumerate(va)]
        vt0 = [t0s[j] for j in va]
        fold_data.append((scored, min(vt0), max(vt0)))
        valid.append(pos)
    if not fold_data:
        return growth, 0
    sel = P.score_shared_operating_point(df, fold_data)
    for k, pos in enumerate(valid):
        growth[pos] = float(sel["fold_growth"][k])
    return growth, int(sel["trades"])


def _fmean(folds):
    vals = [v for v in folds if v is not None]
    return float(np.mean(vals)) if vals else None


def _fold_win_frac(base, cur):
    pairs = [(b, c) for b, c in zip(base, cur) if b is not None and c is not None]
    if not pairs:
        return 0.0
    return sum(1 for b, c in pairs if c > b) / len(pairs)


def search_ticker(ticker, seed):
    # Defense-in-depth: reseed per ticker so a future global-RNG consumer in the hot path can
    # never make fresh-vs-resumed runs diverge (today nothing consumes it; keep it that way).
    P.seed_everything(seed)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, SCRATCH / f"{ticker}_1h.parquet")
    cands = candidate_ids()
    rec = P.derive_output_b(df, ticker, _manifest(ticker, cands))    # ONE compute; common events/folds
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]

    # HPO on the SUPERSET (feature-rich → params that can actually trade), then FREEZE and score every
    # subset at those params — so the score isolates the FEATURE effect, not a per-subset HPO. Tuning
    # on core-only would trap tickers whose tradeable signal only exists once features are present.
    best_params, _cv_ap, _nf = P.layer7_optuna(dfx, dfb, tev, bnds, seed, _manifest(ticker, cands),
                                               trials=HPO_TRIALS)
    base_folds, base_trades = eval_subset(dfx, dfb, tev, bnds, best_params, _names(ticker, []), seed)
    base_mean = _fmean(base_folds)
    if base_mean is None:
        return {"selected": [], "provenance": {"stage": "no_valid_folds", "cv_train_trades": base_trades,
                                               "stop_reason": "no valid baseline score (folds degenerate)",
                                               "evaluated_at": _utcnow(), "recipe_hash": recipe_hash(cands)}}
    # A3 rule (docs/METHODOLOGY.md §7): every acceptance criterion is a function of
    # trades — a baseline below the trade floor can NEVER be applied, so running the 45-candidate
    # prefilter + greedy for it is compute spent at zero statistical power. Exit now.
    if base_trades < MIN_TRADES:
        return {"selected": [], "provenance": {"oof_base": round(base_mean, 6), "oof_best": round(base_mean, 6),
                                               "gain": 0.0, "cv_train_trades": int(base_trades), "n_selected": 0,
                                               "stop_reason": "baseline_below_min_trades",
                                               "stage": "thin_no_trades", "evaluated_at": _utcnow(),
                                               "recipe_hash": recipe_hash(cands)}}

    # 1) prefilter: single-candidate robust marginal edge over core (gates UNCHANGED — this is
    # the golden-calibration screening step: per-fold deltas are kept so each family's one-SE
    # plateau can be computed afterwards)
    survivor_stats = []
    for cid in cands:
        folds, _ = eval_subset(dfx, dfb, tev, bnds, best_params, _names(ticker, [cid]), seed)
        mean = _fmean(folds)
        if mean is None:
            continue
        gain, wf = mean - base_mean, _fold_win_frac(base_folds, folds)
        if gain >= MIN_FEATURE_GAIN and wf >= MIN_FOLD_WIN_FRAC:
            deltas = [(c - b) if (c is not None and b is not None) else None
                      for c, b in zip(folds, base_folds)]
            survivor_stats.append({"id": cid, "mean": gain, "folds": deltas})

    # 2) golden policy v2 (docs/METHODOLOGY.md, #20-21): survivors grouped
    # by OHLCV relationship family; per family the one-SE plateau and its SIMPLEST representative
    # (complexity_score: history window, operators, length; id tiebreak). TWO-STAGE pool:
    # stage 1 sees ONLY representatives; stage 2 probes remaining variants of ACCEPTED families
    # (<= FAMILY_CAP each). Variants of non-accepted families never enter greedy — near-duplicates
    # are spectrum points, not alpha sources.
    fam_of = golden.load_families(FAMILIES_PATH, cands)
    simplicity = {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                  for ns, reg in P.FEATURE_REGISTRIES.items() if ns != "1h"
                  for f in reg["features"] if bool(f.get("implemented", True))}
    reps, fam_variants, fam_report = (golden.family_stage_pool(survivor_stats, fam_of, simplicity)
                                      if survivor_stats else ([], {}, {}))

    # 3) single-pass marginal greedy (#21 — the pool is NOT re-scored every iteration): a
    # candidate joins only if it lifts the complexity-penalized score AND beats the CURRENT
    # subset in >= MIN_FOLD_WIN_FRAC of folds (marginal hypothesis; gates unchanged).
    state = {"selected": [], "best_pen": base_mean, "cur_folds": base_folds}
    stop_reason = None

    def _try_add(cid):
        folds, _ = eval_subset(dfx, dfb, tev, bnds, best_params,
                               _names(ticker, state["selected"] + [cid]), seed)
        mean = _fmean(folds)
        if mean is None:
            return False
        pen = mean - COMPLEXITY_PEN * (len(state["selected"]) + 1)
        if pen > state["best_pen"] + 1e-6 and _fold_win_frac(state["cur_folds"], folds) >= MIN_FOLD_WIN_FRAC:
            state.update(selected=state["selected"] + [cid], best_pen=pen, cur_folds=folds)
            return True
        return False

    accepted_fams = []
    for cid in reps:                                    # stage 1: family representatives only
        if len(state["selected"]) >= MAX_SELECT:
            stop_reason = "max_select_reached"
            break
        if _try_add(cid):
            accepted_fams.append(fam_of[cid])
    for fam in accepted_fams:                           # stage 2: accepted families' variants
        if stop_reason:
            break
        for cid in fam_variants.get(fam, [])[:FAMILY_CAP]:
            if len(state["selected"]) >= MAX_SELECT:
                stop_reason = "max_select_reached"
                break
            _try_add(cid)
    selected = state["selected"]
    if stop_reason is None:
        stop_reason = "pool_exhausted" if reps else "no_candidate_passed_marginal_gates"

    # 4) acceptance (economic: beats core by min_gain; A3 floor already enforced above).
    # cur_folds IS the deterministic eval of the final subset — no redundant re-eval.
    fin_mean = _fmean(state["cur_folds"]) if selected else base_mean
    applied = bool(selected) and fin_mean is not None and (fin_mean >= base_mean + MIN_GAIN)
    if not applied and selected:
        stop_reason = "final_subset_below_min_gain"
    stage = "auto_selected" if applied else "core_only"
    rec = {"selected": sorted(selected) if applied else [],
           "provenance": {"oof_base": round(base_mean, 6),
                          "oof_best": round(fin_mean, 6) if fin_mean is not None else None,
                          "gain": round(fin_mean - base_mean, 6) if applied else 0.0,
                          "cv_train_trades": int(base_trades), "n_selected": len(selected) if applied else 0,
                          "families": fam_report, "family_variant_cap": FAMILY_CAP,
                          "stop_reason": stop_reason,
                          "stage": stage, "evaluated_at": _utcnow(), "recipe_hash": recipe_hash(cands)}}
    report_path = os.environ.get("GOLDEN_ROUND_REPORT")
    if report_path:                                     # §10 minimal round report (night sets the path)
        golden.append_jsonl(report_path, golden.round_report(
            ticker=ticker, model="xgb", recipe_hash=rec["provenance"]["recipe_hash"],
            baseline_score=rec["provenance"]["oof_base"], baseline_trades=int(base_trades),
            families_report=fam_report, selected=rec["selected"],
            final_gain=rec["provenance"]["gain"], verdict=stage, stop_reason=stop_reason))
    return rec


# ------------------------------ overrides IO (atomic + flock) ------------------------------

def _empty_doc():
    return {"schema_version": "per_asset_feature_overrides.v1",
            "_meta": {"written_by": "xgb/tools/feature_search.py — Train-CV only, atomic tmp+os.replace"},
            "asset_overrides": {}}


def _read_doc(path):
    if not Path(path).exists():
        return _empty_doc()
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_override(path, ticker, result):
    """flock the file, read-modify-write, atomic tmp+os.replace. Only a non-empty selection carries
    the id list; a core-only / thin ticker records provenance with an empty selection."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            doc = _read_doc(path)
            doc.setdefault("asset_overrides", {})[ticker] = {
                "selected_optional_ids": [int(i) for i in result["selected"]],
                "provenance": result["provenance"]}
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def universe():
    con = duckdb.connect(str(P.bars_db()), read_only=True)
    try:
        return [t for (t,) in con.execute("select distinct ticker from bars_1h order by ticker").fetchall()]
    finally:
        con.close()


def _done(path, rh):
    doc = _read_doc(path)
    return {t for t, v in doc.get("asset_overrides", {}).items()
            if isinstance(v, dict) and v.get("provenance", {}).get("recipe_hash") == rh}


def _search_and_write(ticker, out):
    """One ticker end-to-end — also the spawn-pool task. Under spawn the module re-imports fresh
    in the child, so the parent's OMP/BLAS=1 env caps (inherited) apply BEFORE numpy loads; the
    per-ticker seed_everything inside search_ticker makes results identical to a serial run."""
    t0 = time.time()
    try:
        res = search_ticker(ticker, P.seed_everything())
    except Exception as e:                                  # fail-soft: one odd ticker never stops the run
        return ticker, None, f"ERROR {type(e).__name__}: {e}", time.time() - t0
    write_override(out, ticker, res)
    return ticker, res, None, time.time() - t0


def _default_jobs():
    """workers × 1 thread ≈ cores (docs/BEST_PRACTICE_ML_LOOPS_SERVER.md §2): fan out processes,
    pin math libs to 1 thread each. Capped so a small box auto-adapts."""
    return int(os.environ.get("XGB_FS_JOBS", 0)) or max(1, min(14, (os.cpu_count() or 2) - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", help="tickers to search (default: full bars universe)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="cap number of tickers this run (chunking)")
    ap.add_argument("--jobs", type=int, default=0, help="parallel worker processes, 1 thread each "
                                                        "(0 = auto: XGB_FS_JOBS or min(14, cores-2))")
    ap.add_argument("--force", action="store_true", help="re-search even if recipe_hash matches")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    seed = P.seed_everything()
    cands = candidate_ids()
    rh = recipe_hash(cands)
    uni = args.tickers or universe()
    done = _done(args.out, rh)

    if args.status:
        print(f"recipe_hash={rh} · candidates={len(cands)} · universe={len(uni)} · "
              f"done(matching hash)={len(done)} · todo={len([t for t in uni if t not in done])}")
        return

    todo = uni if args.force else [t for t in uni if t not in done]
    if args.limit:
        todo = todo[:args.limit]
    jobs = min(args.jobs or _default_jobs(), max(1, len(todo)))
    print(f"xgb feature-search · recipe={rh} · {len(todo)}/{len(uni)} to search · jobs={jobs} "
          f"· out={args.out}", flush=True)

    def report(i, ticker, res, err, dt):
        if err:
            print(f"[{i}/{len(todo)}] {ticker}: {err}", flush=True)
            return
        pv = res["provenance"]
        print(f"[{i}/{len(todo)}] {ticker}: {pv['stage']} sel={res['selected']} "
              f"base={pv.get('oof_base')} best={pv.get('oof_best')} trades={pv.get('cv_train_trades')} "
              f"({dt:.0f}s)", flush=True)

    if jobs <= 1:
        for i, t in enumerate(todo, 1):
            report(i, *_search_and_write(t, args.out))
    else:
        # env caps EXPORTED before pool creation: spawn children inherit them, so BLAS/OMP read
        # "1" at import time in every worker (workers × 1 ≈ cores; overrides already flock-safe)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=jobs,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futs = {pool.submit(_search_and_write, t, args.out): t for t in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                t = futs[fut]
                try:
                    report(i, *fut.result())
                except Exception as e:                       # dead worker (OOM/crash) — log + go on
                    print(f"[{i}/{len(todo)}] {t}: WORKER DIED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
