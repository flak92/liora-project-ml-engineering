#!/usr/bin/env python3
"""S.1-LSTM — the per-asset feature-selection loop (continuous, resumable, PARALLEL).

This is the full self-improving loop and it subsumes the plain universe run. For each bundled
ticker a worker: measures the CORE-only Train CV AUC-PR (a light FIXED LSTM, so the score
reflects the FEATURES, not HPO), forward-selects an optional subset under the overfit gate,
and runs the asset ONCE (its single OOS read) — with the improved subset if the gate cleared,
else core-only — writing that dashboard row. Re-searched tickers are only re-run when a newly
proposed feature makes them gain; applied tickers are never re-read (one-shot OOS preserved).

Parallel by PROCESS: JOBS workers each own one ticker end-to-end (search + apply + run_asset);
each worker uses the configured torch thread count and is process-isolated, so results are
byte-identical to the sequential path (determinism is per-ticker seeded). The coordinator is
the sole writer of search_state.db; per-asset override writes are serialized with a file lock.
Everything is Train-only; OOS is read exactly once per asset, at apply.

  make search-on / search-off / search-status   (JOBS=<n> to override the worker count)
"""
import ctypes
import fcntl
import json
import multiprocessing
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Bound BLAS/OpenMP to the per-model torch thread count BEFORE numpy imports, so JOBS workers
# don't each spawn cores-many BLAS threads and oversubscribe. torch intra-op is bounded per
# training call by model.seed_everything(); this covers numpy/pandas.
_TT = str(json.loads((Path(__file__).resolve().parents[2] / "config" / "lstm.json").read_text())["TRAIN"]["torch_threads"])
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _TT)

import duckdb
import numpy as np
import torch

import pipeline as P
import model as M
import features as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "shared"))
import golden_calibration as golden                     # search POLICY only (gates stay here)
import universal as UNI

FAMILIES_PATH = ROOT.parents[1] / "config" / "feature_families_lstm.json"
BACKBONE_PATH = ROOT / "data" / "universal_backbone.json"
_EVAL_WARM = {"checked": False, "state": None, "names": None}


def _eval_backbone():
    """Eval-arch backbone for warm-starting search evals. Loaded once; disabled when absent,
    when LSTM_COLD_START=1 (A/B validation), or when its hidden size no longer matches the
    configured eval arch (fail-open to cold start — never a wrong-shaped transfer)."""
    if not _EVAL_WARM["checked"]:
        _EVAL_WARM["checked"] = True
        if not os.environ.get("LSTM_COLD_START"):
            bb = UNI.load_backbone(BACKBONE_PATH)
            if bb and bb.get("eval_state") is not None \
                    and int(bb["eval_arch"]["hidden"]) == int(SC["eval_hidden"]):
                _EVAL_WARM["state"], _EVAL_WARM["names"] = bb["eval_state"], bb["superset"]
                print("feature-search: universal eval warm-start ACTIVE", flush=True)
    return _EVAL_WARM


def _eval_init_state():
    return _eval_backbone()["state"]


def _eval_init_names():
    return _eval_backbone()["names"]
STATE = ROOT / "search_state.db"
OVERRIDES = ROOT / "data" / "per_asset_feature_overrides.json"
OVERRIDES_LOCK = ROOT / ".overrides.lock"
CONTROL = ROOT / "search_control.json"
PROPOSED = ROOT / "features_proposed.json"
LOG_DIR = ROOT / "logs" / "search"
STOP_FLAG = LOG_DIR / "STOP.flag"
JOURNAL = LOG_DIR / "agent_journal.md"
SC = P.CONFIG["FEATURE_SEARCH"]
MIN_GAIN = float(SC["min_gain"])
# Independent weight-init seeds the evaluator averages over: reduces seed-luck variance in the
# CV score so a feature's marginal gain has to be REAL to clear the overfit gate. Well-separated
# (prime stride), reproducible, and eval_seeds=1 reproduces the old single-seed score exactly.
EVAL_SEEDS = [int(M.SEED) + i * 10007 for i in range(max(1, int(SC.get("eval_seeds", 1))))]


def jobs():
    """Parallel worker count: JOBS env, else cores minus 2, divided by the per-model torch
    threads (so JOBS × torch_threads ≈ cores)."""
    tt = max(1, int(P.CONFIG["TRAIN"]["torch_threads"]))
    return max(1, int(os.environ.get("JOBS") or max(1, (os.cpu_count() or 2) - 2) // tt))


def _worker_init():
    """Die with the coordinator (no orphaned spawn workers eating CPU). Linux-only; no-op
    elsewhere. Threads/seed are set per training call by model.seed_everything."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, 9, 0, 0, 0)   # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass


def _load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_control():
    """The agent-owned control file: {halt, paused_tickers, min_gain, proposed_features}."""
    return _load_json(CONTROL, {})


def _atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _journal(line):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as fh:
        fh.write(f"- {time.strftime('%Y-%m-%d %H:%M')}  {line}\n")


PANEL = ("AAPL", "JPM", "KO", "NVDA")                   # diverse validation panel (not just AAPL)


def ingest_proposals():
    """Validate + register the Claude-proposed features listed in search_control.json. The
    agent only WRITES the control file; the worker is the sole validator/registrar. A proposal
    must pass the fail-closed DSL gate (causal by construction, dense-finite, non-constant,
    unique) on EVERY ticker of a diverse panel before it gets a MONOTONIC id (never reused)
    in features_proposed.json and enters the search pool. Only the proposals consumed this
    pass are cleared (so a proposal appended during the pass is not lost); every verdict is
    logged to agent_journal.md."""
    ctl = _load_json(CONTROL, {})
    proposals = ctl.get("proposed_features", [])
    if not proposals:
        return
    # epoch discipline: an accepted proposal changes the recipe hash (pool + families), which
    # would make rows already sealed in the OPEN epoch incomparable (janitor CRITICAL). So
    # proposals are only ingested while the open epoch has no sealed reads yet — otherwise they
    # stay queued in search_control.json and land at the next epoch boundary.
    import oos_ledger
    cur = oos_ledger.open_epoch_label("lstm")
    if cur is not None and any(r["event"] == "oos_read" and r.get("epoch") == cur
                               for r in oos_ledger.read_all("lstm")):
        _journal(f"proposals deferred ({len(proposals)}): epoch {cur!r} already has sealed reads — "
                 "they will be ingested at the next epoch boundary")
        return
    reg = F.proposed_registry(PROPOSED)
    feat_names = {k for k in reg if not k.startswith("_")}
    known = set(F.CORE_FEATURE_NAMES) | set(F.OPTIONAL_FEATURE_NAMES) | feat_names
    panel = [P.load_bars(t) for t in PANEL]
    next_id = int(reg.get("_next_id", 501))            # monotonic — persisted, never reused
    consumed, added = set(), []
    fam_doc = _load_json(FAMILIES_PATH, {"families": {}})
    for pr in proposals:
        name, expr, family = pr.get("name"), pr.get("expr"), pr.get("family")
        consumed.add((name, expr))
        if name in known:
            _journal(f"proposal {name!r}: REJECTED — name already exists")
            continue
        # golden §6-7: a proposal must name its OHLCV relationship family (an EXISTING one — the
        # taxonomy is recipe identity; a new family is an operator decision, not an agent's)
        if family not in fam_doc.get("families", {}):
            _journal(f"proposal {name!r}: REJECTED — unknown/missing family {family!r} "
                     f"(must be one of the existing families in feature_families.json)")
            continue
        verdicts = [F.validate_proposal(name, expr, df, known) for df in panel]
        ok = all(o for o, _ in verdicts)
        why = "ok" if ok else next(w for o, w in verdicts if not o)
        _journal(f"proposal {name!r}: {'ACCEPTED id=' + str(next_id) if ok else 'REJECTED — ' + why} "
                 f"[family={family}] :: {expr}")
        if ok:
            reg[name] = {"id": next_id, "expr": expr}
            fam_doc["families"][family].append(next_id)     # keep the map exactly-once complete
            known.add(name)
            added.append(f"{name}#{next_id}")
            next_id += 1
    reg["_next_id"] = next_id
    _atomic_json(PROPOSED, reg)
    _atomic_json(FAMILIES_PATH, fam_doc)
    if added:
        print(f"registered {len(added)} proposed feature(s): {', '.join(added)}", flush=True)
    ctl = _load_json(CONTROL, {})                       # re-read; clear ONLY what we consumed
    ctl["proposed_features"] = [p for p in ctl.get("proposed_features", [])
                                if (p.get("name"), p.get("expr")) not in consumed]
    _atomic_json(CONTROL, ctl)


def universe():
    con = duckdb.connect(f"{ROOT / 'data' / 'sp500_1d.duckdb'}", read_only=True)
    try:
        return [r[0] for r in con.execute("select distinct symbol from bars_1d order by symbol").fetchall()]
    finally:
        con.close()


def optional_ids():
    return sorted(P.optional_id_to_name())            # OPTIONAL (101+) and PROPOSED (501+, Stage 3)


def _prep(ticker):
    """Per-ticker prep, computed ONCE. Fixes a COMMON event set + fold partition from the
    feature SUPERSET (every candidate feature finite), so every subset below is scored on the
    exact same events/folds — a long-warmup feature can never win by deleting hard early
    samples. Builds the raw superset window tensor once; subset evals just slice columns."""
    df = P.load_bars(ticker)
    masks, bounds = P.split_masks(df)
    feats = P.feature_frame(df, ticker)                # CORE+OPTIONAL from the RAM cache if built
    superset = P.resolve_superset_manifest()
    tev = P.generate_candidates(df, masks["train"], feats)
    tev, _ = P.purge_train_events(tev, bounds)
    labeled = P.label_events(df, tev, bounds["train_end_idx"])
    events, X_sup = P.build_sequences(labeled, feats, superset)   # common (superset-eligible) set
    if len(events) < 50:
        return None
    P.assign_uniqueness_weights(events)
    y = np.array([e["y"] for e in events], np.int64)
    w = np.array([e["weight"] for e in events], np.float32)
    folds = P.purged_wf_folds([e["t0"] for e in events], bounds["train_start_idx"], bounds["train_end_idx"])
    if not folds:
        return None
    return {"feats": feats, "masks": masks, "superset": superset, "df": df, "events": events,
            "oos0": bounds["oos_start_idx"],
            "col": {n: i for i, n in enumerate(superset)}, "X_sup": X_sup, "y": y, "w": w, "folds": folds}


def evaluate(prep, opt_ids, trades_out=None):
    """Per-FOLD score (a list) for CORE + opt_ids on the common event set, with causal per-fold
    normalization, AVERAGED over EVAL_SEEDS independent weight inits so the score measures the
    FEATURES not one lucky seed. The metric (FEATURE_SEARCH.metric) is PROFIT by default —
    'oof_log_growth': the best out-of-fold log-growth over the coarse (θ,λ) grid via run_engine, so
    features are selected for TRADEABILITY (consistent with the profit-aligned HPO). 'auc_pr' keeps
    the original ranking metric as a fallback. Returns per-fold values so the caller can require a
    feature to help ROBUSTLY (in a majority of folds).

    trades_out: optional list — for the growth metric, appends per fold the seed-averaged trade
    count at that fold's best operating point (the A3 economic-power floor reads it)."""
    metric = P.CONFIG["FEATURE_SEARCH"].get("metric", "oof_log_growth")
    id2name = P.optional_id_to_name()
    manifest = list(F.CORE_FEATURE_NAMES) + [id2name[i] for i in sorted(opt_ids) if i in id2name]
    idx = [prep["col"][n] for n in manifest]
    Xm = prep["X_sup"][:, :, idx]
    y, w, emb = prep["y"], prep["w"], P.CONFIG["EMBARGO_BARS"]
    df, events, t0s = prep["df"], prep["events"], [e["t0"] for e in prep["events"]]
    H = P.CONFIG["H"]
    out = [None] * len(prep["folds"])
    if metric == "auc_pr":                              # ranking fallback — no engine, unchanged
        for pos, (tr, va, val_lo) in enumerate(prep["folds"]):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            st = P.train_norm_stats(prep["feats"], prep["masks"]["train"], manifest, before_idx=val_lo - emb)
            mu = np.array([st[n]["mean"] for n in manifest], np.float32)
            sd = np.array([st[n]["std"] for n in manifest], np.float32)
            Xf = (Xm - mu) / sd
            vals = []
            for sv in EVAL_SEEDS:
                m, _, _ = M.train_model(Xf[tr], y[tr], w[tr], SC["eval_hidden"], SC["eval_lr"],
                                        SC["eval_dropout"], epochs=int(SC["eval_epochs"]), seed=sv,
                                        init_state=_eval_init_state(), init_names=_eval_init_names(),
                                        feat_names=manifest)
                vals.append(P.average_precision(y[va], M.predict_proba(
                    m, torch.from_numpy(np.ascontiguousarray(Xf[va])))))
            out[pos] = float(np.mean(vals))
        return out
    # oof_log_growth: accumulate every fold's OOF predictions per seed, then ONE shared
    # floor-respecting operating point per seed (op_select) — never a per-fold best theta
    # (the fold-oracle bias); under all-in the point replays at f=1, the sizing it deploys.
    valid, per_seed = [], [[] for _ in EVAL_SEEDS]
    for pos, (tr, va, val_lo) in enumerate(prep["folds"]):
        if len(np.unique(y[tr])) < 2:
            continue
        st = P.train_norm_stats(prep["feats"], prep["masks"]["train"], manifest, before_idx=val_lo - emb)
        mu = np.array([st[n]["mean"] for n in manifest], np.float32)
        sd = np.array([st[n]["std"] for n in manifest], np.float32)
        Xf = (Xm - mu) / sd
        lo, hi = min(t0s[j] for j in va), max(t0s[j] for j in va)
        assert hi + H < prep["oos0"], "feature-search: fold label horizon reaches OOS"
        for si, sv in enumerate(EVAL_SEEDS):
            m, _, _ = M.train_model(Xf[tr], y[tr], w[tr], SC["eval_hidden"], SC["eval_lr"],
                                    SC["eval_dropout"], epochs=int(SC["eval_epochs"]), seed=sv,
                                    init_state=_eval_init_state(), init_names=_eval_init_names(),
                                    feat_names=manifest)
            oof = M.predict_proba(m, torch.from_numpy(np.ascontiguousarray(Xf[va])))
            per_seed[si].append(([(events[j], float(oof[k])) for k, j in enumerate(va)], lo, hi))
        valid.append(pos)
    if valid:
        sels = [M.score_shared_operating_point(df, fd) for fd in per_seed]
        for k, pos in enumerate(valid):
            out[pos] = float(np.mean([s["fold_growth"][k] for s in sels]))
        if trades_out is not None:
            for k, pos in enumerate(valid):
                trades_out.append(float(np.mean([s["fold_trades"][k] for s in sels])))
    return out


def _mean(aps):
    vals = [a for a in aps if a is not None]
    return float(np.mean(vals)) if vals else None


def _robust_gain(cand, base):
    """(mean_gain, fraction of folds where the candidate beats base). Only folds scored in
    BOTH are compared — same events/folds, so this is apples-to-apples."""
    pairs = [(c, b) for c, b in zip(cand, base) if c is not None and b is not None]
    if not pairs:
        return None, 0.0
    gain = float(np.mean([c - b for c, b in pairs]))
    win = float(np.mean([c > b for c, b in pairs]))
    return gain, win


def search_ticker(ticker):
    """Forward feature selection with an OVERFIT GATE. A feature is a candidate only if its
    marginal gain >= min_feature_gain AND it beats core in >= min_fold_win_frac of folds
    (a robust edge, not one lucky fold). Forward selection then maximizes a COMPLEXITY-
    PENALIZED score (mean CV − complexity_penalty·n_added), so a feature is added only when
    its gain outweighs the penalty — useless features are never bolted on. Returns
    {base, best_cv, subset}; the caller applies only if best_cv beats base by min_gain."""
    prep = _prep(ticker)
    if prep is None:
        return None
    base_trades_folds = []
    base = evaluate(prep, [], trades_out=base_trades_folds)
    base_mean = _mean(base)
    if base_mean is None:
        return None
    # A3 economic-power floor (mirrors xgb/tools/feature_search.py): every gain criterion is a
    # function of trades — a baseline with (almost) no Train-CV trades cannot support selection,
    # so skip the whole candidate sweep instead of burning ~25 single-feature evals for a
    # guaranteed-empty subset. The floor is result-affecting, so it lives in config.json
    # (recipe identity) — a missing key fails closed, never a silent code default (#24).
    base_trades = int(round(sum(base_trades_folds)))
    min_trades = int(SC["min_trades"])
    if base_trades_folds and base_trades < min_trades:
        return {"base": base_mean, "best_cv": base_mean, "subset": [],
                "thin_trades": True, "cv_trades": base_trades, "families": {},
                "stop_reason": "baseline below min_trades (A3 economic-power floor)"}
    mfg, pen, mwf = SC["min_feature_gain"], SC["complexity_penalty"], SC["min_fold_win_frac"]
    # candidate pool: features with a robust marginal edge over core (gates UNCHANGED); per-fold
    # deltas kept for the golden one-SE plateau per family
    survivor_stats = []
    for i in optional_ids():
        cand = evaluate(prep, [i])
        gain, win = _robust_gain(cand, base)
        if gain is not None and gain >= mfg and win >= mwf:
            deltas = [(c - b) if (c is not None and b is not None) else None
                      for c, b in zip(cand, base)]
            survivor_stats.append({"id": i, "mean": gain, "folds": deltas})
    # golden policy (docs/METHODOLOGY.md): family representatives (simplest
    # one-SE plateau member) first, ordered by family strength; near-duplicate variants of the
    # same relationship enter greedy last and only via the marginal-vs-current-subset rule
    fam_of = golden.load_families(FAMILIES_PATH, optional_ids())
    id2name = P.optional_id_to_name()
    formulas = {**F.OPTIONAL_FEATURE_FORMULAS, **F.XAS_FEATURE_FORMULAS,
                **{n: s["expr"] for n, s in F.proposed_registry(PROPOSED).items()
                   if not n.startswith("_")}}
    simplicity = {i: len(str(formulas.get(id2name.get(i), id2name.get(i, "")))) for i in fam_of}
    pool, fam_report = (golden.order_pool(survivor_stats, fam_of, simplicity)
                        if survivor_stats else ([], {}))
    # greedy forward: add a feature only if it lifts the penalized score AND beats the CURRENT
    # subset in a majority of folds. cur_folds tracks the current subset's per-fold CV (no re-eval).
    subset, best_mean, best_pen, cur_folds = [], base_mean, base_mean, base
    for i in pool:
        if len(subset) >= int(SC["max_select"]):
            break
        cand = evaluate(prep, subset + [i])
        cand_mean = _mean(cand)
        if cand_mean is None:
            continue
        _, win_vs_subset = _robust_gain(cand, cur_folds)
        pen_score = cand_mean - pen * (len(subset) + 1)
        if pen_score > best_pen + 1e-6 and win_vs_subset >= mwf:
            subset, best_mean, best_pen, cur_folds = subset + [i], cand_mean, pen_score, cand
    return {"base": base_mean, "best_cv": best_mean, "subset": subset, "families": fam_report,
            "stop_reason": ("no candidate passed the marginal gates" if not survivor_stats
                            else "greedy converged over family representatives")}


def set_override(ticker, subset):
    """Set (subset non-empty) or clear (subset empty) this ticker's optional-feature override.
    A non-empty entry carries PROVENANCE (#17-18) — the recipe hash it was selected under —
    so the sealed run can refuse a stale override instead of silently sealing an
    incomparable manifest. File-locked read-modify-write; atomic temp + os.replace."""
    import oos_ledger
    with open(OVERRIDES_LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        ov = json.loads(OVERRIDES.read_text(encoding="utf-8")) if OVERRIDES.exists() else {}
        if subset:
            ov[ticker] = {"selected_optional_ids": sorted(int(i) for i in subset),
                          "provenance": {"recipe_hash": oos_ledger.lstm_recipe_hash(),
                                         "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
        else:
            ov.pop(ticker, None)
        tmp = OVERRIDES.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ov, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, OVERRIDES)


def process_ticker(ticker, min_gain, first_time):
    """One worker's complete job for a ticker (runs in a pool worker): search, set/clear the
    override, and — if the subset gained OR this is the ticker's first search — run the asset
    once to write its OOS row. Returns the verdict for the coordinator to record. Self-contained
    and side-effect-safe under parallelism (override write is file-locked; run_asset writes a
    disjoint Assets/<T>/ and a WAL-safe oos_metrics row)."""
    try:
        res = search_ticker(ticker)
    except Exception as e:
        return {"ticker": ticker, "error": repr(e)[:200]}
    if res is None:
        return {"ticker": ticker, "thin": True}
    gain = bool(res["subset"]) and res["best_cv"] >= res["base"] + min_gain
    set_override(ticker, res["subset"] if gain else [])
    verdict = ("applied" if gain else
               "thin_trades" if res.get("thin_trades") else "core_only")
    stop = res.get("stop_reason", "")
    if not gain and res.get("subset"):
        stop = "final subset below min_gain — override cleared, core_only"
    report_path = os.environ.get("GOLDEN_ROUND_REPORT")
    if report_path:                                     # golden §10 minimal round report
        import oos_ledger
        golden.append_jsonl(report_path, golden.round_report(
            ticker=ticker, model="lstm", recipe_hash=oos_ledger.lstm_recipe_hash(),
            baseline_score=round(res["base"], 6), baseline_trades=res.get("cv_trades"),
            families_report=res.get("families", {}), selected=res["subset"] if gain else [],
            final_gain=round(res["best_cv"] - res["base"], 6) if gain else 0.0,
            verdict=verdict, stop_reason=stop))
    ran = False
    if gain or first_time:
        run_asset(ticker)
        ran = True
    return {"ticker": ticker, "base": res["base"], "best_cv": res["best_cv"],
            "subset": res["subset"], "applied": int(gain), "ran": ran,
            "thin_trades": bool(res.get("thin_trades")), "cv_trades": res.get("cv_trades")}


def run_asset(ticker):
    return subprocess.run([sys.executable, str(ROOT / "run_asset.py"), f"TICKER={ticker}"],
                          env=dict(os.environ, OMP_NUM_THREADS=_TT,
                                   OOS_SEAL_DRIVER="lstm/feature_search.py"),
                          cwd=str(ROOT)).returncode


def db():
    con = sqlite3.connect(str(STATE), timeout=30.0)
    con.execute("create table if not exists searched(ticker text primary key, base_cv real, "
                "best_cv real, subset text, applied integer, pool_size integer, ts real)")
    try:
        con.execute("alter table searched add column pool_size integer")   # migrate older dbs
    except sqlite3.OperationalError:
        pass
    con.commit()
    return con


def _mark(con, t, base, best, subset, applied, pool):
    con.execute("insert or replace into searched(ticker,base_cv,best_cv,subset,applied,pool_size,ts) "
                "values(?,?,?,?,?,?,?)", (t, base, best, json.dumps(subset), applied, pool, time.time()))
    con.commit()


def status():
    con = db()
    d = dict(con.execute("select applied, count(*) from searched group by applied").fetchall())
    con.close()
    print(f"feature-search: {sum(d.values())}/{len(universe())} searched — "
          f"applied {d.get(1, 0)}, core-only {d.get(0, 0)}")


def _reband(con, t, res, applied, pool):
    _mark(con, t, res["base"] if res else None, res["best_cv"] if res else None,
          res["subset"] if res else [], applied, pool)


def main():
    if "--status" in sys.argv:
        return status()
    con = db()
    uni = universe()
    while True:                                        # continuous: re-drains proposals, reopens
        if STOP_FLAG.exists():                         # core-only tickers when the pool grows
            print("STOP.flag — exiting", flush=True)
            break
        ingest_proposals()
        ctl = read_control()
        if ctl.get("halt"):
            print("halt=true in search_control.json — standing down", flush=True)
            break
        paused = set(ctl.get("paused_tickers", []))
        min_gain = float(ctl.get("min_gain", MIN_GAIN))    # control overrides config
        pool_now = len(optional_ids())
        seen = {r[0]: (r[1] or 0, r[2] or 0) for r in
                con.execute("select ticker, applied, pool_size from searched").fetchall()}
        # to-do: never-searched, OR core-only whose search predates a now-larger feature pool
        todo = [t for t in uni if t not in paused and
                (t not in seen or (seen[t][0] == 0 and seen[t][1] < pool_now))]
        if not todo:
            print(f"feature-search: nothing to do (pool={pool_now}); idle — waiting for new "
                  f"proposals / a lower min_gain (make search-off to stop)", flush=True)
            for _ in range(12):                        # ~60 s idle, STOP-responsive
                if STOP_FLAG.exists():
                    break
                time.sleep(5)
            continue
        j = jobs()
        print(f"feature-search round: {len(todo)} tickers, {j} parallel workers "
              f"(pool={pool_now}, min_gain={min_gain})", flush=True)
        first_time = {t: t not in seen for t in todo}
        done, pool_cur = 0, pool_now
        with ProcessPoolExecutor(max_workers=j, initializer=_worker_init,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            futs = {pool.submit(process_ticker, t, min_gain, first_time[t]): t for t in todo}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:                     # a dead worker (OOM / crash) — park + go on
                    r = {"ticker": t, "error": repr(e)[:200]}
                done += 1
                if "error" in r:
                    print(f"{t}: SKIPPED ({r['error']})", flush=True)
                    _reband(con, t, None, 0, pool_cur)
                elif r.get("thin"):
                    _reband(con, t, None, 0, pool_cur)
                else:
                    if r["applied"]:
                        tag = f"APPLIED {len(r['subset'])} feat (CV {r['base']:.3f}->{r['best_cv']:.3f})"
                    elif r.get("thin_trades"):
                        tag = f"core-only, thin ({r.get('cv_trades')} CV trades < floor — selection skipped)"
                    else:
                        tag = f"core-only (CV {r['base']:.3f})"
                    print(f"{t}: {tag}", flush=True)
                    _mark(con, t, r["base"], r["best_cv"], r["subset"], r["applied"], pool_cur)
                # frequently absorb the Opus agent's proposals mid-pass — a newly registered
                # feature is on disk within ~12 tickers and the next workers search WITH it;
                # pool_cur tracks the live pool so re-open bookkeeping stays correct.
                if done % 12 == 0:
                    ingest_proposals()
                    pool_cur = len(optional_ids())
                if done % 20 == 0:
                    subprocess.run([sys.executable, str(ROOT / "build_dashboard.py")], cwd=str(ROOT))
                    print(f"--- {done}/{len(todo)} searched (pool={pool_cur})", flush=True)
                if STOP_FLAG.exists() or read_control().get("halt"):
                    print("stop requested — cancelling remaining", flush=True)
                    for f2 in futs:
                        f2.cancel()
                    break
        subprocess.run([sys.executable, str(ROOT / "build_dashboard.py")], cwd=str(ROOT))
    con.close()


if __name__ == "__main__":
    main()
