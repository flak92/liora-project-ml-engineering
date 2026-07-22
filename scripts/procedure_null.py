#!/usr/bin/env python3
"""Rung 5 — the procedure-level null. Does the WHOLE acceptance rule survive its own search?

The rotation-level null in `max_null.py` answers a smaller question than the one that matters. It
controls "45 candidates, one maximum, one rotation". The procedure being defended is larger: four
rotations, a unit must recur across at least two of them, a majority of its confirmation deltas
must be positive, and their median must clear the complexity charge. A null that never reproduces
those clauses cannot control the multiplicity they introduce.

So the unit here is `ticker x outer_fold x permutation_id`, and the decisive design point is:

    the permutation belongs to outer-train, not to a rotation and not to an inner fold.

One permutation produces one null realisation of the whole outer-train optional matrix; the four
rotations are then merely different VIEWS of that same realisation. This dissolves a problem that
has no solution otherwise — purged walk-forward inner folds overlap, a row belongs to several of
them at once, so "permute within a fold" is not even well defined. We do not try to protect the
boundary of each overlapping fold. We protect the boundary that exists: outer-train against
outer-validation, with purge and embargo applied afterwards when the folds are cut out of the
permuted object. Both arms consume the same realisation, so their results are paired.

Each permutation returns exactly one scalar per arm, computed by `acceptance.T` — the same
function, the same code path, that produced the real verdict.

Three null constructions, selectable with `--null`, differing ONLY in how the optional matrix is
made uninformative. Everything downstream is byte-identical between them, which is what makes them
comparable:

  a1  global block permutation over outer-train. Blocks are runs of consecutive event rows
      spanning at least L = 24 bars; block order is permuted; all 45 columns move together.
  a2  grouped block permutation. Outer-train is cut into G = 4 chronological macro-segments of
      equal bar span and blocks are permuted only WITHIN a segment. A1 can carry a block from a
      distant volatility regime to the start of the window: local autocorrelation survives but the
      local distribution does not. A2 keeps the regime and is the more defensive of the two.
  b   conditional residual null. A1 and A2 destroy `optional <-> outcome` AND `optional <-> core`,
      which makes them marginal nulls, not the conditional `X_optional _||_ Y | X_core` we mean.
      Here a cross-fitted g estimates `optional_hat = g(core)`, the residual is grouped-block
      permuted, and the matrix is rebuilt as `optional_hat + residual_perm`, so the dependence on
      core is retained. Because `P(X_optional | X_core)` is ESTIMATED rather than known, this is a
      sensitivity check, not an exact conditional randomization test, and must never be reported
      as one.

Resumable at the permutation. Every finished unit is a ledger line, so a machine that dies at
hour four resumes at hour four. Halting is cooperative: `control.json` is read between units, never
inside one, so the ledger is always consistent.

    python3 scripts/procedure_null.py --null a1 --jobs 4
    python3 scripts/procedure_null.py --null a1 --folds 3          # smoke gate
    python3 scripts/procedure_null.py --null b --survivors-from xgb/data/procedure_null_a1.json
"""
import argparse
import hashlib
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

import acceptance as ACC                                                   # noqa: E402
from artifact_io import read_json, write_json_atomic                       # noqa: E402
from ledger import Ledger                                                  # noqa: E402

CROSSFIT = XGB / "data" / "crossfit_selection.json"
REGISTER = XGB / "data" / "feature_utility.json"
CONTRACT = ROOT / "config" / "feature_discovery_contract.json"

_C = json.loads(CONTRACT.read_text(encoding="utf-8"))["max_null"]
M_MAX = int(_C["permutations_max"])                       # 50
PASS_B = 4                                                # b <= 4  ->  p_mc <= 5/51
FUTILITY_B = PASS_B + 1                                   # b = 5   ->  p_mc >= 6/51, cannot pass
L_BLOCK = int(_C["block_length"]["value_bars"])           # 24 bars
MAX_FRAC = float(_C["block_length"]["max_block_fraction_of_fold"])
G_SEGMENTS = 4                                            # macro-segments for the grouped nulls
MIN_DISPLACED = 0.5                                       # at least half the blocks must move
EMBARGO_BARS = 35
MODE = "quantile"
SEED = 42


# ---------------------------------------------------------------------------------------------
# permutation machinery
# ---------------------------------------------------------------------------------------------

def blocks_by_bar_time(t0s, rows, L):
    """Runs of consecutive event rows spanning at least L bars.

    Bar time, not row count: events cluster, so a fixed number of rows would be a short block in a
    busy stretch and a long one in a quiet stretch. The final run may be shorter than L and is kept
    whole, so every row belongs to exactly one block.
    """
    out, cur, start = [], [], t0s[rows[0]]
    for j in rows:
        if t0s[j] - start >= L and cur:
            out.append(cur)
            cur, start = [], t0s[j]
        cur.append(j)
    if cur:
        out.append(cur)
    return out


def segments_by_bar_span(t0s, rows, g):
    """G chronological macro-segments of equal BAR span (not equal row count).

    Equal bar span is what makes a segment a regime: an equal-row split would make a quiet stretch
    span years and a busy one span weeks, which is the opposite of holding the local distribution
    fixed.
    """
    lo, hi = t0s[rows[0]], t0s[rows[-1]]
    width = max(1, (hi - lo + 1) / g)
    segs = [[] for _ in range(g)]
    for j in rows:
        segs[min(g - 1, int((t0s[j] - lo) / width))].append(j)
    return [s for s in segs if s]


def draw_permutation(blocks, rng, groups=None):
    """A block reordering, rejected unless it actually moves things.

    `groups` restricts reordering to within a group of block indices (the grouped nulls). With ~200
    blocks the identity has probability 1/200! and duplicates are equally unthinkable, but the
    contract asks for the assertion rather than the probability argument, and the cost is nothing.
    A permutation that leaves more than half the blocks in place is redrawn: it would be a
    permutation in name while testing almost the original alignment.
    """
    n = len(blocks)
    for _ in range(64):
        order = list(range(n))
        if groups is None:
            order = list(rng.permutation(n))
        else:
            for g in groups:
                sub = list(rng.permutation(g))
                for slot, src in zip(g, sub):
                    order[slot] = src
        displaced = sum(1 for i, s in enumerate(order) if i != s)
        if displaced >= MIN_DISPLACED * n:
            return order, displaced
    raise RuntimeError("nie udało się wylosować permutacji o wymaganym przemieszczeniu — "
                       "za mało bloków? sprawdź n_blocks")


def index_hash(order):
    """Identifies the realisation. Recorded so a run is auditable and so a repeat is detectable."""
    return hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()[:16]


def apply_block_order(matrix, blocks, order):
    """Reorder the block CONTENTS. Columns move together, so optional-optional correlation and
    each column's autocorrelation survive; only the alignment to the outcome dies."""
    src = [j for b in order for j in blocks[b]]
    dst = [j for b in blocks for j in b]
    out = matrix.copy()
    out[dst, :] = matrix[src, :]
    return out


# ---------------------------------------------------------------------------------------------
# the three null constructions
# ---------------------------------------------------------------------------------------------

def _grouped_blocks(t0s, rows, L, g):
    """Blocks plus the index groups that a grouped permutation may shuffle within."""
    blocks, groups, k = [], [], 0
    for seg in segments_by_bar_span(t0s, rows, g):
        sb = blocks_by_bar_time(t0s, seg, L)
        blocks.extend(sb)
        groups.append(list(range(k, k + len(sb))))
        k += len(sb)
    return blocks, groups


def build_null(kind, ctx, rows, rng):
    """Return (permuted optional matrix, provenance) for one permutation of one outer fold."""
    import numpy as np

    t0s, L = ctx["t0s"], ctx["L"]
    optional = ctx["optional"]

    if kind == "a1":
        blocks = blocks_by_bar_time(t0s, rows, L)
        order, disp = draw_permutation(blocks, rng)
        return apply_block_order(optional, blocks, order), {
            "kind": "a1", "n_blocks": len(blocks), "displaced": disp,
            "index_hash": index_hash(order)}

    blocks, groups = _grouped_blocks(t0s, rows, L, G_SEGMENTS)
    order, disp = draw_permutation(blocks, rng, groups=groups)
    prov = {"kind": kind, "n_blocks": len(blocks), "displaced": disp,
            "segments": len(groups), "blocks_per_segment": [len(g) for g in groups],
            "index_hash": index_hash(order)}

    if kind == "a2":
        return apply_block_order(optional, blocks, order), prov

    # b — permute the part of optional that core does NOT explain, then put core's part back.
    fitted, resid = ctx["fitted"], ctx["residual"]
    perm = apply_block_order(resid, blocks, order)
    rebuilt = fitted + perm
    prov["alpha"] = ctx["ridge_alpha"]
    prov["residual_share"] = ctx["residual_share"]
    prov["_marginal_note"] = ("fitted + permuted residual does not reproduce each column's original "
                              "marginal; the drift is recorded in the artifact rather than hidden")
    prov["marginal_drift"] = float(np.abs(np.nanstd(rebuilt, axis=0)
                                          - np.nanstd(optional, axis=0)).mean())
    return rebuilt, prov


def fit_conditional(ctx, rows):
    """Cross-fitted g: core -> all 45 optional columns, for the conditional null.

    Cross-fitted on contiguous chronological blocks with the pipeline's own embargo, so a row's
    fitted value never comes from a model that saw that row, and neighbouring rows inside the
    embargo cannot stand in for it either. The inner-fold structure is deliberately NOT reused:
    those folds overlap, so "trained without this row" would be unenforceable. Nothing outside
    outer-train is ever touched.

    Alpha is chosen by held-out reconstruction MSE of the FEATURES. No outcome, no label and no
    trading result enters the choice, so the nuisance model cannot smuggle in the thing being
    tested.
    """
    import numpy as np
    from sklearn.linear_model import Ridge

    core, optional, t0s = ctx["core_matrix"], ctx["optional"], ctx["t0s"]
    rows = np.asarray(rows)
    K = 5
    edges = np.linspace(0, len(rows), K + 1).astype(int)
    folds = [rows[edges[i]:edges[i + 1]] for i in range(K)]

    def _train_idx(te):
        lo, hi = t0s[te[0]] - EMBARGO_BARS, t0s[te[-1]] + EMBARGO_BARS
        return np.array([j for j in rows if not (lo <= t0s[j] <= hi)])

    # A handful of optional columns (the 1w multi-timeframe features) carry NaN. XGBoost handles
    # NaN natively, so A1 and A2 pass it straight through — but sklearn's Ridge rejects NaN in its
    # target. Impute the FIT target with train-column means so g can fit; the residual is then taken
    # against the ORIGINAL optional, so `optional - fitted` stays NaN exactly where the feature was
    # missing. Block-permuting that residual and adding fitted back reproduces the same NaN COUNT,
    # its positions shuffled — which is what the null intends anyway. Core has no NaN, so the Ridge
    # inputs and predictions are always defined.
    def _fill(y):
        if not np.isnan(y).any():
            return y
        y = y.copy()
        cm = np.nanmean(y, axis=0)
        cm = np.where(np.isnan(cm), 0.0, cm)              # a column that is all-NaN on this fold -> 0
        idx = np.where(np.isnan(y))
        y[idx] = np.take(cm, idx[1])
        return y

    grid = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best, best_mse = None, np.inf
    splits = [(_train_idx(te), te) for te in folds]
    for a in grid:
        errs = []
        for tr, te in splits:
            if len(tr) < 50:
                continue
            mu, sd = core[tr].mean(0), core[tr].std(0) + 1e-12
            m = Ridge(alpha=a, fit_intercept=True)
            m.fit((core[tr] - mu) / sd, _fill(optional[tr]))
            se = (m.predict((core[te] - mu) / sd) - optional[te]) ** 2
            errs.append(float(np.nanmean(se)))            # ignore cells the feature never had
        if errs and np.mean(errs) < best_mse:
            best, best_mse = a, float(np.mean(errs))

    fitted = np.zeros_like(optional)
    for tr, te in splits:
        if len(tr) < 50:
            fitted[te] = _fill(optional[tr]).mean(0) if len(tr) else 0.0
            continue
        mu, sd = core[tr].mean(0), core[tr].std(0) + 1e-12
        m = Ridge(alpha=best, fit_intercept=True)
        m.fit((core[tr] - mu) / sd, _fill(optional[tr]))
        fitted[te] = m.predict((core[te] - mu) / sd)

    resid = optional - fitted                              # NaN preserved where optional was NaN
    var_o = float(np.nansum(np.nanvar(optional[rows], axis=0)))
    share = float(np.nansum(np.nanvar(resid[rows], axis=0)) / var_o) if var_o > 0 else 1.0
    return {"fitted": fitted, "residual": resid, "ridge_alpha": best,
            "residual_share": round(share, 4), "recon_mse": round(best_mse, 8),
            "nan_columns": int((np.isnan(optional).any(axis=0)).sum())}


# ---------------------------------------------------------------------------------------------
# the search, reproduced in full under each permutation
# ---------------------------------------------------------------------------------------------

def one_rotation(ctx, dfp, inner, r, params, base, base_conf):
    """One rotation's flat and hierarchical pick, on the permuted frame."""
    import crossfit_selection as CF
    import feature_search as FS
    import golden

    ticker = ctx["ticker"]
    disc = [f for i, f in enumerate(inner) if i != r]
    conf = inner[r]
    dfx, tev = ctx["dfx"], ctx["tev"]

    scores = {}
    for cid in ctx["cands"]:
        d = CF.discover(dfx, dfp, tev, disc, params, FS._names(ticker, [cid]), SEED)
        if d is None:
            continue
        scores[int(cid)] = {"gain": d["mean"] - base["mean"], "q": d["q"],
                            "folds": [c - b for c, b in zip(d["fold_growth"], base["fold_growth"])]}
    if not scores:
        return None

    out = {"confirmation_fold": r, "arms": {}}
    top = max(scores, key=lambda i: scores[i]["gain"])
    c = CF.confirm(dfx, dfp, tev, conf, params, FS._names(ticker, [top]), SEED, scores[top]["q"])
    out["arms"]["flat"] = {"picked": top, "family": ctx["fam_of"].get(top),
                           "confirm_delta": None if c is None else c["growth"] - base_conf["growth"]}

    byfam = {}
    for i, s in scores.items():
        byfam.setdefault(ctx["fam_of"][i], []).append({"id": i, "mean": s["gain"], "folds": s["folds"]})
    fam_best = {f: max(v, key=lambda x: x["mean"])["mean"] for f, v in byfam.items()}
    fam = max(fam_best, key=fam_best.get)
    plateau, best_id, _ = golden.one_se_set(byfam[fam])
    rep = golden.pick_representative(plateau, ctx["simplicity"]) or best_id
    c = CF.confirm(dfx, dfp, tev, conf, params, FS._names(ticker, [rep]), SEED, scores[rep]["q"])
    out["arms"]["hierarchical"] = {"picked": int(rep), "family": fam,
                                   "confirm_delta": None if c is None else c["growth"] - base_conf["growth"]}
    return out


def one_permutation(ctx, ofold, perm_id):
    """One null realisation of outer-train, seen through all four rotations, collapsed to two scalars."""
    import numpy as np
    import pandas as pd

    inner = ctx["inner"][ofold]
    params = ctx["params"][ofold]
    rows = ctx["rows"][ofold]

    rng = np.random.default_rng(SEED + 100003 * ofold + perm_id)
    mat, prov = build_null(ctx["null_kind"], dict(ctx, L=ctx["L"]), rows, rng)

    dfp = ctx["dfb"].copy(deep=False)
    for c, col in enumerate(ctx["opt_cols"]):
        dfp[col] = mat[:, c]

    rots = []
    for r in range(len(inner)):
        got = one_rotation(ctx, dfp, inner, r, params,
                           ctx["base"][ofold][r], ctx["base_conf"][ofold][r])
        if got:
            rots.append(got)

    t_flat, _ = ACC.T(rots, "flat")
    t_hier, _ = ACC.T(rots, "hierarchical")

    # The rotation-level statistic falls out of the same work, so the diagnostic that would
    # otherwise cost a separate 7.8 core-hour run is free here — and better. Run separately it
    # would use its own permutations, so "how many conclusions change when the whole rule is
    # aggregated instead of tested rotation by rotation" would compare two independent draws.
    # Taken from the same realisation the comparison is PAIRED, which is the only way that
    # question has a clean answer.
    per_rot = [[None if r["arms"]["flat"]["confirm_delta"] is None
                else round(float(r["arms"]["flat"]["confirm_delta"]), 6),
                None if r["arms"]["hierarchical"]["confirm_delta"] is None
                else round(float(r["arms"]["hierarchical"]["confirm_delta"]), 6)]
               for r in rots]
    return {"T_flat": round(float(t_flat), 8), "T_hierarchical": round(float(t_hier), 8),
            "rotation_deltas": per_rot, "rotations_scored": len(rots), **prov}


# ---------------------------------------------------------------------------------------------
# per-ticker driver
# ---------------------------------------------------------------------------------------------

def prepare(ticker, scope, null_kind, run_id, block_mult):
    """Everything that does not change between permutations, computed once."""
    import numpy as np
    import crossfit_selection as CF
    import feature_search as FS
    import golden
    import nested_validation as NV
    import pipeline as P

    reg = json.loads(REGISTER.read_text(encoding="utf-8"))["tables"][ticker]
    scratch = runtime_init.scratch_dir(run_id, ticker)
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, scratch / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]

    core_names = FS._names(ticker, [])
    opt_cols = [c for c in FS._names(ticker, cands) if c not in set(core_names)]
    ctx = {"ticker": ticker, "dfx": dfx, "dfb": dfb, "tev": tev, "t0s": t0s,
           "cands": cands, "opt_cols": opt_cols, "null_kind": null_kind,
           "L": L_BLOCK * block_mult,
           "optional": dfb[opt_cols].to_numpy(float),
           "core_matrix": dfb[core_names].to_numpy(float),
           "fam_of": golden.load_families(FS.FAMILIES_PATH, cands),
           "simplicity": {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                          for ns, rr in P.FEATURE_REGISTRIES.items() if ns != "1h"
                          for f in rr["features"] if bool(f.get("implemented", True))},
           "params": {}, "inner": {}, "rows": {}, "base": {}, "base_conf": {}}

    folds = {ofold for (tk, ofold) in scope if tk == ticker}
    for f in reg["folds"]:
        i = f["outer_fold"]
        if i not in folds or f.get("stage") == "no_viable_model":
            continue
        bi = dict(bnds, train_end_idx=int(f["inner_train_end_idx"]))
        inner = P.purged_wf_folds(t0s, bi["train_start_idx"], bi["train_end_idx"])
        assert all(max(t0s[j] for j in va) < bnds["oos_start_idx"] for _, va in inner if va), \
            "procedure_null: an inner val fold reaches OOS (purge invariant violated)"
        ctx["params"][i] = f["frozen_params"]
        ctx["inner"][i] = inner
        ctx["rows"][i] = sorted({j for tr, va in inner for j in list(tr) + list(va)})
        # Core is untouched by every null, so its discovery and confirmation are invariant across
        # all fifty permutations. Computing them once is exact, not an approximation.
        ctx["base"][i], ctx["base_conf"][i] = {}, {}
        for r in range(len(inner)):
            disc = [x for k, x in enumerate(inner) if k != r]
            b = CF.discover(dfx, dfb, tev, disc, ctx["params"][i], core_names, SEED)
            ctx["base"][i][r] = b
            ctx["base_conf"][i][r] = CF.confirm(dfx, dfb, tev, inner[r], ctx["params"][i],
                                                core_names, SEED, b["q"])

    if null_kind == "b":
        allrows = sorted({j for rr in ctx["rows"].values() for j in rr})
        ctx.update(fit_conditional(ctx, allrows))
    return ctx


def halted(control):
    if not control:
        return False
    d = read_json(control, {}) or {}
    return bool(d.get("halt"))


def run_ticker(job):
    """All scoped outer folds of one ticker, checkpointing every permutation."""
    ticker, scope, real, null_kind, run_id, block_mult, ledger_path, control, m_max = job
    led = Ledger(ledger_path)
    stage = f"null_{null_kind}"
    # One pass over the ledger, not one per permutation: a resume late in a long run would
    # otherwise re-read and re-parse the whole file for every unit it skips.
    cache = {Ledger.key(u): p for u, p in led.payloads(stage)}

    t_start = time.time()
    ctx = prepare(ticker, scope, null_kind, run_id, block_mult)

    folds_out = []
    for (tk, ofold) in sorted(scope):
        if tk != ticker or ofold not in ctx["inner"]:
            continue
        arms = {a: v for a, v in real[(tk, ofold)].items()}
        state = {a: {"exceed": 0, "null": [], "stopped_at": None} for a in arms}
        executed = 0
        prov_last = {}
        rot_deltas, hashes, disp_fracs = [], [], []

        # Replay whatever the ledger already holds, so a resume neither repeats work nor loses
        # the exceedance counts that drive futility stopping.
        for pid in range(m_max):
            unit = {"ticker": ticker, "outer_fold": ofold, "permutation_id": pid}
            cached = cache.get(Ledger.key(unit))
            if cached is None:
                if halted(control):
                    break
                if all(s["stopped_at"] is not None for s in state.values()):
                    break
                led.append(stage, unit, "running")
                _u0 = time.time()
                cached = one_permutation(ctx, ofold, pid)
                led.append(stage, unit, "completed", payload={
                    "T_flat": cached["T_flat"], "T_hierarchical": cached["T_hierarchical"],
                    "rotation_deltas": cached["rotation_deltas"],
                    "index_hash": cached["index_hash"], "n_blocks": cached["n_blocks"],
                    "displaced": cached.get("displaced"),
                    "seconds": round(time.time() - _u0, 2)})    # per-unit cost for cost_report.py
            executed = pid + 1
            prov_last = cached
            rot_deltas.append(cached.get("rotation_deltas"))
            # The displacement fraction is summarised per fold, not sampled from the last
            # permutation: draw_permutation guarantees each draw moves >= MIN_DISPLACED of the
            # blocks or raises, so a completed unit already implies the invariant. Persisting it
            # per unit lets the gate re-check the fold's WORST permutation even after a resume
            # rebuilds the fold from cache. Legacy units written before this field contribute None.
            d, nb = cached.get("displaced"), cached.get("n_blocks")
            if d is not None and nb:
                disp_fracs.append(d / nb)
            h = cached.get("index_hash")
            if h in hashes:
                raise RuntimeError(
                    f"powtórzona permutacja {h} w {ticker}/{ofold} — losowanie nie jest niezależne")
            hashes.append(h)
            for a, s in state.items():
                if s["stopped_at"] is not None:
                    continue
                v = cached["T_flat"] if a == "flat" else cached["T_hierarchical"]
                s["null"].append(v)
                if v >= arms[a]["real_statistic"]:
                    s["exceed"] += 1
                if s["exceed"] >= FUTILITY_B:
                    s["stopped_at"] = executed

        prov = {k: v for k, v in prov_last.items()
                if k not in ("T_flat", "T_hierarchical", "rotation_deltas")}
        # Worst displacement over the whole fold — reliably persisted, so the gate survives a
        # resume. None only when every unit predates the field (a pure-cache legacy replay), which
        # the gate treats as "guaranteed at generation, not re-checkable" rather than a violation.
        prov["min_displaced_fraction"] = round(min(disp_fracs), 4) if disp_fracs else None
        row = {"ticker": ticker, "outer_fold": ofold, "permutations_executed": executed,
               "provenance": prov,
               "index_hashes": hashes,
               "rotation_level_diagnostic": {
                   "_role": ("paired by-product: the rotation-level null from the SAME permutations, "
                             "so 'how many conclusions change when the whole rule is aggregated' "
                             "is a paired comparison rather than two independent draws"),
                   "_must_not": "be used as the max-null verdict",
                   "null_deltas_per_permutation": rot_deltas},
               "arms": {}}
        for a, s in state.items():
            n = s["stopped_at"] or executed
            b = s["exceed"]
            if b >= FUTILITY_B:
                v = {"verdict": "rejected_early", "permutations_executed": n, "exceedances": b,
                     "final_p_lower_bound": round((1 + b) / (m_max + 1), 6)}
            elif executed >= m_max:
                v = {"verdict": "passed", "permutations_executed": n, "exceedances": b,
                     "p_mc": round((1 + b) / (m_max + 1), 6)}
            else:
                v = {"verdict": "incomplete", "permutations_executed": n, "exceedances": b}
            row["arms"][a] = dict(arms[a], **v, null_statistics=s["null"])
        folds_out.append(row)

    return {"ticker": ticker, "seconds": round(time.time() - t_start, 1), "folds": folds_out}


# ---------------------------------------------------------------------------------------------

def real_statistics(path, survivors_from=None):
    """T_real per (ticker, outer fold, arm), read from the run that actually happened.

    Only folds the procedure accepted are tested: `T_real <= 0` means the acceptance contract
    already returned nothing there, and a null can reject an acceptance but never create one.
    """
    doc = read_json(path)
    keep = None
    if survivors_from:
        prev = read_json(survivors_from)
        keep = {(f["ticker"], f["outer_fold"], a)
                for t in prev["tables"].values() for f in t["folds"]
                for a, v in f["arms"].items() if v["verdict"] == "passed"}
    out = {}
    for ticker, rec in doc["tables"].items():
        for f in rec["folds"]:
            for arm in ("flat", "hierarchical"):
                v = f["verdict"][arm]
                if not v.get("accepted") or float(v.get("T", 0)) <= 0:
                    continue
                if keep is not None and (ticker, f["outer_fold"], arm) not in keep:
                    continue
                out.setdefault((ticker, f["outer_fold"]), {})[arm] = {
                    "unit": v["unit"], "real_statistic": float(v["T"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--null", choices=("a1", "a2", "b"), default="a1")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--folds", type=int, default=0, help="smoke gate: cap the number of outer folds")
    ap.add_argument("--permutations", type=int, default=M_MAX)
    ap.add_argument("--block-mult", type=int, default=1)
    ap.add_argument("--survivors-from", default=None,
                    help="restrict to arms that passed an earlier null")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--control", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    real = real_statistics(CROSSFIT, args.survivors_from)
    keys = sorted(real)
    if args.tickers:
        keys = [k for k in keys if k[0] in args.tickers]
    if args.folds:
        keys = keys[:args.folds]
    scope = set(keys)
    tickers = sorted({k[0] for k in scope})

    run_id = f"proc_null_{args.null}_L{args.block_mult}"
    run_dir = Path(args.run_dir) if args.run_dir else XGB / "data" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else XGB / "data" / f"procedure_null_{args.null}.json"
    ledger_path = run_dir / "ledger.jsonl"

    led = Ledger(ledger_path)
    orphans = led.reconcile_orphans(f"null_{args.null}")
    resumed = len(led.completed(f"null_{args.null}"))

    print(f"rung 5 — procedure-level null '{args.null}'\n"
          f"  {len(scope)} outer foldów, {len(tickers)} tabel, do {args.permutations} permutacji\n"
          f"  jednostka: ticker × outer_fold × permutation_id; permutacja należy do OUTER-TRAIN,\n"
          f"  rotacje są jej widokami — oba ramiona z tej samej realizacji\n"
          f"  blok {L_BLOCK * args.block_mult} barów"
          + (f", {G_SEGMENTS} makrosegmenty" if args.null in ("a2", "b") else "")
          + (f"\n  wznowienie: {resumed} jednostek z ledgera, {orphans} sierot domkniętych"
             if resumed or orphans else "") + "\n")

    from run_manifest import RunManifest
    man = RunManifest(run_dir / "run_manifest.json", stage=f"null_{args.null}",
                      workers=args.jobs).start({"scope_outer_folds": len(scope), "null": args.null,
                                                "block_bars": L_BLOCK * args.block_mult})

    jobs = [(t, scope, real, args.null, run_id, args.block_mult, ledger_path,
             args.control, args.permutations) for t in tickers]
    results = {}
    try:
        if args.jobs > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                for r in ex.map(run_ticker, jobs):
                    results[r["ticker"]] = r
                    _line(r, len(tickers), len(results))
        else:
            for i, j in enumerate(jobs, 1):
                r = run_ticker(j)
                results[r["ticker"]] = r
                _line(r, len(tickers), i)
    except BaseException as e:
        man.finish("FAILED_SAFE", extra={"error": f"{type(e).__name__}: {e}"})
        raise

    verdicts, perms = {}, {}
    for t, r in results.items():
        for f in r["folds"]:
            perms[f"{t}:{f['outer_fold']}"] = f["permutations_executed"]
            for a, v in f["arms"].items():
                verdicts[f"{t}:{f['outer_fold']}:{a}"] = v["verdict"]

    sha = write_json_atomic(out, {
        "contract": {"null": args.null, "permutations_max": args.permutations,
                     "pass_b": PASS_B, "futility_b": FUTILITY_B,
                     "block_length_bars": L_BLOCK * args.block_mult,
                     "segments": G_SEGMENTS if args.null in ("a2", "b") else None,
                     "unit": "ticker x outer_fold x permutation_id",
                     "statistic": "acceptance.T — the same code that produced the real verdict",
                     "controls": _C["controls"], "does_not_control": _C["does_not_control"],
                     "seed": SEED, "mode": MODE},
        "tables": results})
    man.finish("COMPLETED", verdicts=verdicts, artifacts=[out, ledger_path],
               permutations_executed=perms)
    print(f"\nwrote {out}  sha256 {sha[:16]}…")
    return 0


def _line(r, total, done):
    p = sum(1 for f in r["folds"] for a in f["arms"].values() if a["verdict"] == "passed")
    e = sum(1 for f in r["folds"] for a in f["arms"].values() if a["verdict"] == "rejected_early")
    i = sum(1 for f in r["folds"] for a in f["arms"].values() if a["verdict"] == "incomplete")
    print(f"  [{done}/{total}] {r['ticker']:<6} foldy={len(r['folds'])}  przeszło={p}  "
          f"odrzucone={e}" + (f"  niedokończone={i}" if i else "") + f"  ({r['seconds']:.0f}s)")


if __name__ == "__main__":
    raise SystemExit(main())
