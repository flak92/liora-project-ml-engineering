#!/usr/bin/env python3
"""Rung 5 — is the confirmed edge bigger than the maximum a search produces by itself?

Stage 3B cut provisional acceptances from 66/80 to 14 (flat) and 11 (hierarchical) by judging each
pick on a fold that played no part in choosing it. What cross-fitting cannot answer is how large a
confirmation delta the ACT OF SEARCHING produces when there is nothing to find: forty-five
candidates are ranked, one maximum is taken, and a maximum of noise is not centred on zero.

    H0: optional features carry no incremental information over the frozen core, and the observed
        confirmation delta is what searching a pool of candidates produces by itself.

The null block-shifts the whole optional-feature matrix. Core, labels, weights and the economic
outcome stay exactly where they are — permuting `Y_outcome` alone would leave the engine reading
unpermuted trade outcomes, so the null would not match the statistic being tested. Moving the
optional columns together preserves every feature's autocorrelation and every correlation among
them, and destroys only their temporal alignment to the outcome.

Blocks are cut in BAR time, `L_block = max(H, L_dependency) = 24` under the rule frozen in the
contract before any of this ran. Events are irregularly spaced, so a block is a run of consecutive
event rows spanning at least L bars; block lengths therefore vary in rows and the permutation is
a reordering of those runs. Nothing leaves the outer-train window: the confirmation fold is
permuted along with the discovery folds, because H0 says the features are uninformative everywhere,
and outer-validation and OOS are never read at all.

Each permutation reproduces the entire act of choosing — the ranking of all 45, the flat maximum,
the hierarchical `family -> one-SE plateau -> simplest representative` path, AND the operating-point
selection, because the chosen q is candidate-dependent in 60% of configurations. Both arms are
scored from the SAME permutations, so their results are paired, but each keeps its own maximum
distribution, its own exceedance counter and its own verdict. There is no shared threshold.

Only rotations supporting a provisional acceptance are tested. The null can reject an acceptance;
it can never rescue a candidate that failed confirmation, so running it where nothing was accepted
cannot change a decision.

    python3 scripts/max_null.py --jobs 4                 # the contract run, futility stopping on
    python3 scripts/max_null.py --full TDG ORLY          # rung 5C: all 50 regardless, to validate it
    python3 scripts/max_null.py --block-mult 2           # sensitivity at 2L
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

CROSSFIT = XGB / "data" / "crossfit_selection.json"
REGISTER = XGB / "data" / "feature_utility.json"
CONTRACT = ROOT / "config" / "feature_discovery_contract.json"
DEFAULT_OUT = XGB / "data" / "max_null.json"

# Frozen before the run, and read back from the contract so the two cannot drift apart.
_C = json.loads(CONTRACT.read_text(encoding="utf-8"))["max_null"]
M_MAX = int(_C["permutations_max"])                       # 50
PASS_B = 4                                                # b <= 4  ->  p_mc = (1+b)/51 <= 5/51
FUTILITY_B = PASS_B + 1                                   # b = 5   ->  p_mc >= 6/51, cannot pass
L_BLOCK = int(_C["block_length"]["value_bars"])           # 24 bars
MAX_FRAC = float(_C["block_length"]["max_block_fraction_of_fold"])
MODE = "quantile"
SEED = 42


def scope_from_crossfit(tables):
    """The rotations a null could still overturn: those supporting a provisional acceptance.

    A rotation is one permutation budget shared by both arms, so it is keyed by
    (ticker, outer fold, confirmation fold) and carries the real statistic of every arm it supports.
    """
    out = {}
    for ticker, rec in tables.items():
        for f in rec["folds"]:
            for arm, key in (("flat", "picked"), ("hierarchical", "family")):
                v = f["verdict"].get(arm, {})
                if not v.get("accepted"):
                    continue
                for rot in f["rotations"]:
                    a = rot["arms"][arm]
                    if a[key] != v["unit"] or a["confirm_delta"] is None:
                        continue
                    k = (ticker, f["outer_fold"], rot["confirmation_fold"])
                    out.setdefault(k, {})[arm] = {
                        "unit": v["unit"], "picked": a["picked"],
                        "real_statistic": float(a["confirm_delta"])}
    return out


def blocks_by_bar_time(t0s, rows, L):
    """Cut the rows into runs of consecutive events spanning at least L bars.

    The unit is deliberately bar time, not event count: events cluster, so a fixed number of rows
    would be a short block in a busy stretch and a long one in a quiet stretch. The last run may be
    shorter than L — it is kept whole rather than merged, so every row belongs to exactly one block.
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


def permute_optional(dfb, opt_cols, blocks, rng):
    """One block permutation of the optional-feature matrix; everything else stays put.

    The columns move TOGETHER — one row order applied to all 45 — so the cross-correlation
    structure among optional features survives intact. Only their alignment to the outcome dies.
    """
    src = [j for b in rng.permutation(len(blocks)) for j in blocks[b]]
    dst = [j for b in blocks for j in b]
    out = dfb.copy(deep=False)
    block = dfb[opt_cols].to_numpy(float)
    perm = block.copy()
    perm[dst, :] = block[src, :]
    for c, col in enumerate(opt_cols):
        out[col] = perm[:, c]
    return out


def search(dfx, dfb, tev, disc, conf, params, ticker, cands, fam_of, simplicity, base, base_conf):
    """The full act of choosing, reproduced verbatim: rank 45, take the flat maximum, walk the
    hierarchical family -> plateau -> representative path, and confirm each at its own q."""
    import crossfit_selection as CF
    import feature_search as FS
    import golden

    scores = {}
    for cid in cands:
        d = CF.discover(dfx, dfb, tev, disc, params, FS._names(ticker, [cid]), SEED)
        if d is None:
            continue
        scores[int(cid)] = {"gain": d["mean"] - base["mean"], "q": d["q"],
                            "folds": [c - b for c, b in zip(d["fold_growth"], base["fold_growth"])]}
    if not scores:
        return None

    stat = {}
    top = max(scores, key=lambda i: scores[i]["gain"])
    c = CF.confirm(dfx, dfb, tev, conf, params, FS._names(ticker, [top]), SEED, scores[top]["q"])
    stat["flat"] = None if c is None else c["growth"] - base_conf["growth"]

    byfam = {}
    for i, s in scores.items():
        byfam.setdefault(fam_of[i], []).append({"id": i, "mean": s["gain"], "folds": s["folds"]})
    fam_best = {f: max(v, key=lambda x: x["mean"])["mean"] for f, v in byfam.items()}
    fam = max(fam_best, key=fam_best.get)
    plateau, best_id, _ = golden.one_se_set(byfam[fam])
    rep = golden.pick_representative(plateau, simplicity) or best_id
    c = CF.confirm(dfx, dfb, tev, conf, params, FS._names(ticker, [rep]), SEED, scores[rep]["q"])
    stat["hierarchical"] = None if c is None else c["growth"] - base_conf["growth"]
    return stat


def verdict(arm_state, executed):
    """Report an early stop as what it is: a bound, never a fixed-budget p-value.

    Stopping at b = 5 is not a shortcut — it is deterministic. Even if every remaining permutation
    fell below the real value, p_mc = (1+b)/51 >= 6/51 = 0.1176 and the candidate cannot pass the
    contract. Quoting (1+b)/(1+n) here would understate the p-value of a test that chose when to
    stop, so what is reported instead is a lower bound on the p-value it would have had at 50.
    """
    b = arm_state["exceedances"]
    if b >= FUTILITY_B:
        return {"verdict": "rejected_early", "permutations_executed": executed, "exceedances": b,
                "final_p_lower_bound": round((1 + b) / (M_MAX + 1), 6)}
    if executed >= M_MAX:
        return {"verdict": "passed", "permutations_executed": executed, "exceedances": b,
                "p_mc": round((1 + b) / (M_MAX + 1), 6)}
    return {"verdict": "incomplete", "permutations_executed": executed, "exceedances": b}


def run_rotation(ctx, key, arms, full):
    """Up to 50 permutations for one rotation, both arms scored from each of them."""
    import numpy as np
    import crossfit_selection as CF
    import feature_search as FS

    ticker, ofold, r = key
    dfx, dfb, tev = ctx["dfx"], ctx["dfb"], ctx["tev"]
    params = ctx["params"][ofold]
    inner = ctx["inner"][ofold]
    disc = [f for i, f in enumerate(inner) if i != r]
    conf = inner[r]

    core = FS._names(ticker, [])
    base = CF.discover(dfx, dfb, tev, disc, params, core, SEED)
    base_conf = CF.confirm(dfx, dfb, tev, conf, params, core, SEED, base["q"])

    rows = sorted({j for tr, va in inner for j in list(tr) + list(va)})
    span = ctx["t0s"][rows[-1]] - ctx["t0s"][rows[0]]
    L = ctx["L"]
    clamped = L > span * MAX_FRAC
    if clamped:
        L = max(1, int(span * MAX_FRAC))
    blocks = blocks_by_bar_time(ctx["t0s"], rows, L)
    opt_cols = ctx["opt_cols"]

    state = {a: {"exceedances": 0, "null": [], "done": False} for a in arms}
    rng = np.random.default_rng(SEED + 1000 * ofold + r)
    executed = 0
    for m in range(M_MAX):
        if all(s["done"] for s in state.values()) and not full:
            break
        dfp = permute_optional(dfb, opt_cols, blocks, rng)
        stat = search(dfx, dfp, tev, disc, conf, params, ticker, ctx["cands"],
                      ctx["fam_of"], ctx["simplicity"], base, base_conf)
        executed = m + 1
        if stat is None:
            continue
        for a, s in state.items():
            if s["done"] and not full:
                continue
            v = stat.get(a)
            if v is None:
                continue
            s["null"].append(round(float(v), 8))
            if v >= arms[a]["real_statistic"]:
                s["exceedances"] += 1
            if s["exceedances"] >= FUTILITY_B:
                s["done"] = True
                s.setdefault("stopped_at", executed)

    out = {"ticker": ticker, "outer_fold": ofold, "confirmation_fold": r,
           "block_length_bars": L, "block_length_clamped": bool(clamped),
           "n_blocks": len(blocks), "n_rows": len(rows), "arms": {}}
    for a, s in state.items():
        n = s.get("stopped_at", executed) if not full else executed
        out["arms"][a] = dict(arms[a], **verdict(s, n),
                              null_max=max(s["null"]) if s["null"] else None,
                              null_median=float(np.median(s["null"])) if s["null"] else None,
                              null_statistics=s["null"])
        if full and s.get("stopped_at"):
            # Verdict agreement is not the interesting quantity: b never decreases, so once it
            # reaches 5 it is still >= 5 at fifty and the two verdicts agree by construction. The
            # futility bound is deterministic, not statistical. What IS worth checking on real data
            # is that the number reported at the stop genuinely BOUNDS the one a full run produces.
            k = s["stopped_at"]
            b_early = sum(1 for v in s["null"][:k] if v >= arms[a]["real_statistic"])
            lower = (1 + b_early) / (M_MAX + 1)
            actual = (1 + b) / (M_MAX + 1)
            out["arms"][a]["early_stop_check"] = {
                "stopped_at": k, "exceedances_at_stop": b_early, "exceedances_at_50": b,
                "reported_lower_bound": round(lower, 6), "p_mc_at_50": round(actual, 6),
                "bound_holds": bool(lower <= actual),
                "_note": "the bound must not exceed the full-budget p-value; equality means no "
                         "further exceedance appeared after the stop"}
    return out


def table(job):
    """All the scoped rotations of one ticker, sharing one derived event table."""
    ticker, scope, full, block_mult, run_id = job
    import feature_search as FS
    import golden
    import nested_validation as NV
    import pipeline as P

    t0 = time.time()
    reg = json.loads(REGISTER.read_text(encoding="utf-8"))["tables"][ticker]
    scratch = runtime_init.scratch_dir(run_id, ticker)
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, scratch / f"{ticker}_1h.parquet")
    cands = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]

    core = set(FS._names(ticker, []))
    opt_cols = [c for c in FS._names(ticker, cands) if c not in core]
    fam_of = golden.load_families(FS.FAMILIES_PATH, cands)
    simplicity = {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                  for ns, rr in P.FEATURE_REGISTRIES.items() if ns != "1h"
                  for f in rr["features"] if bool(f.get("implemented", True))}

    params, inner = {}, {}
    for f in reg["folds"]:
        if f.get("stage") == "no_viable_model":
            continue
        bi = dict(bnds, train_end_idx=int(f["inner_train_end_idx"]))
        params[f["outer_fold"]] = f["frozen_params"]
        inner[f["outer_fold"]] = P.purged_wf_folds(t0s, bi["train_start_idx"], bi["train_end_idx"])

    assert all(max(t0s[j] for j in va) < bnds["oos_start_idx"]
               for fs in inner.values() for _, va in fs if va), \
        "max_null: an inner val fold reaches OOS (purge invariant violated)"

    ctx = {"dfx": dfx, "dfb": dfb, "tev": tev, "t0s": t0s, "params": params, "inner": inner,
           "cands": cands, "fam_of": fam_of, "simplicity": simplicity, "opt_cols": opt_cols,
           "L": L_BLOCK * block_mult}
    rots = [run_rotation(ctx, k, arms, full) for k, arms in sorted(scope.items())]
    return {"ticker": ticker, "seconds": round(time.time() - t0, 1),
            "n_optional_columns": len(opt_cols), "rotations": rots}


def _line(r, total, done):
    v = [a for rot in r["rotations"] for a in rot["arms"].values()]
    p = sum(1 for a in v if a["verdict"] == "passed")
    e = sum(1 for a in v if a["verdict"] == "rejected_early")
    perms = sum(a["permutations_executed"] for a in v) // max(len(v), 1)
    print(f"  [{done}/{total}] {r['ticker']:<6} rotacje={len(r['rotations'])}  "
          f"przeszło={p}  odrzucone wcześnie={e}  śr. permutacji={perms}  ({r['seconds']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--full", action="store_true",
                    help="run all 50 permutations even after futility, to validate early stopping")
    ap.add_argument("--block-mult", type=int, default=1, help="1 for L, 2 for the 2L sensitivity")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    tables = json.loads(CROSSFIT.read_text(encoding="utf-8"))["tables"]
    scope = scope_from_crossfit(tables)
    tickers = args.tickers or sorted({k[0] for k in scope})
    per = {t: {k: v for k, v in scope.items() if k[0] == t} for t in tickers}
    n_rot = sum(len(v) for v in per.values())
    run_id = f"maxnull_L{args.block_mult}" + ("_full" if args.full else "")

    print(f"rung 5 — max-null, blokowe przesunięcie macierzy optional features\n"
          f"  {n_rot} rotacji, {len(tickers)} tabel, do {M_MAX} permutacji na rotację\n"
          f"  blok = {L_BLOCK * args.block_mult} barów (L_block = max(H, L_dependency), "
          f"mnożnik {args.block_mult}), pass gdy b <= {PASS_B}, futility przy b = {FUTILITY_B}\n"
          f"  ramiona dzielą permutacje, ale każde ma własny licznik i własny werdykt"
          + ("\n  --full: pełne 50 mimo futility (walidacja early stopping)" if args.full else "")
          + "\n")

    jobs = [(t, per[t], args.full, args.block_mult, run_id) for t in tickers]
    results = {}
    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for r in ex.map(table, jobs):
                results[r["ticker"]] = r
                _line(r, len(tickers), len(results))
    else:
        for i, j in enumerate(jobs, 1):
            r = table(j)
            results[r["ticker"]] = r
            _line(r, len(tickers), i)

    Path(args.out).write_text(json.dumps(
        {"contract": {"permutations_max": M_MAX, "pass_b": PASS_B, "futility_b": FUTILITY_B,
                      "block_length_bars": L_BLOCK * args.block_mult,
                      "block_mult": args.block_mult, "mode": MODE, "seed": SEED,
                      "full_run": args.full,
                      "scope": "rotations supporting a provisional acceptance",
                      "controls": _C["controls"], "does_not_control": _C["does_not_control"]},
         "tables": results}, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
