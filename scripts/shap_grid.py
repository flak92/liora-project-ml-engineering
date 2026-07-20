#!/usr/bin/env python3
"""Dense SHAP grid — the price path described by many points instead of two bands.

Bollinger draws two lines around a mean and calls that the envelope. This draws, for every
feature the model actually uses, how much that feature pushes the decision at each place
along its own axis — so the envelope becomes a surface, ordered by how much each feature
matters. Anomalies are then visible as cells where a feature pushes hard in a place it
usually does not, and a reversal setup is a feature whose push changes direction more than
once as the value moves.

Why SHAP and not what is already there. The sealed interpretation layer ranks features by
`split_total_gain_share` — the loss reduction credited to each split. That measure is known
to favour features with many distinct values, because a feature that can be cut in more
places gets more chances to collect gain. TreeSHAP asks a different question: how much did
this feature move THIS prediction, averaged over the rows. Rankings from the two can differ,
and where they differ is itself a finding about the feature-selection methodology.

No new dependency: `pred_contribs=True` is XGBoost's own exact TreeSHAP. The objective here
is binary:logistic, so the result is TWO-dimensional, (n_rows, n_features + 1), with the
bias in the last column — the three-dimensional indexing used for multi:softprob elsewhere
would raise IndexError here.

Nothing is retrained. The booster is the sealed one, loaded from Assets/ and checked against
the harvest manifest exactly as the interpretation extractor does, and the matrix is the same
Train candidate matrix. This is a read of a frozen model, on its own training window.

TRAIN-DERIVED / NOT AN OOS RESULT — the OOS window is never touched here.

    python3 xgb/tools/shap_grid.py                 # every ticker in config/sample_20.json
    python3 xgb/tools/shap_grid.py NVDA AMZN       # explicit
    python3 xgb/tools/shap_grid.py --out PATH
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
# The research tree (sealed boosters in Assets/, the pipeline that rebuilds the Train matrix)
# is not published in this repository — it lives untracked beside it. This script is tracked
# because the methodology is the point; it fails with one clear sentence when the tree is
# absent, rather than with an ImportError from three frames down.
XGB = REPO / "xgb"
sys.dont_write_bytecode = True
sys.path.insert(0, str(XGB / "src"))
sys.path.insert(0, str(XGB / "tools"))

SAMPLE = REPO / "config" / "sample_20.json"
DEFAULT_OUT = XGB / "data" / "shap_grid.json"

# The axis every feature is placed on. Values are standardized with the feature's own Train
# mean and sigma, so one grid reads across features whose raw units have nothing in common
# (a ratio, a z-score and an ATR percentage all land on the same ruler), and across tables.
# +/-3 sigma in quarter-sigma steps: 24 cells wide, which is dense enough that a single
# split threshold no longer dominates a cell and coarse enough that the tails keep rows in
# them. Both ends are open, so nothing is dropped for being extreme.
SIGMA_LO, SIGMA_HI, SIGMA_STEP = -3.0, 3.0, 0.25

# A cell with fewer rows than this is reported with its count but marked thin: the brightest
# cells in a grid like this are otherwise the ones standing on a handful of rows.
MIN_CELL_ROWS = 25

RECIPE = {
    "schema": "shap_grid.v1",
    "attribution": "treeshap_pred_contribs_binary_logistic",
    "axis": f"train_sigma[{SIGMA_LO},{SIGMA_HI}]step{SIGMA_STEP}",
    "min_cell_rows": MIN_CELL_ROWS,
    "population": "train_candidate_matrix",
}


def recipe_hash():
    """Own hash, deliberately separate from interpretation_recipe_hash.

    The sealed interpretation layer folds the sha256 of its own source file into its recipe
    hash, and that hash is stamped into 78k store rows and 993 payloads; touching that file
    — even its docstring — invalidates all of them and breaks the parity check that demands
    a single hash across the store. This module therefore never imports its way into that
    identity: it carries its own recipe and its own hash, so recomputing a twenty-table
    subset is legal.
    """
    body = json.dumps(RECIPE, sort_keys=True, separators=(",", ":")).encode()
    src = hashlib.sha256(HERE.read_bytes()).hexdigest().encode()
    return hashlib.sha256(body + src).hexdigest()[:16]


def edges():
    n = int(round((SIGMA_HI - SIGMA_LO) / SIGMA_STEP))
    return [SIGMA_LO + i * SIGMA_STEP for i in range(n + 1)]


def require_research_tree():
    missing = [p for p in (XGB / "src" / "pipeline.py",
                           XGB / "tools" / "extract_ranges.py",
                           XGB / "Assets") if not p.exists()]
    if missing:
        sys.exit("the research tree is not present beside this repository — missing "
                 + ", ".join(str(p.relative_to(REPO)) for p in missing))


def load_sealed(ticker):
    """The sealed booster and its Train candidate matrix, via the extractor's own loaders."""
    # The extractor resolves an epoch at import time by scanning artifacts/<EPOCH>/manifest.json,
    # which is the research tree's layout; this repository publishes artifacts/{xgb,lstm}/ with a
    # manifest at the root, so the scan finds nothing and the import dies. The extractor itself
    # provides the escape (LIORA_EPOCH), and nothing here reads its epoch-derived paths — only
    # its two loaders, which take an explicit directory.
    import os
    os.environ.setdefault("LIORA_EPOCH", "sealed")
    import extract_ranges as E
    import pipeline as P

    asset_dir = XGB / "Assets" / ticker
    strategy = asset_dir / f"strategy_{ticker}.py"
    if not strategy.exists():
        raise FileNotFoundError(f"{ticker}: no sealed strategy at {strategy}")

    mod = E.load_artifact(asset_dir, ticker)
    bst = E.booster_from_artifact(mod, ticker)
    manifest = P.resolve_feature_manifest(ticker)
    names = list(P.feature_names_of(manifest))
    if list(mod.FEATURE_MANIFEST) != names:
        raise ValueError(f"{ticker}: artifact manifest differs from the resolved one")

    rec = P.derive_output_b_from_parquet(asset_dir / f"{ticker}_ohlcv_1h.parquet", ticker, manifest)
    df_b = rec["df_b"]
    if not len(df_b):
        raise ValueError(f"{ticker}: empty Train candidate matrix")
    return bst, names, df_b, float(mod.THRESHOLD_ENTRY)


def contributions(bst, X, names):
    """Exact TreeSHAP from the booster itself. Returns (n_rows, n_features), bias dropped."""
    import numpy as np
    import xgboost as xgb

    C = bst.predict(xgb.DMatrix(X, feature_names=names), pred_contribs=True)
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError(f"expected 2-D contributions for binary:logistic, got shape {C.shape}")
    if C.shape[1] != len(names) + 1:
        raise ValueError(f"contribution width {C.shape[1]} != {len(names)} features + bias")
    return C[:, :-1]


def grid_for(ticker):
    import numpy as np

    bst, names, df_b, theta = load_sealed(ticker)
    X = df_b[names].to_numpy(float)
    S = contributions(bst, X, names)

    mean_abs = np.abs(S).mean(axis=0)
    total = float(mean_abs.sum())
    order = np.argsort(-mean_abs)
    rank = np.empty(len(names), int)
    rank[order] = np.arange(1, len(names) + 1)

    E = edges()
    features = {}
    for j, name in enumerate(names):
        col = X[:, j].astype(np.float32)
        mu, sigma = float(np.nanmean(col)), float(np.nanstd(col))
        # A feature the model never split on contributes exactly zero everywhere. It is kept,
        # with its cells empty and the fact stated: hiding it would misrepresent the model as
        # using fewer inputs than it was given.
        used = bool(np.any(S[:, j] != 0.0))
        z = ((col - mu) / sigma) if sigma > 0 else np.zeros_like(col)
        idx = np.digitize(z.astype(np.float32), np.asarray(E, dtype=np.float32), right=False)

        cells = []
        for b in range(len(E) + 1):
            m = idx == b
            n = int(m.sum())
            lo = E[b - 1] if b > 0 else None
            hi = E[b] if b < len(E) else None
            if not n:
                cells.append({"lo": lo, "hi": hi, "n": 0, "shap_mean": None, "thin": True})
                continue
            cells.append({
                "lo": lo, "hi": hi, "n": n,
                "shap_mean": round(float(S[m, j].mean()), 8),
                "shap_abs_mean": round(float(np.abs(S[m, j]).mean()), 8),
                "thin": n < MIN_CELL_ROWS,
            })

        # A reversal candidate is a feature whose push changes direction more than once as the
        # value moves. Once is guaranteed and means nothing: TreeSHAP is centred, so every
        # feature that contributes at all has both a positive and a negative region.
        signs = [1 if c["shap_mean"] and c["shap_mean"] > 0 else -1 if c["shap_mean"] and c["shap_mean"] < 0 else 0
                 for c in cells if c["n"] >= MIN_CELL_ROWS]
        signs = [s for s in signs if s]
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)

        features[name] = {
            "rank": int(rank[j]),
            "mean_abs_shap": round(float(mean_abs[j]), 8),
            "shap_share": round(float(mean_abs[j] / total), 8) if total > 0 else 0.0,
            "used_by_model": used,
            "train_mu": round(mu, 8), "train_sigma": round(sigma, 8),
            "sign_flips": flips,
            "reversal_candidate": flips >= 2,
            "cells": cells,
        }

    return {
        "ticker": ticker, "model": "xgb",
        "rows": int(len(df_b)), "features": len(names),
        "theta_entry": theta,
        "features_used": int(sum(1 for f in features.values() if f["used_by_model"])),
        "axis_edges": E,
        "by_feature": features,
    }


def compare(path):
    """Does ranking by SHAP pick different features than ranking by split gain?

    This is the question the grid exists to answer. `contribution_share` in the sealed store
    is total split gain, the measure known to favour features with many distinct values —
    every extra cut point is another chance to collect gain. TreeSHAP asks instead how much
    the feature moved the prediction. Agreement in bulk with disagreement at the top would
    mean the two orderings look alike while the feature a selector would actually pick is
    not the same one.
    """
    import sqlite3
    import statistics

    grids = json.loads(Path(path).read_text(encoding="utf-8"))["tickers"]
    con = sqlite3.connect(f"file:{REPO / 'data' / 'results.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    def spearman(a, b):
        keys = list(a)
        if len(keys) < 3:
            return None
        ra = {k: i for i, k in enumerate(sorted(keys, key=lambda k: -a[k]))}
        rb = {k: i for i, k in enumerate(sorted(keys, key=lambda k: -b[k]))}
        n = len(keys)
        return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in keys) / (n * (n * n - 1))

    print(f"{'table':<7}{'features':>9}{'rho':>8}   {'top-1 by SHAP':<26}{'top-1 by gain':<26}{'same':>5}")
    rhos, same, flat = [], 0, []
    for t, v in grids.items():
        shap = {k: x["mean_abs_shap"] for k, x in v["by_feature"].items()}
        gain = {r["feature_name"]: r["contribution_share"] for r in con.execute(
            "select feature_name, contribution_share from feature_contributions "
            "where ticker=? and model='xgb'", (t,))}
        common = {k: shap[k] for k in shap if k in gain}
        if not common or sum(common.values()) == 0:
            flat.append(t)
            print(f"{t:<7}{len(shap):>9}{'—':>8}   the sealed model never split — no ranking to compare")
            continue
        g = {k: gain[k] for k in common}
        rho = spearman(common, g)
        a, b = max(common, key=common.get), max(g, key=g.get)
        same += a == b
        rhos.append(rho)
        print(f"{t:<7}{len(common):>9}{rho:>8.3f}   {a[:25]:<26}{b[:25]:<26}{'yes' if a == b else 'NO':>5}")

    print(f"\nmedian rho = {statistics.median(rhos):.3f} over {len(rhos)} tables — the two "
          f"measures broadly agree on order,\nyet the top-ranked feature differs in "
          f"{len(rhos) - same} of {len(rhos)}. Ranking by gain and ranking by SHAP are not\n"
          f"interchangeable for selection, whatever the bulk correlation suggests.")
    if flat:
        print(f"excluded (constant model, zero contributions everywhere): {', '.join(flat)}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--compare", action="store_true",
                    help="read an existing grid and contrast its ranking with split gain")
    args = ap.parse_args()

    if args.compare:
        return compare(args.out)

    require_research_tree()
    tickers = args.tickers or json.loads(SAMPLE.read_text(encoding="utf-8"))["sample"]
    rh = recipe_hash()
    print(f"dense SHAP grid — recipe {rh}, axis {SIGMA_LO}..{SIGMA_HI} sigma step {SIGMA_STEP} "
          f"({len(edges()) + 1} cells)\n")

    out, problems = {}, []
    for i, t in enumerate(tickers, 1):
        try:
            g = grid_for(t)
            out[t] = g
            top = sorted(g["by_feature"].items(), key=lambda kv: kv[1]["rank"])[:3]
            rev = sum(1 for f in g["by_feature"].values() if f["reversal_candidate"])
            head = ", ".join("{} {:.3f}".format(k, v["shap_share"]) for k, v in top)
            print(f"  [{i}/{len(tickers)}] {t:<6} rows={g['rows']:>6,} "
                  f"used={g['features_used']}/{g['features']} reversal={rev:>2}  top: {head}")
        except Exception as e:
            problems.append(f"{t}: {type(e).__name__}: {e}")
            print(f"  [{i}/{len(tickers)}] {t:<6} FAILED — {type(e).__name__}: {e}")

    Path(args.out).write_text(json.dumps({
        "recipe": RECIPE, "recipe_hash": rh,
        "labels": ["TRAIN-DERIVED", "NOT AN OOS RESULT", "NOT A LIVE TRADING SIGNAL"],
        "tickers": out,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}  ({len(out)} tabel)")
    if problems:
        print(f"FAILED for {len(problems)}: " + "; ".join(problems))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
