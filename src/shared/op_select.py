#!/usr/bin/env python3
"""Shared Train-OOF operating-point selection — ONE implementation for both pipelines
(xgb L7/L8 + feature search, lstm D7/D8 + feature search), so the theta spectrum, the
trade floor and the robustness constraints can never drift between scorers (#13-14).

The contract is ACCUMULATE-THEN-SELECT: every fold's out-of-fold predictions are replayed
through the deployed engine at each grid point, per-fold log-growth/trade-counts are
accumulated ACROSS folds, and ONE shared point is chosen for the whole Train window.
A per-fold best-theta ("fold oracle") is forbidden: no deployable strategy can switch
theta between folds, so scoring each fold at its own best point upward-biases every
score it touches (HPO trials, feature gains) — exactly the defect this module removes.

Selection order (all deterministic, pure):
  1. floor: points with >= min_oof_trades total OOF trades (min-2-trades mandate);
  2. fold spread: among floor-viable points, prefer those with trades in >=
     min_active_folds folds AND no single fold holding more than max_fold_trade_share
     of them (a one-fold burst is not evidence); relaxing this is flagged, never silent;
  3. plateau: within one standard error (sample std, ddof=1) of the pool's best
     log-growth, prefer an INTERIOR theta over a spectrum edge (#30-33);
  4. conservative ties: higher theta, then smaller lambda (None==all-in sorts first),
     then direction 'both' before 'long_only'.
If nothing clears the floor the most-trading point is scored and flagged
trade_floor_met=False — the caller must NOT promote such a result (#9).

Grid rows: {"theta": float, "lambda": float|None, "direction": str (optional),
            "fold_growth": [float per fold], "fold_trades": [int per fold]}.
"""
import math


def _sample_std(xs):
    """Sample standard deviation (ddof=1) — the one-SE estimator (#34)."""
    k = len(xs)
    if k < 2:
        return 0.0
    m = sum(xs) / k
    return (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5


def _enrich(p):
    g = [float(x) for x in p["fold_growth"]]
    n = [int(x) for x in p["fold_trades"]]
    tot = int(sum(n))
    return {**p, "log_growth": float(sum(g)), "trades": tot,
            "active_folds": int(sum(1 for x in n if x > 0)),
            "max_fold_share": (max(n) / tot) if tot > 0 else 0.0}


def _tie_key(p):
    """Conservative deterministic order: higher theta, smaller lambda, 'both' first."""
    lam = 0.0 if p.get("lambda") is None else float(p["lambda"])
    d = 0 if p.get("direction", "both") == "both" else 1
    return (-p["theta"], lam, d)


def select_operating_point(grid, *, min_oof_trades=0, min_active_folds=0,
                           max_fold_trade_share=1.0, theta_spectrum=None):
    """Choose ONE shared operating point from an accumulate-then-select grid.
    Returns the chosen (enriched) point plus audit flags:
      trade_floor_met   — some point cleared min_oof_trades (chosen from those)
      fold_spread_relaxed — the spread constraints had to be dropped to pick a point
      theta_boundary    — the chosen theta sits on the spectrum edge (audit flag, #33)
      se, plateau_size  — the one-SE plateau the choice was made from
    """
    if not grid:
        raise ValueError("select_operating_point: empty grid")
    pts = [_enrich(p) for p in grid]
    n_folds = len(pts[0]["fold_trades"])
    viable = [p for p in pts if p["trades"] >= int(min_oof_trades)]
    floor_met = bool(viable)
    spread_relaxed = False
    if viable:
        pool = viable
        if n_folds >= 2 and (min_active_folds > 1 or max_fold_trade_share < 1.0):
            spread_ok = [p for p in pool
                         if p["active_folds"] >= min(int(min_active_folds), n_folds)
                         and p["max_fold_share"] <= float(max_fold_trade_share) + 1e-12]
            if spread_ok:
                pool = spread_ok
            else:
                spread_relaxed = True
    else:
        # nothing clears the floor -> the most-trading point, honestly flagged (never promoted)
        pool = [max(pts, key=lambda p: (p["trades"], p["log_growth"], _neg_tie(p)))]
    gmax = max(p["log_growth"] for p in pool)
    best = min((p for p in pool if p["log_growth"] >= gmax - 1e-12), key=_tie_key)
    se = _sample_std(best["fold_growth"]) / math.sqrt(n_folds) if n_folds > 1 else 0.0
    plateau = [p for p in pool if p["log_growth"] >= gmax - se - 1e-12]
    spectrum = sorted(set(float(t) for t in (theta_spectrum
                                             if theta_spectrum is not None
                                             else [p["theta"] for p in pts])))
    edges = {spectrum[0], spectrum[-1]} if spectrum else set()
    interior = [p for p in plateau if p["theta"] not in edges]
    chosen = min(interior or plateau, key=_tie_key)
    return {**chosen,
            "trade_floor_met": floor_met,
            "fold_spread_relaxed": spread_relaxed,
            "theta_boundary": chosen["theta"] in edges,
            "se": float(se), "plateau_size": len(plateau)}


def _neg_tie(p):
    """max()-compatible mirror of _tie_key (higher is preferred)."""
    lam = 0.0 if p.get("lambda") is None else float(p["lambda"])
    d = 0 if p.get("direction", "both") == "both" else 1
    return (p["theta"], -lam, -d)
