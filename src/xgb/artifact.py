#!/usr/bin/env python3
"""Writers used by the per-asset notebook.

They emit the standalone strategy artifact, the OOS README, the feature-table
helper, and the SQLite row consumed by the dashboard. Compute stays in
pipeline.py; this file only writes the seven-file deliverable and the dashboard
side store.
"""
from datetime import date
from pathlib import Path

import pipeline as P


def _write_text(path, text):
    P.atomic_write(path, lambda p: Path(p).write_text(text, encoding="utf-8"))


def write_strategy(path, m):
    """File #6: strategy_<TICKER>.py — a standalone artifact with the XGBoost model embedded as base64 (MODEL_B64).
    `m` is pipeline.strategy_meta(...) (+ m["ACCEPTANCE"]). Reloads + selfchecks against the golden vectors on run."""
    body = f'''"""Standalone strategy artifact for {m["ticker"]} (L8). Imports with no training-data access.
LABEL_CONTRACT = {m["LABEL_CONTRACT"]}; the XGBoost meta-label model is embedded as base64 (MODEL_B64).
"""
import base64

nan = float("nan")

TICKER = {m["ticker"]!r}
LABEL_CONTRACT = {m["LABEL_CONTRACT"]!r}
FEATURE_MANIFEST = {m["FEATURE_MANIFEST"]!r}
FEATURE_IDS = {m["FEATURE_IDS"]!r}
FEATURE_NAMESPACES = {m["FEATURE_NAMESPACES"]!r}
THRESHOLD_ENTRY = {m["THRESHOLD_ENTRY"]!r}
CALIBRATION = {m["CALIBRATION"]!r}
MODEL_HASH = {m["MODEL_HASH"]!r}
TRAIN_WINDOW = {m["TRAIN_WINDOW"]!r}
EXECUTION_CONTRACT = {m["EXECUTION_CONTRACT"]!r}
ACCEPTANCE = {m["ACCEPTANCE"]!r}
BEST_PARAMS = {m["best_params"]!r}
_GOLDEN_VECTORS = {m["golden_vectors"]!r}
_GOLDEN_PRED = {m["golden_pred"]!r}
MODEL_B64 = "{m["MODEL_B64"]}"


def _load():
    import xgboost as xgb, tempfile, os, hashlib
    raw = base64.b64decode(MODEL_B64)
    assert hashlib.sha256(raw).hexdigest() == MODEL_HASH, "MODEL_HASH mismatch"
    bst = xgb.Booster()
    with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
        f.write(raw); tmp = f.name
    try:
        bst.load_model(tmp)
    finally:
        os.unlink(tmp)
    return bst


def predict_proba(X):
    import xgboost as xgb, numpy as np
    return _load().predict(xgb.DMatrix(np.asarray(X, float).reshape(-1, len(FEATURE_MANIFEST)),
                                       feature_names=FEATURE_MANIFEST))


def selfcheck():
    if not _GOLDEN_VECTORS:
        return True
    import numpy as np
    assert np.allclose(predict_proba(_GOLDEN_VECTORS), _GOLDEN_PRED, atol=1e-6), "selfcheck divergence"
    return True


if __name__ == "__main__":
    print("selfcheck:", selfcheck())
'''
    _write_text(path, body)


def features_table_lines(manifest):
    """The effective feature manifest as a Markdown table in model order."""
    formulas = {}
    for ns, reg in P.FEATURE_REGISTRIES.items():
        for f in reg["features"]:
            formulas[int(f["id"])] = f.get("formula", "")
    rows, pos = [], 0
    for blk in manifest["per_namespace"]:
        ns = blk["namespace"]
        for fid, name in zip(blk["feature_ids"], blk["feature_names"]):
            pos += 1
            rows.append(f"| {pos} | {fid} | `{name}` | {ns} | {formulas.get(int(fid), '')} |")
    return [f"## Features used to train the model ({manifest['effective_feature_count']} X columns)", "",
            f"- active namespaces: {', '.join(manifest['active_namespaces'])}",
            f"- model column order: namespace order, then ascending feature ID",
            f"- selection source: `{manifest['feature_selection_source']}`", "",
            "| # | ID | Feature | Namespace | Formula |",
            "|--|--|--|--|--|"] + rows


def write_readme(path, s, ledger, manifest):
    """File #7: <TICKER>_README.md — the OOS capital path + feature table + Triple Barrier trade ledger. `s` is the OOS
    run_engine summary (+ s["ticker"]); `ledger` is the trade list; `manifest` is the resolved feature manifest."""
    sp = P.PIPELINE_PARAMETERS["splits"]                                   # the OOS window return_pct is earned over
    oos_days = (date.fromisoformat(sp["oos_end"]) - date.fromisoformat(sp["oos_start"])).days
    roi_per_365 = s["return_pct"] * 365.0 / oos_days if oos_days else 0.0  # return_pct annualized to 365 days
    cap_mode = s.get("capital_mode", P.PIPELINE_PARAMETERS["CAPITAL_MODE"])
    lam = s.get("kelly_fraction")
    L = [f"# {s['ticker']} — OOS report (current cycle)", "",
         "- EXECUTION_SCOPE: SIMPLE_FEATURES_TRIPLE_BARRIER",
         "- RESULT_INTERPRETATION: historical behaviour under this fill / cost / Triple Barrier / position-sizing / "
         "compounding model; not broker-specific execution proof."]
    if s.get("hodl_fallback"):
        L.append("- OOS_MODE: the model produced 0 trades in the OOS window -> HODL fallback: one long "
                 "buy-and-hold trade (buy the first OOS bar's open, sell the last OOS bar's close, same "
                 "fill/cost model). The ledger below is that benchmark trade, NOT a model trade.")
    L += ["",
         f"## Capital path ({cap_mode})", "",
         f"- ROI/365: {roi_per_365:.2f}%",
         f"- data range: {sp['oos_start']} → {sp['oos_end']} ({oos_days} days)",
         f"- start_capital: {s['start_capital']}", f"- end_capital: {s['end_capital']:.2f}",
         f"- return_pct: {s['return_pct']:.2f}%", f"- profit_factor: {s['profit_factor']}",
         f"- max_drawdown_pct: {s['max_drawdown_pct']:.2f}%", f"- win_rate_pct: {s['win_rate_pct']:.2f}%",
         f"- trades: {s['trades']} (wins {s['wins']} / losses {s['losses']})",
         f"- time_in_market_pct: {s['time_in_market_pct']}",
         f"- uncovered_loss_total_usd: {s['uncovered_loss_total_usd']:.2f} (max {s['max_uncovered_loss_usd']:.2f})",
         f"- capital_depleted: {s['capital_depleted']}"]
    if s.get("theta_entry") is not None:
        L.append(f"- operating point (Train-OOF calibrated): entry θ = {s['theta_entry']:.2f}"
                 + ("" if s.get("trade_floor_met", True) else
                    " — TRAIN_OOF_FLOOR_NOT_MET (no grid point cleared the trade floor; not promoted)"))
    if s.get("result_mode"):
        L.append(f"- result_mode: {s['result_mode']}")
    if lam is not None:
        L.append(f"- kelly_fraction (λ): {lam:.4f} — per-trade f = clip(λ·(2p−1), 0, "
                 f"{P.PIPELINE_PARAMETERS['KELLY_CAP']}); b=1 symmetric Triple Barrier, Train-OOF calibrated")
    else:
        L.append("- sizing: all-in compounding — every trade reinvests the FULL running capital "
                 "(no position sizing; the same capital game as buy-and-hold)")
    L.append("")
    L += features_table_lines(manifest)
    L += ["",
          "## Triple Barrier trade ledger (ORDER BY trade_id ASC)", "",
          "| # | dir | entry_fill_timestamp | entry | target | stop | exit_fill_timestamp | exit | reason | acct_net | cap_after |",
          "|--|--|--|--|--|--|--|--|--|--|--|"]
    import math
    fmt4 = lambda v: "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"
    for l in ledger[:50]:
        L.append(f"| {l['trade_id']} | {l['direction']} | {l['entry_fill_timestamp']} | {l['entry_fill']:.4f} "
                 f"| {fmt4(l['target_level'])} | {fmt4(l['stop_level'])} | {l['exit_fill_timestamp']} | {l['exit_fill']:.4f} "
                 f"| {l['market_exit_reason']} | {l['account_net_pnl_usd']:.2f} | {l['capital_after']:.2f} |")
    if len(ledger) > 50:
        L.append(f"| … | | {len(ledger)-50} more | | | | | | | | |")
    _write_text(path, "\n".join(L) + "\n")


def write_oos_metrics(db_path, row):
    """OOS results store: UPSERT one asset's verdict into oos_metrics.db — the per-asset table the Dashboard
    reads. Side-effect OUTSIDE the 7-file deliverable (lives in Structure/, keyed by ticker). stdlib sqlite3; only OOS
    result columns — no lineage / contract / source-QC."""
    import sqlite3
    cols = ["ticker", "start_capital", "end_capital", "net_pnl_usd", "return_pct", "profit_factor",
            "max_drawdown_pct", "win_rate_pct", "trades", "wins", "losses", "time_in_market_pct",
            "capital_depleted", "cv_auc_pr", "cv_folds", "oos_window",
            # v4 semantics (#2, #8, #36-41): trades == MODEL trades; the benchmark trade and the
            # result taxonomy are explicit columns; the Train-OOF calibration record travels
            # with the row so the presentation layer never guesses from trade counts.
            "model_trades", "benchmark_trades", "result_mode", "theta_entry",
            "oof_trades", "trade_floor_met", "oof_log_growth", "recipe_hash"]
    con = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        con.execute("PRAGMA journal_mode=WAL")      # concurrent run_asset finishes (parallel apply
        con.execute("PRAGMA busy_timeout=10000")    # phase) UPSERT disjoint ticker rows -> brief
        con.execute(                                 # serialized commits, not 'database is locked'
            "create table if not exists oos_metrics ("
            "ticker text primary key, start_capital real, end_capital real, net_pnl_usd real, return_pct real, "
            "profit_factor real, max_drawdown_pct real, win_rate_pct real, trades integer, wins integer, "
            "losses integer, time_in_market_pct real, capital_depleted integer, cv_auc_pr real, cv_folds integer, "
            "oos_window text, model_trades integer, benchmark_trades integer, result_mode text, "
            "theta_entry real, oof_trades integer, trade_floor_met integer, oof_log_growth real, "
            "recipe_hash text)")
        # idempotent v3->v4 migration: a pre-existing 16-column store gains the new columns
        # in place (reset clears ROWS, never the schema; old rows read back as NULLs)
        have = {r[1] for r in con.execute("pragma table_info(oos_metrics)")}
        for c, typ in (("model_trades", "integer"), ("benchmark_trades", "integer"),
                       ("result_mode", "text"), ("theta_entry", "real"), ("oof_trades", "integer"),
                       ("trade_floor_met", "integer"), ("oof_log_growth", "real"),
                       ("recipe_hash", "text")):
            if c not in have:
                con.execute(f"alter table oos_metrics add column {c} {typ}")
        vals = [row.get(c) for c in cols]
        for b in ("capital_depleted", "trade_floor_met"):                     # normalize bool -> 0/1
            i = cols.index(b)
            if vals[i] is not None:
                vals[i] = int(bool(vals[i]))
        con.execute("insert or replace into oos_metrics (" + ",".join(cols) + ") values (" +
                    ",".join("?" * len(cols)) + ")", vals)
        con.commit()
    finally:
        con.close()
