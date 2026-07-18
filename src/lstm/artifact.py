#!/usr/bin/env python3
"""Per-asset deliverable writers (3 files) + the oos_metrics side store.

Compute stays in pipeline.py / model.py; this file only serializes:
  1. best_params.json      — HPO winner, CV score, Kelly λ, frozen normalization stats
  2. strategy_<T>.py       — standalone artifact: LSTM state_dict as base64 + MODEL_HASH,
                             frozen NORM_STATS, golden-vector selfcheck (runs with torch+numpy only)
  3. <T>_README.md         — the OOS report (capital path, feature table, trade ledger)
  4. (side-effect) oos_metrics.db UPSERT — schema identical to the parent project, so the
     Basket Simulator app and the dashboard consume it unchanged.
"""
import base64
import io
import json
import os
from datetime import date
from pathlib import Path

import numpy as np
import torch

import pipeline as P


def _write_text(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def strategy_meta(model, ticker, manifest, best_params, refit_epochs, cal, cv_ap, cv_folds,
                  norm_stats, X_raw):
    """Everything the standalone artifact embeds. Golden vectors are RAW (un-normalized)
    windows over the per-asset manifest — the artifact reproduces the full NORM_STATS +
    model path."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    gv = X_raw[:2] if len(X_raw) >= 2 else X_raw
    gp = []
    if len(gv):
        mu, sd = _stats_vec(norm_stats, manifest)
        gn = np.stack([(win - mu) / sd for win in gv]).astype(np.float32)
        from model import predict_proba
        gp = predict_proba(model, torch.from_numpy(gn)).tolist()
    sp = P.CONFIG["splits"]
    return {"ticker": ticker, "MODEL_B64": base64.b64encode(raw).decode("ascii"),
            "MODEL_HASH": P.sha256_bytes(raw),
            "ARCH": {"n_features": len(manifest), "hidden": int(best_params["hidden"]),
                     "dropout": float(best_params["dropout"]),
                     "num_layers": int(best_params.get("num_layers", 1))},
            "SEQ_LEN": P.CONFIG["SEQ_LEN"], "FEATURE_MANIFEST": manifest,
            "NORM_STATS": norm_stats, "THRESHOLD_ENTRY": float(cal["theta_entry"]),
            "DIRECTION_MODE": cal["direction_mode"],
            "CALIBRATION": {k: cal.get(k) for k in
                            ("theta_entry", "kelly_fraction", "direction_mode", "oof_trades",
                             "trade_floor_met", "oof_log_growth")},
            "LABEL_CONTRACT": "TripleBarrier.ATR.daily.v1",
            "best_params": {**best_params, "refit_epochs": refit_epochs},
            "kelly_fraction": None if cal["kelly_fraction"] is None else float(cal["kelly_fraction"]),
            "cv_auc_pr": cv_ap, "cv_folds": cv_folds,
            "TRAIN_WINDOW": f"{sp['train_start']} -> {sp['train_end']}",
            "EXECUTION_CONTRACT": {"entry_fill": "next_session_open",
                                   "exit_fill": "condition: next open; scheduled: session close",
                                   "commission_bps": P.CONFIG["COMMISSION_BPS"],
                                   "slippage_bps": P.CONFIG["SLIPPAGE_BPS"],
                                   "barrier_mode": P.CONFIG["BARRIER_MODE"],
                                   "triple_barrier_tp": f"ATR{P.CONFIG['W_ATR']} * {P.CONFIG['TB_ATR_TP']}",
                                   "triple_barrier_sl": f"ATR{P.CONFIG['W_ATR']} * {P.CONFIG['TB_ATR_SL']}",
                                   "reward_risk_b": float(P.CONFIG["TB_ATR_TP"]) / float(P.CONFIG["TB_ATR_SL"]),
                                   "kelly_cap": P.CONFIG["KELLY_CAP"],
                                   "kelly_basis": "per_trade_fractional_kelly_f=lambda*(p-(1-p)/b)"},
            "golden_vectors": gv.tolist() if len(gv) else [], "golden_pred": gp}


def _stats_vec(norm_stats, names):
    mu = np.array([norm_stats[n]["mean"] for n in names], np.float32)
    sd = np.array([norm_stats[n]["std"] for n in names], np.float32)
    return mu, sd


def write_best_params(path, m):
    keep = {k: m[k] for k in ("ticker", "best_params", "kelly_fraction", "THRESHOLD_ENTRY",
                              "DIRECTION_MODE", "CALIBRATION", "cv_auc_pr", "cv_folds",
                              "SEQ_LEN", "FEATURE_MANIFEST", "NORM_STATS", "TRAIN_WINDOW",
                              "LABEL_CONTRACT", "MODEL_HASH")}
    keep["RANDOM_SEED"] = P.CONFIG["RANDOM_SEED"]
    _write_text(path, json.dumps(keep, indent=2) + "\n")


def write_strategy(path, m):
    body = f'''"""Standalone strategy artifact for {m["ticker"]} (D8). Imports with no training-data access.
LABEL_CONTRACT = {m["LABEL_CONTRACT"]}; the LSTM state_dict is embedded as base64 (MODEL_B64).
predict_proba() consumes RAW feature windows of shape (N, SEQ_LEN, n_features) — it applies the
frozen TRAIN-only NORM_STATS itself, then the reloaded model. `python3 {Path(str(path)).name}` runs the selfcheck.
"""
import base64
import hashlib
import io

TICKER = {m["ticker"]!r}
LABEL_CONTRACT = {m["LABEL_CONTRACT"]!r}
FEATURE_MANIFEST = {m["FEATURE_MANIFEST"]!r}
SEQ_LEN = {m["SEQ_LEN"]!r}
ARCH = {m["ARCH"]!r}
NORM_STATS = {m["NORM_STATS"]!r}
THRESHOLD_ENTRY = {m["THRESHOLD_ENTRY"]!r}
DIRECTION_MODE = {m["DIRECTION_MODE"]!r}
MODEL_HASH = {m["MODEL_HASH"]!r}
TRAIN_WINDOW = {m["TRAIN_WINDOW"]!r}
EXECUTION_CONTRACT = {m["EXECUTION_CONTRACT"]!r}
BEST_PARAMS = {m["best_params"]!r}
KELLY_FRACTION = {m["kelly_fraction"]!r}
_GOLDEN_VECTORS = {m["golden_vectors"]!r}
_GOLDEN_PRED = {m["golden_pred"]!r}
MODEL_B64 = "{m["MODEL_B64"]}"


def _load():
    import torch
    import torch.nn as nn
    raw = base64.b64decode(MODEL_B64)
    assert hashlib.sha256(raw).hexdigest() == MODEL_HASH, "MODEL_HASH mismatch"

    class LSTMClassifier(nn.Module):
        def __init__(self, n_features, hidden, dropout, num_layers=1):
            super().__init__()
            num_layers = int(num_layers)
            self.lstm = nn.LSTM(n_features, hidden, num_layers=num_layers, batch_first=True,
                                dropout=float(dropout) if num_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    model = LSTMClassifier(ARCH["n_features"], ARCH["hidden"], ARCH["dropout"],
                           ARCH.get("num_layers", 1))
    model.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu"))
    model.eval()
    return model


def predict_proba(X_raw):
    """X_raw: (N, SEQ_LEN, n_features) RAW feature windows in FEATURE_MANIFEST order.
    A single (SEQ_LEN, n_features) window is also accepted. Shapes are validated, never
    silently reshaped — a transposed input must fail loudly, not predict garbage."""
    import numpy as np
    import torch
    X = np.asarray(X_raw, np.float32)
    if X.ndim == 2:
        X = X[None]
    assert X.ndim == 3 and X.shape[1:] == (SEQ_LEN, len(FEATURE_MANIFEST)), \\
        f"expected (N, {{SEQ_LEN}}, {{len(FEATURE_MANIFEST)}}) raw windows, got {{X.shape}}"
    mu = np.array([NORM_STATS[n]["mean"] for n in FEATURE_MANIFEST], np.float32)
    sd = np.array([NORM_STATS[n]["std"] for n in FEATURE_MANIFEST], np.float32)
    with torch.no_grad():
        return torch.sigmoid(_load()(torch.from_numpy((X - mu) / sd))).numpy()


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
    core = set(P.CORE_FEATURE_NAMES)
    rows = [f"| {i} | `{n}` | {'core' if n in core else 'selected'} | "
            f"{P.FEATURE_FORMULAS.get(n, 'Claude-proposed DSL')} |" for i, n in enumerate(manifest, 1)]
    n_opt = len(manifest) - len(core & set(manifest))
    return [f"## Features (the LSTM's input channels, {len(manifest)} per session — "
            f"{len(core & set(manifest))} core + {n_opt} selected)", "",
            f"- model input: the window of the last {P.CONFIG['SEQ_LEN']} sessions ending at the",
            "  decision close, z-scored with per-asset TRAIN-only statistics (frozen in the artifact)", "",
            "| # | Feature | Kind | Formula |", "|--|--|--|--|"] + rows


def write_readme(path, s, ledger, manifest):
    sp = P.CONFIG["splits"]
    oos_days = (date.fromisoformat(sp["oos_end"]) - date.fromisoformat(sp["oos_start"])).days
    roi_per_365 = s["return_pct"] * 365.0 / oos_days if oos_days else 0.0
    lam = s.get("kelly_fraction")
    L = [f"# {s['ticker']} — OOS report", "",
         "- EXECUTION_SCOPE: DAILY_LSTM_TRIPLE_BARRIER",
         "- RESULT_INTERPRETATION: historical behaviour under this fill / cost / Triple Barrier / "
         "position-sizing / compounding model; not broker-specific execution proof."]
    if s.get("hodl_fallback"):
        L.append("- OOS_MODE: the model produced 0 trades in the OOS window -> HODL fallback: one long "
                 "buy-and-hold trade (buy the first OOS session's open, sell the last OOS session's close, "
                 "same fill/cost model). The ledger below is that benchmark trade, NOT a model trade.")
    L += ["",
          f"## Capital path ({s.get('capital_mode', P.CONFIG['CAPITAL_MODE'])})", "",
          f"- ROI/365: {roi_per_365:.2f}%",
          f"- OOS window: {sp['oos_start']} -> {sp['oos_end']} ({oos_days} days)",
          f"- start_capital: {s['start_capital']}", f"- end_capital: {s['end_capital']:.2f}",
          f"- return_pct: {s['return_pct']:.2f}%", f"- profit_factor: {s['profit_factor']}",
          f"- max_drawdown_pct: {s['max_drawdown_pct']:.2f}%", f"- win_rate_pct: {s['win_rate_pct']:.2f}%",
          f"- trades: {s['trades']} (wins {s['wins']} / losses {s['losses']})",
          f"- time_in_market_pct: {s['time_in_market_pct']}",
          f"- uncovered_loss_total_usd: {s.get('uncovered_loss_total_usd', 0.0):.2f} "
          f"(max {s.get('max_uncovered_loss_usd', 0.0):.2f})",
          f"- capital_depleted: {s['capital_depleted']}"]
    L.append(f"- operating point (Train-OOF calibrated): entry θ = {s.get('theta_entry', float('nan')):.2f}, "
             f"direction = {s.get('direction_mode', 'both')}"
             + ("" if s.get("trade_floor_met", True) else
                " — TRAIN_OOF_FLOOR_NOT_MET (no grid point cleared the trade floor; not promoted)"))
    if s.get("result_mode"):
        L.append(f"- result_mode: {s['result_mode']}")
    if lam is not None:
        _b = float(P.CONFIG["TB_ATR_TP"]) / float(P.CONFIG["TB_ATR_SL"])
        L.append(f"- kelly_fraction (λ): {lam:.4f} — per-trade f = clip(λ·(p−(1−p)/b), 0, "
                 f"{P.CONFIG['KELLY_CAP']}); reward:risk b = {_b:.2f} "
                 f"(TP {P.CONFIG['TB_ATR_TP']}·ATR / SL {P.CONFIG['TB_ATR_SL']}·ATR)")
    else:
        L.append("- sizing: all-in compounding — every trade reinvests the FULL running capital "
                 "(no position sizing; the same capital game as buy-and-hold)")
    L.append("")
    L += features_table_lines(manifest)
    L += ["", "## Triple Barrier trade ledger (ORDER BY trade_id ASC)", "",
          "| # | dir | entry_date | entry | target | stop | exit_date | exit | reason | acct_net | cap_after |",
          "|--|--|--|--|--|--|--|--|--|--|--|"]
    import math
    fmt4 = lambda v: "—" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.4f}"
    for l in ledger[:50]:
        L.append(f"| {l['trade_id']} | {l['direction']} | {l['entry_fill_date']} | {l['entry_fill']:.4f} "
                 f"| {fmt4(l['target_level'])} | {fmt4(l['stop_level'])} | {l['exit_fill_date']} "
                 f"| {l['exit_fill']:.4f} | {l['market_exit_reason']} | {l['account_net_pnl_usd']:.2f} "
                 f"| {l['capital_after']:.2f} |")
    if len(ledger) > 50:
        L.append(f"| … | | {len(ledger) - 50} more | | | | | | | | |")
    _write_text(path, "\n".join(L) + "\n")


def write_oos_metrics(db_path, row):
    """UPSERT one asset's OOS verdict — schema identical to the parent project, so the
    Basket Simulator and the dashboard read it unchanged."""
    import sqlite3
    cols = ["ticker", "start_capital", "end_capital", "net_pnl_usd", "return_pct", "profit_factor",
            "max_drawdown_pct", "win_rate_pct", "trades", "wins", "losses", "time_in_market_pct",
            "capital_depleted", "cv_auc_pr", "cv_folds", "oos_window",
            # sealed-store semantics (#2, #8, #36-41): trades == MODEL trades; the benchmark trade and the
            # result taxonomy are explicit columns; the Train-OOF calibration record travels
            # with the row so the presentation layer never guesses from trade counts.
            "model_trades", "benchmark_trades", "result_mode", "theta_entry",
            "oof_trades", "trade_floor_met", "oof_log_growth", "recipe_hash"]
    con = sqlite3.connect(str(db_path), timeout=10.0)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=10000")
        con.execute(
            "create table if not exists oos_metrics ("
            "ticker text primary key, start_capital real, end_capital real, net_pnl_usd real, return_pct real, "
            "profit_factor real, max_drawdown_pct real, win_rate_pct real, trades integer, wins integer, "
            "losses integer, time_in_market_pct real, capital_depleted integer, cv_auc_pr real, cv_folds integer, "
            "oos_window text, model_trades integer, benchmark_trades integer, result_mode text, "
            "theta_entry real, oof_trades integer, trade_floor_met integer, oof_log_growth real, "
            "recipe_hash text)")
        # idempotent store migration: a pre-existing 16-column store gains the new columns
        # in place (reset clears ROWS, never the schema; old rows read back as NULLs)
        have = {r[1] for r in con.execute("pragma table_info(oos_metrics)")}
        for c, typ in (("model_trades", "integer"), ("benchmark_trades", "integer"),
                       ("result_mode", "text"), ("theta_entry", "real"), ("oof_trades", "integer"),
                       ("trade_floor_met", "integer"), ("oof_log_growth", "real"),
                       ("recipe_hash", "text")):
            if c not in have:
                con.execute(f"alter table oos_metrics add column {c} {typ}")
        vals = [row.get(c) for c in cols]
        for b in ("capital_depleted", "trade_floor_met"):          # normalize bool -> 0/1
            i = cols.index(b)
            if vals[i] is not None:
                vals[i] = int(bool(vals[i]))
        con.execute("insert or replace into oos_metrics (" + ",".join(cols) + ") values (" +
                    ",".join("?" * len(cols)) + ")", vals)
        con.commit()
    finally:
        con.close()
