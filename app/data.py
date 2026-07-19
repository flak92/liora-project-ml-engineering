"""app/data.py — the ONE data-access module for the presentation console.

The ONLY place in the app that:
- opens data/results.db (sqlite, mode=ro),
- verifies the schema and dataset completeness (fail-closed statuses),
- holds every SQL query the seven pages need,
- caches small aggregates (lru_cache; the db is sealed, so caches never go stale),
- lazy-loads per-asset JSONs ONLY after an asset is selected, strictly via
  asset_results.artifact_path -> manifest.json / parameters.json / metrics.json /
  interpretation.json (no folder scanning, ever).

Pages import this module and nothing else data-related. No page opens SQLite or
touches the filesystem on its own. The data layer is sealed — this module treats
it as immutable. Cached DataFrames are shared across sessions: pages must treat
them as read-only (filtering/sorting returns copies, so plain use is safe).
"""
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "results.db"

EXPECTED_TABLES = {
    "research_run", "asset_results", "asset_features", "feature_search_summary",
    "integrity_checks", "feature_train_stats", "feature_contributions", "xgb_entry_ranges",
    "oos_read_summary",
}
EXPECTED_FREEZE_PREFIX = "public/"   # stable-1, stable-2, … — the label carries the release
ARTIFACT_JSONS = ("manifest.json", "parameters.json", "metrics.json", "interpretation.json")
CONFIG_JSONS = ("xgb.json", "lstm.json")   # the frozen configuration the pipelines read
MODEL_KEY = {"XGBoost": "xgb", "LSTM": "lstm"}   # display name -> store key

# fail-closed statuses (STREAMLIT_DESIGN §4)
OK = "OK"
NOT_FOUND = "NOT FOUND"
SCHEMA_MISMATCH = "SCHEMA MISMATCH"
INTEGRITY_FAILED = "DATA INTEGRITY: FAILED"
PARTIAL = "DATASET STATUS: PARTIAL"


def _connect():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _rows(sql, params=()):
    con = _connect()
    con.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in con.execute(sql, params)]
    finally:
        con.close()


# ---------------------------------------------------------------- health (fail-closed)

@lru_cache(maxsize=1)
def health():
    """Single fail-closed verdict the app banners on. Never raises."""
    if not DB_PATH.exists():
        return {"status": NOT_FOUND, "detail": "data/results.db missing — clone incomplete"}
    try:
        tables = {r["name"] for r in _rows("select name from sqlite_master where type='table'")}
    except sqlite3.Error as exc:
        return {"status": NOT_FOUND, "detail": f"unreadable: {exc}"}
    missing = EXPECTED_TABLES - tables
    if missing:
        return {"status": SCHEMA_MISMATCH, "detail": f"missing tables: {sorted(missing)}"}
    run = research_run()
    bad = _rows("select check_name, status from integrity_checks where status != 'PASS'")
    if bad:
        return {"status": INTEGRITY_FAILED, "detail": str([b["check_name"] for b in bad][:5]), "run": run}
    freeze = run.get("presentation_freeze") or ""
    if run.get("research_status") != "FROZEN_FINAL_RESEARCH_SNAPSHOT" \
            or not freeze.startswith(EXPECTED_FREEZE_PREFIX):
        return {"status": PARTIAL,
                "detail": f"status={run.get('research_status')} freeze={freeze or '-'}", "run": run}
    counts = {r["model"]: r["n"] for r in
              _rows("select model, count(*) as n from asset_results group by model")}
    expected = {"xgb": run.get("xgb_assets"), "lstm": run.get("lstm_assets")}
    if counts != expected:
        return {"status": PARTIAL,
                "detail": f"asset rows {counts} != declared {expected}", "run": run}
    return {"status": OK, "detail": freeze, "run": run}


# ---------------------------------------------------------------- small aggregates (cached)

@lru_cache(maxsize=1)
def research_run():
    rows = _rows("select * from research_run")
    return rows[0] if rows else {}


@lru_cache(maxsize=1)
def tickers():
    return [r["ticker"] for r in _rows("select distinct ticker from asset_results order by ticker")]


@lru_cache(maxsize=1)
def integrity():
    return _rows("select * from integrity_checks order by check_name")


# ---------------------------------------------------------------- page DataFrames (cached)

@lru_cache(maxsize=1)
def results_df():
    """Every result row for distributions (Comparison) and medians (Overview)."""
    rows = _rows(
        "select ticker, model, result_mode, return_pct, profit_factor, model_trades,"
        " hodl_return_pct, beats_hodl, max_drawdown_pct, win_rate_pct,"
        " theta_entry, theta_boundary from asset_results")
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def overview_stats():
    """Per-model medians for Overview. Profit-factor stats are computed ONLY on
    rows with model_trades >= 2 and a non-null PF, and carry that coverage count."""
    df = results_df()
    out = {}
    for model, g in df.groupby("model"):
        pf = g[(g["model_trades"] >= 2) & g["profit_factor"].notna()]["profit_factor"]
        out[model] = {
            "n_assets": int(len(g)),
            "median_return_pct": float(g["return_pct"].median()),
            "median_hodl_return_pct": float(g["hodl_return_pct"].median()),
            "beats_hodl_n": int(g["beats_hodl"].sum()),
            "beats_hodl_pct": float(100.0 * g["beats_hodl"].sum() / len(g)),
            "median_profit_factor": float(pf.median()) if len(pf) else None,
            "pf_coverage_n": int(len(pf)),
        }
    return out


# ---------------------------------------------------------------- basket simulator

@lru_cache(maxsize=2)
def simulator_rows(model):
    """Capital endpoints + the win/loss split for one pipeline, for the Basket Simulator.

    Deliberately NOT part of results_df(): the simulator needs end_capital, wins,
    losses and trade_floor_met, which the distribution pages never read — widening
    the shared frame would make four pages pay for two. There is no per-trade data
    anywhere in this release, so a basket is the sum of per-asset ENDPOINTS: no
    equity curve, no drawdown path, no timing.
    """
    return pd.DataFrame(_rows(
        "select ticker, result_mode, end_capital, return_pct, model_trades, trades,"
        " wins, losses, max_drawdown_pct, win_rate_pct, profit_factor,"
        " trade_floor_met, hodl_return_pct, benchmark_trades, oos_window"
        " from asset_results where model=? order by ticker", (MODEL_KEY.get(model, model),)))


@lru_cache(maxsize=2)
def hodl_returns(model):
    """{ticker: price-only buy-and-hold return %} — the benchmark leg of a basket.
    Assets whose benchmark is null are dropped, so the caller can tell how many of
    its picks actually carry a benchmark."""
    rows = _rows("select ticker, hodl_return_pct from asset_results"
                 " where model=? and hodl_return_pct is not null",
                 (MODEL_KEY.get(model, model),))
    return {r["ticker"]: r["hodl_return_pct"] for r in rows}


@lru_cache(maxsize=1)
def payoff_ratios():
    """The REALIZED median win/loss payoff per model, recovered from sealed scalars.

    profit_factor = (wins x avg_win) / (losses x avg_loss), so avg_win/avg_loss =
    PF x losses / wins. The barrier geometry is nominally 2:1 (2xATR target against
    a 1xATR stop); this is what actually settled once triggers on the close, next-open
    fills, gaps and two-sided costs were paid. Promoted rows only, and only where both
    a win and a loss exist — the ratio is undefined otherwise.
    """
    rows = _rows("select model, profit_factor, wins, losses from asset_results"
                 " where result_mode='ML_MULTI_TRADE' and profit_factor is not null"
                 " and wins > 0 and losses > 0")
    out = {}
    for model in ("xgb", "lstm"):
        vals = pd.Series([r["profit_factor"] * r["losses"] / r["wins"]
                          for r in rows if r["model"] == model])
        out[model] = {"median_payoff": float(vals.median()) if len(vals) else None,
                      "n": int(len(vals))}
    return out


# Basket presets. Every membership is DERIVED from the store, never a typed list.
# The first five are cross-model (the same set whichever model is selected); the
# last four depend on the selected model and say so in their own label.
PRESETS = (
    ("universe", "Whole universe", False),
    ("active_both", "Promoted in both models", False),
    ("beats_both", "Beat buy & hold in both", False),
    ("idle_both", "No model result in either", False),
    ("disagree", "Models disagree on buy & hold", False),
    ("top10", "Top 10 by OOS return", True),
    ("bottom10", "Bottom 10 by OOS return", True),
    ("busiest10", "Busiest 10 by model trades", True),
    ("random10", "Random 10 (seed 0)", False),
)
PRESET_LABELS = {key: label for key, label, _ in PRESETS}
PRESET_PER_MODEL = {key: per_model for key, _, per_model in PRESETS}
# Reverse map. st.pills sends the FORMATTED label over the wire and maps it back through
# the label->option table it built during the current render; when that lookup misses it
# hands the raw label to session_state instead of the option. The labels above are constant
# for exactly that reason, and this map lets a caller recover from a label anyway.
PRESET_KEY_BY_LABEL = {label: key for key, label, _ in PRESETS}


@lru_cache(maxsize=32)
def preset_tickers(name, model):
    """One basket preset as a tuple of tickers, derived by SQL from the sealed rows.

    'Both' presets require the ticker to be present twice (the two pipelines seal
    498 and 495 assets, so three tickers exist for XGB only)."""
    key = MODEL_KEY.get(model, model)
    if name == "universe":
        sql, params = "select distinct ticker from asset_results order by ticker", ()
    elif name == "active_both":
        sql, params = ("select ticker from asset_results where result_mode='ML_MULTI_TRADE'"
                       " group by ticker having count(*)=2 order by ticker"), ()
    elif name == "beats_both":
        sql, params = ("select ticker from asset_results where beats_hodl=1"
                       " group by ticker having count(*)=2 order by ticker"), ()
    elif name == "idle_both":
        sql, params = ("select ticker from asset_results where result_mode in"
                       " ('HODL_FALLBACK_NO_MODEL_TRADES','TRAIN_OOF_FLOOR_NOT_MET')"
                       " group by ticker having count(*)=2 order by ticker"), ()
    elif name == "disagree":
        sql, params = ("select ticker from (select ticker,"
                       " max(case when model='xgb' then beats_hodl end) as x,"
                       " max(case when model='lstm' then beats_hodl end) as l"
                       " from asset_results group by ticker)"
                       " where x is not null and l is not null and x != l order by ticker"), ()
    elif name in ("top10", "bottom10"):
        order = "desc" if name == "top10" else "asc"
        sql, params = (f"select ticker from asset_results where model=?"
                       f" order by return_pct {order} limit 10"), (key,)
    elif name == "busiest10":
        sql, params = ("select ticker from asset_results where model=?"
                       " order by model_trades desc limit 10"), (key,)
    elif name == "random10":
        import random
        pool = [r["ticker"] for r in _rows(
            "select distinct ticker from asset_results order by ticker")]
        return tuple(sorted(random.Random(0).sample(pool, 10)))
    else:
        raise ValueError(f"unknown preset: {name}")
    return tuple(r["ticker"] for r in _rows(sql, params))


# ---------------------------------------------------------------- per-asset queries

def asset(ticker, model):
    """One sealed row. Accepts either vocabulary for `model` ('XGBoost' or 'xgb'), like
    every other model-keyed accessor here — a display name used to return None silently."""
    rows = _rows("select * from asset_results where ticker=? and model=?",
                 (ticker, MODEL_KEY.get(model, model)))
    return rows[0] if rows else None


def features(ticker, model):
    return _rows("select * from asset_features where ticker=? and model=? order by feature_id",
                 (ticker, model))


def contributions(ticker, model):
    return _rows("select * from feature_contributions where ticker=? and model=? "
                 "order by contribution_share desc", (ticker, model))


def train_stats(ticker, model):
    return _rows("select * from feature_train_stats where ticker=? and model=? "
                 "order by feature_position", (ticker, model))


def entry_ranges(ticker, direction=None, feature_key=None):
    sql = "select * from xgb_entry_ranges where ticker=?"
    params = [ticker]
    if direction:
        sql += " and direction=?"
        params.append(direction)
    if feature_key:
        sql += " and feature_key=?"
        params.append(feature_key)
    return _rows(sql + " order by feature_key, direction, segment_no", tuple(params))


def range_feature_keys(ticker):
    """Feature keys that have XGB entry-range segments (feeds the range explorer)."""
    return [r["feature_key"] for r in _rows(
        "select distinct feature_key from xgb_entry_ranges where ticker=? order by 1",
        (ticker,))]


def family_shares(ticker, model):
    return _rows(
        "select feature_family, max(family_share) as family_share,"
        " count(*) as n_features from feature_contributions"
        " where ticker=? and model=? group by feature_family order by family_share desc",
        (ticker, model))


# ---------------------------------------------------------------- lazy artifact JSONs

def artifact_dir(ticker, model):
    """Resolved via asset_results.artifact_path ONLY (the tracked artifacts/ tree on a
    clean clone; never a folder scan)."""
    row = asset(ticker, model)
    if not row or not row.get("artifact_path"):
        return None
    return ROOT / row["artifact_path"]


@lru_cache(maxsize=64)
def artifact_json(ticker, model, name):
    """Lazy per-asset JSON (call only AFTER an asset is selected). name must be one of
    ARTIFACT_JSONS; returns None when the file is absent (fail-soft, page shows status)."""
    if name not in ARTIFACT_JSONS:
        raise ValueError(f"not an artifact json: {name}")
    base = artifact_dir(ticker, model)
    if base is None:
        return None
    p = base / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


@lru_cache(maxsize=2)
def frozen_config(name):
    """One frozen pipeline config from config/, with the `_`-prefixed editorial keys
    dropped — those are authoring notes (some still cite research-branch paths) and
    must never reach the screen. Fail-soft: returns {} when the file is absent."""
    if name not in CONFIG_JSONS:
        raise ValueError(f"not a config json: {name}")
    p = ROOT / "config" / name
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def frozen_parameters():
    """The parameters the Integrity page has to show: label horizon, purge, embargo,
    the split boundaries, the barrier contract and the operating grid — read from the
    frozen config rather than typed into prose. Returns (xgb, lstm) dicts; a key the
    config does not declare comes back as None so the page can render it as absent."""
    out = {}
    for model, name in (("xgb", "xgb.json"), ("lstm", "lstm.json")):
        cfg = frozen_config(name)
        sp = cfg.get("splits", {})
        op = {k: v for k, v in (cfg.get("OPERATING_SPACE") or {}).items()
              if not k.startswith("_")}
        theta = op.get("theta") or []
        out[model] = {
            "H": cfg.get("H"),
            "purge": cfg.get("PURGE_CANDLES", cfg.get("PURGE_BARS")),
            "embargo": cfg.get("EMBARGO_BARS"),
            "seq_len": cfg.get("SEQ_LEN"),
            "train": f"{sp.get('train_start')} → {sp.get('train_end')}" if sp else None,
            "oos": f"{sp.get('oos_start')} → {sp.get('oos_end')}" if sp else None,
            "warmup": f"{sp.get('warmup_start')} → {sp.get('warmup_end')}" if sp else None,
            "tp_atr": cfg.get("TB_ATR_TP"),
            "sl_atr": cfg.get("TB_ATR_SL"),
            "barrier_mode": cfg.get("BARRIER_MODE"),
            "costs_bps": (cfg.get("COMMISSION_BPS"), cfg.get("SLIPPAGE_BPS")),
            "entry_fill": cfg.get("ENTRY_FILL"),
            "exit_fill": cfg.get("EXIT_FILL"),
            "scheduled_exit_fill": cfg.get("SCHEDULED_EXIT_FILL"),
            "capital_mode": cfg.get("CAPITAL_MODE"),
            "theta_grid": f"{min(theta)}–{max(theta)} (step {round(theta[1] - theta[0], 3)})"
                          if len(theta) > 1 else None,
            "min_oof_trades": op.get("min_oof_trades"),
            "seed": cfg.get("RANDOM_SEED"),
        }
    return out["xgb"], out["lstm"]


def interpretation(ticker, model):
    """The interpretation payload (per_bin / trajectories / disclaimer / labels live here)."""
    return artifact_json(ticker, model, "interpretation.json")


def interpretation_labels(ticker, model):
    """Mandatory banner text: TRAIN-DERIVED INTERPRETATION / NOT AN OOS RESULT /
    NOT A LIVE TRADING SIGNAL (from the payload, with a hard fallback)."""
    doc = interpretation(ticker, model)
    labels = (doc or {}).get("labels") or [
        "TRAIN-DERIVED INTERPRETATION", "NOT AN OOS RESULT", "NOT A LIVE TRADING SIGNAL"]
    return " · ".join(labels), (doc or {}).get("disclaimer", "")


@lru_cache(maxsize=1)
def oos_reads():
    """Per-pipeline OOS-read discipline: how many reads this epoch and the CUMULATIVE
    per-ticker counter across all epochs. The one-shot rule is the project's core
    contract, so the console has to be able to show it, not just claim it."""
    return _rows("select * from oos_read_summary order by pipe")


@lru_cache(maxsize=1)
def result_mode_matrix():
    """result_mode counts per model — the direct evidence for 'the model knows when to
    stay idle', which is otherwise scattered across Overview and Universe."""
    rows = _rows("select model, result_mode, count(*) as n from asset_results "
                 "group by model, result_mode order by model, n desc")
    models = sorted({r["model"] for r in rows})
    modes = sorted({r["result_mode"] for r in rows})
    return {"models": models, "modes": modes,
            "counts": {(r["model"], r["result_mode"]): r["n"] for r in rows}}


@lru_cache(maxsize=1)
def model_hash_coverage():
    """How many distinct sealed models the interpretation layer actually covers —
    proves the layer describes THESE artifacts, not a stale set."""
    return _rows("select model, count(distinct model_hash) as models, "
                 "count(distinct ticker) as tickers, "
                 "count(distinct interpretation_recipe_hash) as recipes "
                 "from feature_train_stats group by model order by model")
