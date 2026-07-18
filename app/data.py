"""app/data.py — the ONE data-access module for the presentation console.

The ONLY place in the app that:
- opens data/results.db (sqlite, mode=ro),
- verifies the schema and dataset completeness (fail-closed statuses),
- holds every SQL query the six pages need,
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
}
EXPECTED_FREEZE_PREFIX = "public/"
ARTIFACT_JSONS = ("manifest.json", "parameters.json", "metrics.json", "interpretation.json")

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
def model_summary():
    return _rows("select * from v_model_summary order by model")


@lru_cache(maxsize=1)
def universe():
    """The operational table (page 2.2): one row per ticker, both models pivoted."""
    return _rows("select * from v_universe_summary order by ticker")


@lru_cache(maxsize=1)
def tickers():
    return [r["ticker"] for r in _rows("select distinct ticker from asset_results order by ticker")]


@lru_cache(maxsize=1)
def result_mode_shares():
    """Overview panel 'kiedy model jest bezczynny': result_mode counts per model."""
    return _rows("select model, result_mode, count(*) as n from asset_results "
                 "group by model, result_mode order by model, n desc")


@lru_cache(maxsize=1)
def integrity():
    return _rows("select * from integrity_checks order by check_name")


@lru_cache(maxsize=1)
def family_contribution_summary():
    """Universe-level 'z czego składają się opisy': mean family share per model."""
    return _rows("select model, feature_family, avg(family_share) as mean_family_share, "
                 "count(distinct ticker) as assets from feature_contributions "
                 "group by model, feature_family order by model, mean_family_share desc")


# ---------------------------------------------------------------- page DataFrames (cached)

@lru_cache(maxsize=1)
def universe_df():
    """Universe page: one row per ticker, both models pivoted. HODL is taken
    explicitly from the XGB row — the two models have different OOS windows, so a
    cross-model max() (as in v_universe_summary) would belong to neither window."""
    rows = _rows(
        "select ticker,"
        " max(case when model='xgb'  then result_mode end)  as xgb_status,"
        " max(case when model='lstm' then result_mode end)  as lstm_status,"
        " max(case when model='xgb'  then return_pct end)   as xgb_return_pct,"
        " max(case when model='lstm' then return_pct end)   as lstm_return_pct,"
        " max(case when model='xgb'  then hodl_return_pct end) as hodl_return_pct,"
        " max(case when model='xgb'  then model_trades end) as xgb_trades,"
        " max(case when model='lstm' then model_trades end) as lstm_trades"
        " from asset_results group by ticker order by ticker")
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def results_df():
    """All 993 result rows for distributions (Comparison) and medians (Overview)."""
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


# ---------------------------------------------------------------- per-asset queries

def asset(ticker, model):
    rows = _rows("select * from asset_results where ticker=? and model=?", (ticker, model))
    return rows[0] if rows else None


def asset_models(ticker):
    return _rows("select * from asset_results where ticker=? order by model", (ticker,))


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


def parameters(ticker, model):
    return artifact_json(ticker, model, "parameters.json")


def manifest(ticker, model):
    return artifact_json(ticker, model, "manifest.json")


def metrics(ticker, model):
    return artifact_json(ticker, model, "metrics.json")


def calibration(ticker, model):
    """metrics.json['calibration'] — the ONLY source of direction_mode (the DB
    column is null for every row); also theta/floor/OOF details per model."""
    doc = metrics(ticker, model)
    return (doc or {}).get("calibration")
