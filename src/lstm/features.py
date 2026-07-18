#!/usr/bin/env python3
"""The feature plane — CORE features (always on) + an OPTIONAL bank the per-asset search
draws from, plus (Stage 3) a safe DSL for Claude-proposed features.

CORE (13) is the frozen minimum every model gets. OPTIONAL (ids 101+) are extra causal
daily indicators; the feature search picks a per-asset subset by Train CV AUC-PR. Every
feature here is BACKWARD-LOOKING by construction (shifts/rolling windows only), so a value
at session t uses only data at or before close[t] — no leakage possible from the feature
side. Self-contained (numpy/pandas only) so it never imports the trading mechanics.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----- shared causal primitives (the CORE frame reuses these too) -----

def wilder_atr(high, low, close, window):
    high, low, close = map(np.asarray, (high, low, close))
    prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    atr = np.full(len(tr), np.nan)
    if len(tr) >= window:
        atr[window - 1] = tr[:window].mean()
        for i in range(window, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


def ema(values, span):
    return pd.Series(values).ewm(span=span, adjust=False, min_periods=span).mean().to_numpy()


def rsi(close, window):
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.mask((avg_loss == 0.0) & (avg_gain > 0.0), 100.0)
    out = out.mask((avg_loss == 0.0) & (avg_gain == 0.0), 50.0)
    return out.to_numpy()


# ----- CORE (13) — the frozen minimum -----

CORE_FEATURE_NAMES = [
    "log_return_1", "log_return_5", "log_return_20",
    "dist_to_sma_20", "dist_to_sma_50", "sma_20_sma_50_ratio",
    "volume_z_score_20", "atr_pct_14", "realized_volatility_20",
    "rsi_14", "bollinger_percent_b_20_2", "bollinger_bandwidth_20_2",
    "macd_hist_12_26_9",
]

CORE_FEATURE_FORMULAS = {
    "log_return_1": "ln(close_t / close_{t-1})",
    "log_return_5": "ln(close_t / close_{t-5})",
    "log_return_20": "ln(close_t / close_{t-20})",
    "dist_to_sma_20": "close / SMA20(close) - 1",
    "dist_to_sma_50": "close / SMA50(close) - 1",
    "sma_20_sma_50_ratio": "SMA20(close) / SMA50(close) - 1",
    "volume_z_score_20": "(volume - mean20(volume)) / std20(volume)",
    "atr_pct_14": "100 * WilderATR14 / close",
    "realized_volatility_20": "std20(log_return_1) * sqrt(20)",
    "rsi_14": "Wilder RSI(14)",
    "bollinger_percent_b_20_2": "(close - BB_lower) / (BB_upper - BB_lower), 20/2.0",
    "bollinger_bandwidth_20_2": "(BB_upper - BB_lower) / BB_mid, 20/2.0",
    "macd_hist_12_26_9": "EMA12(close) - EMA26(close) - EMA9(macd_line)",
}


def core_frame(df, cfg):
    """The 13 CORE features + __atr_abs (barrier width). Math identical to the original
    feature_frame; `cfg` is the config dict (window sizes single-homed in config.json)."""
    import math
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    close_s, vol_s = pd.Series(c), pd.Series(v)
    log1 = np.log(close_s / close_s.shift(1)).to_numpy()
    sma20 = close_s.rolling(cfg["W_SMA_FAST"]).mean().to_numpy()
    sma50 = close_s.rolling(cfg["W_SMA_SLOW"]).mean().to_numpy()
    vol_mean = vol_s.rolling(cfg["W_VOL"]).mean().to_numpy()
    vol_std = vol_s.rolling(cfg["W_VOL"]).std(ddof=0).to_numpy()
    atr = wilder_atr(h, l, c, cfg["W_ATR"])
    bb_std = close_s.rolling(cfg["W_BB"]).std(ddof=0).to_numpy()
    bb_upper, bb_lower = sma20 + cfg["BB_K"] * bb_std, sma20 - cfg["BB_K"] * bb_std
    bb_width = bb_upper - bb_lower
    macd_line = ema(c, cfg["MACD_FAST"]) - ema(c, cfg["MACD_SLOW"])
    macd_signal = pd.Series(macd_line).ewm(span=cfg["MACD_SIGNAL"], adjust=False,
                                           min_periods=cfg["MACD_SIGNAL"]).mean().to_numpy()
    f = pd.DataFrame(index=df.index)
    f["log_return_1"] = log1
    f["log_return_5"] = np.log(close_s / close_s.shift(5)).to_numpy()
    f["log_return_20"] = np.log(close_s / close_s.shift(20)).to_numpy()
    f["dist_to_sma_20"] = c / sma20 - 1.0
    f["dist_to_sma_50"] = c / sma50 - 1.0
    f["sma_20_sma_50_ratio"] = sma20 / sma50 - 1.0
    f["volume_z_score_20"] = np.where(vol_std == 0.0, 0.0, (v - vol_mean) / vol_std)
    f["atr_pct_14"] = 100.0 * atr / c
    f["realized_volatility_20"] = pd.Series(log1).rolling(20).std(ddof=0).to_numpy() * math.sqrt(20.0)
    f["rsi_14"] = rsi(c, cfg["W_RSI"])
    f["bollinger_percent_b_20_2"] = np.where(bb_width == 0.0, 0.5, (c - bb_lower) / bb_width)
    f["bollinger_bandwidth_20_2"] = bb_width / sma20
    f["macd_hist_12_26_9"] = macd_line - macd_signal
    f["__atr_abs"] = atr
    return f


# ----- OPTIONAL bank (ids 101+) — the search pool -----

OPTIONAL_FEATURE_FORMULAS = {
    "momentum_60": "ln(close / close_{t-60})",
    "return_120": "ln(close / close_{t-120})",
    "gap_open_z20": "zscore(open/close_{t-1} - 1, 20)",
    "close_position_in_range": "(close - low) / (high - low)",
    "high_low_range_pct": "100 * (high - low) / close",
    "volume_trend_5_20": "SMA5(volume) / SMA20(volume) - 1",
    "obv_z20": "zscore(cumsum(sign(dclose) * volume), 20)",
    "atr_ratio_5_20": "WilderATR5 / WilderATR20",
    "dist_to_high_252": "close / max(high, 252) - 1",
    "drawdown_from_max_60": "close / max(close, 60) - 1",
    "ret_skew_20": "rolling skew of log_return_1 over 20",
    "ret_kurt_20": "rolling kurtosis of log_return_1 over 20",
    "up_streak_10": "min(consecutive up-days, 10) / 10",
    "rsi_2": "Wilder RSI(2) (fast mean-reversion)",
    "dist_to_sma_200": "close / SMA200(close) - 1",
    # --- mean-reversion family (how stretched from a local mean / where in the range) ---
    "close_z_score_20": "(close - SMA20(close)) / std20(close)",
    "williams_r_14": "-100 * (max(high,14) - close) / (max(high,14) - min(low,14))",
    "stoch_k_14": "100 * (close - min(low,14)) / (max(high,14) - min(low,14))",
    "dist_to_sma_10": "close / SMA10(close) - 1",
    "down_streak_10": "min(consecutive down-days, 10) / 10",
    "reversal_2d": "-ln(close / close_{t-2}) (2-day mean-reversion signal)",
    "dist_from_low_20": "close / min(low, 20) - 1",
    "mr_dist_atr_20": "(close - SMA20(close)) / WilderATR14 (reversion distance in ATR units)",
}

# stable id assignment (101..) — the manifest orders features by id
OPTIONAL_FEATURE_IDS = {name: 101 + i for i, name in enumerate(OPTIONAL_FEATURE_FORMULAS)}
OPTIONAL_FEATURE_NAMES = list(OPTIONAL_FEATURE_FORMULAS)


def _zscore(x, w):
    s = pd.Series(x)
    return ((s - s.rolling(w).mean()) / s.rolling(w).std(ddof=0).replace(0.0, np.nan)).to_numpy()


def optional_frame(df):
    """All OPTIONAL features as columns. Every column is causal (shift/rolling only).
    Warmup rows are legitimately NaN (long windows) and get filtered out downstream, so
    divide/invalid warnings during the warmup are expected and silenced."""
    with np.errstate(all="ignore"):
        return _optional_frame(df)


def _optional_frame(df):
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    close_s = pd.Series(c)
    log1 = np.log(close_s / close_s.shift(1)).to_numpy()
    rng = h - l
    up = (close_s.diff() > 0).astype(int)
    streak = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    obv = np.cumsum(np.sign(np.nan_to_num(close_s.diff().to_numpy())) * v)
    f = pd.DataFrame(index=df.index)
    f["momentum_60"] = np.log(close_s / close_s.shift(60)).to_numpy()
    f["return_120"] = np.log(close_s / close_s.shift(120)).to_numpy()
    f["gap_open_z20"] = _zscore(o / np.concatenate([[np.nan], c[:-1]]) - 1.0, 20)
    f["close_position_in_range"] = np.where(rng == 0.0, 0.5, (c - l) / rng)
    f["high_low_range_pct"] = 100.0 * rng / c
    f["volume_trend_5_20"] = (pd.Series(v).rolling(5).mean() / pd.Series(v).rolling(20).mean() - 1.0).to_numpy()
    f["obv_z20"] = _zscore(obv, 20)
    f["atr_ratio_5_20"] = wilder_atr(h, l, c, 5) / wilder_atr(h, l, c, 20)
    f["dist_to_high_252"] = c / pd.Series(h).rolling(252).max().to_numpy() - 1.0
    f["drawdown_from_max_60"] = c / close_s.rolling(60).max().to_numpy() - 1.0
    f["ret_skew_20"] = pd.Series(log1).rolling(20).skew().to_numpy()
    f["ret_kurt_20"] = pd.Series(log1).rolling(20).kurt().to_numpy()
    f["up_streak_10"] = np.minimum(streak.to_numpy(), 10) / 10.0
    f["rsi_2"] = rsi(c, 2)
    f["dist_to_sma_200"] = c / close_s.rolling(200).mean().to_numpy() - 1.0
    # --- mean-reversion family ---
    sma20 = close_s.rolling(20).mean().to_numpy()
    std20 = close_s.rolling(20).std(ddof=0).to_numpy()
    hh14 = pd.Series(h).rolling(14).max().to_numpy()
    ll14 = pd.Series(l).rolling(14).min().to_numpy()
    rng14 = hh14 - ll14
    down = (close_s.diff() < 0).astype(int)
    dstreak = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
    f["close_z_score_20"] = np.where(std20 == 0.0, 0.0, (c - sma20) / std20)
    f["williams_r_14"] = np.where(rng14 == 0.0, -50.0, -100.0 * (hh14 - c) / rng14)
    f["stoch_k_14"] = np.where(rng14 == 0.0, 50.0, 100.0 * (c - ll14) / rng14)
    f["dist_to_sma_10"] = c / close_s.rolling(10).mean().to_numpy() - 1.0
    f["down_streak_10"] = np.minimum(dstreak.to_numpy(), 10) / 10.0
    f["reversal_2d"] = -np.log(close_s / close_s.shift(2)).to_numpy()
    f["dist_from_low_20"] = c / pd.Series(l).rolling(20).min().to_numpy() - 1.0
    f["mr_dist_atr_20"] = (c - sma20) / wilder_atr(h, l, c, 14)
    return f[OPTIONAL_FEATURE_NAMES]


# ==================== XAS mini-tier (ids 130+): broad-market context ====================
# Pilot cross-asset features (max 2 by design). The market reference is the CAUSAL equal-weight
# ex-self mean of daily log returns over the FULL committed universe (sp500_1d.duckdb, ~503
# symbols) — never a small basket. Computed fresh like the PROPOSED tier (never cached), so the
# CORE/OPTIONAL cache token is untouched. All inputs are completed daily bars: the reference for
# session t uses closes <= t (same availability as the ticker's own close at decision time);
# the *lead* feature additionally shifts the reference by one full session.

XAS_FEATURE_FORMULAS = {
    "xas_mkt_lead_ret_1": "previous session's broad-market log return (eq-weight ex-self mean, full universe)",
    "xas_rs_mkt_20": "sum20(own log return) - sum20(broad-market ex-self log return)",
}
XAS_FEATURE_IDS = {"xas_mkt_lead_ret_1": 130, "xas_rs_mkt_20": 131}
XAS_FEATURE_NAMES = list(XAS_FEATURE_FORMULAS)
_XAS_CTX = {}


def _xas_market_context(db_path):
    """{date -> (sum of per-symbol daily log returns, count of symbols with a return)} over the
    full committed store. One window-function query, cached per process."""
    key = str(db_path)
    if key not in _XAS_CTX:
        import duckdb
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                "with lr as (select date, ln(close / lag(close) over (partition by symbol order by date)) r "
                "from bars_1d) "
                "select date, sum(r), count(r) from lr where r is not null group by date order by date"
            ).fetchall()
        finally:
            con.close()
        _XAS_CTX[key] = {pd.Timestamp(d): (float(s), int(n)) for d, s, n in rows}
    return _XAS_CTX[key]


def xas_frame(df, ticker, db_path):
    """The 2 XAS pilot columns for one ticker, aligned to df. ticker=None (no identity for the
    ex-self rule) yields NaN columns so superset shapes stay stable."""
    out = pd.DataFrame(index=range(len(df)))
    if ticker is None:
        for n in XAS_FEATURE_NAMES:
            out[n] = np.nan
        return out
    with np.errstate(all="ignore"):
        ctx = _xas_market_context(db_path)
        c = pd.Series(df["close"].to_numpy(float))
        own = np.log(c / c.shift(1)).to_numpy()
        dates = pd.to_datetime(df["date"]).dt.normalize()
        sums = np.array([ctx.get(d, (np.nan, 0))[0] for d in dates], float)
        ns = np.array([ctx.get(d, (np.nan, 0))[1] for d in dates], float)
        # ex-self equal weight: exclude the ticker's own return from the market mean
        ref = np.where(ns > 1, (sums - own) / (ns - 1), np.nan)
        ref_s = pd.Series(ref)
        out["xas_mkt_lead_ret_1"] = ref_s.shift(1).to_numpy()
        out["xas_rs_mkt_20"] = (pd.Series(own).rolling(20, min_periods=20).sum()
                                - ref_s.rolling(20, min_periods=20).sum()).to_numpy()
    return out


# ============================ Claude-proposed features (safe DSL) ============================
# A Claude Opus agent may propose NEW features as small expressions over daily OHLCV. The
# grammar exposes ONLY backward-looking primitives, so any expression that parses is causal by
# construction (no way to reference the future). Proposals are validated fail-closed before they
# ever enter the search pool (features_proposed.json, ids 501+).
import ast

_DSL_VARS = {"o", "h", "l", "c", "v"}
_DSL_FUNCS = {"shift", "rolling_mean", "rolling_std", "rolling_min", "rolling_max",
              "ewm", "zscore", "rank", "log", "abs", "sign", "clip"}


def _dsl_check(node):
    """Whitelist the AST: only the allowed vars/funcs/ops/number literals; shift needs n>=1."""
    if isinstance(node, ast.Expression):
        return _dsl_check(node.body)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        _dsl_check(node.left); _dsl_check(node.right); return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _dsl_check(node.operand); return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _DSL_FUNCS:
            raise ValueError("call not allowed")
        if node.keywords:
            raise ValueError("keyword args not allowed")
        for a in node.args:
            _dsl_check(a)
        if node.func.id == "shift":
            n = node.args[1] if len(node.args) > 1 else None
            if not (isinstance(n, ast.Constant) and isinstance(n.value, int) and n.value >= 1):
                raise ValueError("shift(x, n) requires an integer n >= 1 (backward only)")
        return
    if isinstance(node, ast.Name):
        if node.id not in _DSL_VARS:
            raise ValueError(f"name not allowed: {node.id}")
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return
    raise ValueError(f"node not allowed: {type(node).__name__}")


def _dsl_env(df):
    def S(x):
        return x if isinstance(x, pd.Series) else pd.Series(x)
    return {
        "__builtins__": {},
        "o": pd.Series(df["open"].to_numpy(float)), "h": pd.Series(df["high"].to_numpy(float)),
        "l": pd.Series(df["low"].to_numpy(float)), "c": pd.Series(df["close"].to_numpy(float)),
        "v": pd.Series(df["volume"].to_numpy(float)),
        "shift": lambda x, n: S(x).shift(int(n)),
        "rolling_mean": lambda x, w: S(x).rolling(int(w)).mean(),
        "rolling_std": lambda x, w: S(x).rolling(int(w)).std(ddof=0),
        "rolling_min": lambda x, w: S(x).rolling(int(w)).min(),
        "rolling_max": lambda x, w: S(x).rolling(int(w)).max(),
        "ewm": lambda x, span: S(x).ewm(span=int(span), adjust=False, min_periods=int(span)).mean(),
        "zscore": lambda x, w: (S(x) - S(x).rolling(int(w)).mean()) / S(x).rolling(int(w)).std(ddof=0).replace(0.0, np.nan),
        "rank": lambda x, w: S(x).rolling(int(w)).apply(lambda a: float((a <= a[-1]).mean()), raw=True),
        "log": lambda x: np.log(S(x)),
        "abs": lambda x: S(x).abs(),
        "sign": lambda x: np.sign(S(x)),
        "clip": lambda x, lo, hi: S(x).clip(float(lo), float(hi)),
    }


def dsl_eval(expr, df):
    """Compile + evaluate a validated DSL expression to a numpy array aligned to df."""
    tree = ast.parse(expr, mode="eval")
    _dsl_check(tree)
    with np.errstate(all="ignore"):
        out = eval(compile(tree, "<dsl>", "eval"), _dsl_env(df))
    return np.asarray(pd.Series(out).to_numpy(float))


def validate_proposal(name, expr, sample_df, known_names):
    """Fail-closed gate a proposal must pass before it can enter the search pool:
    parses under the whitelist, evaluates, ≥80% finite after warmup, not constant, unique name.
    Returns (ok, reason)."""
    if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
        return False, "name must be a simple identifier"
    if name in known_names:
        return False, "name already exists"
    try:
        vals = dsl_eval(expr, sample_df)
    except Exception as e:
        return False, f"invalid DSL: {e}"
    if len(vals) != len(sample_df):
        return False, "length mismatch"
    tail = vals[252:]                                    # after the warmup year
    finite = np.isfinite(tail)
    if finite.mean() < 0.99:                              # dense: it joins the COMMON event set; a
        return False, f"only {finite.mean():.0%} finite after warmup (need >=99%)"  # sparse feature shifts prevalence
    if np.nanstd(tail[finite]) < 1e-12:
        return False, "constant feature"
    return True, "ok"


def proposed_registry(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def proposed_frame(df, path):
    """Evaluate every registered proposed feature (features_proposed.json) into a frame.
    A feature that fails to evaluate for a given ticker becomes all-NaN (the sequence
    finiteness filter then just drops it for that ticker) — one odd ticker never crashes
    the run."""
    reg = proposed_registry(path)
    f = pd.DataFrame(index=df.index)
    for nm, spec in reg.items():
        if nm.startswith("_"):                          # skip the _next_id monotonic sentinel
            continue
        try:
            f[nm] = dsl_eval(spec["expr"], df)
        except Exception:
            f[nm] = np.nan
    return f
