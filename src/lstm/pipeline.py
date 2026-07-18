#!/usr/bin/env python3
"""1000-LSTM-Liora pipeline: layers D1-D6 + the trade engine.

D1  bundled daily store   data/sp500_1d.duckdb::bars_1d — the frozen input (no acquisition here)
D2  load + source QC      one ticker ordered by date; corrupt OHLCV ⇒ RuntimeError, never cleaned
D3  time split            warmup / train / OOS masks + index bounds; purge (=H) + embargo on events
D4  daily features        13 causal indicators; z-scored with TRAIN-only per-asset stats
D5  candidates + label    side = sign(log_return_5); symmetric ATR Triple Barrier, H sessions;
                          entry next open, costs both sides; label-uniqueness weights
D6  sequence tensor       per candidate: the SEQ_LEN×13 window of normalized features ending at t0

The engine (run_engine / hodl_fallback), the purged walk-forward folds and the tie-invariant
average_precision are ports of the parent XGBoost project (liora-project-ml-engineering) —
identical trade mechanics, so results differ only by the model. Everything is deterministic
for a fixed seed; the only leakage controls are structural (purge/embargo asserts, TRAIN-only
normalization, one-shot OOS in run_asset.py).
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import features

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "lstm.json"
CONFIG = {k: v for k, v in json.loads(CONFIG_PATH.read_text(encoding="utf-8")).items()
          if not k.startswith("_")}
# Experiment override (mirrors OOS_METRICS_DB): the CAPITAL_MODE env var flips the capital policy
# for one run without touching the committed config. Fail-closed on unknown values.
CONFIG.setdefault("CAPITAL_MODE", "kelly_fractional_compounding")
if os.environ.get("CAPITAL_MODE"):
    CONFIG["CAPITAL_MODE"] = os.environ["CAPITAL_MODE"]
if CONFIG["CAPITAL_MODE"] not in {"kelly_fractional_compounding", "all_in_compounding_per_asset"}:
    raise RuntimeError(f"invalid CAPITAL_MODE {CONFIG['CAPITAL_MODE']!r}")
DB_PATH = ROOT / "data" / "sp500_1d.duckdb"
EPS = float(CONFIG["EPS"])
assert CONFIG["PURGE_BARS"] == CONFIG["H"], \
    "config: PURGE_BARS must equal H — the purge window IS the label window"
# The feature plane lives in features.py. CORE is always on; the per-asset manifest is
# CORE + the OPTIONAL/PROPOSED ids selected for the ticker (feature search — Stage 2/3).
CORE_FEATURE_NAMES = features.CORE_FEATURE_NAMES
FEATURE_NAMES = CORE_FEATURE_NAMES                 # the default manifest (no overrides)
FEATURE_FORMULAS = {**features.CORE_FEATURE_FORMULAS, **features.OPTIONAL_FEATURE_FORMULAS}
# Overridable for experiments (mirrors OOS_METRICS_DB): point PER_ASSET_OVERRIDES_PATH at an
# experimental overrides file to run pilot manifests without touching the committed one.
OVERRIDES_PATH = Path(os.environ.get("PER_ASSET_OVERRIDES_PATH")
                      or (ROOT / "data" / "per_asset_feature_overrides.json"))
PROPOSED_PATH = ROOT / "features_proposed.json"


def optional_id_to_name():
    """id -> name for OPTIONAL (101+), XAS (130+) and Claude-PROPOSED (501+) features."""
    m = {i: n for n, i in features.OPTIONAL_FEATURE_IDS.items()}
    m.update({i: n for n, i in features.XAS_FEATURE_IDS.items()})
    if PROPOSED_PATH.exists():
        for name, spec in json.loads(PROPOSED_PATH.read_text(encoding="utf-8")).items():
            if name.startswith("_"):                   # skip the _next_id monotonic sentinel
                continue
            m[int(spec["id"])] = name
    return m


def load_overrides():
    """The raw per-asset overrides document ({} when absent)."""
    if not OVERRIDES_PATH.exists():
        return {}
    return json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))


def override_record(ticker):
    """The ticker's override entry normalized to the provenance-carrying shape
    {"selected_optional_ids": [...], "provenance": {...}} (#17-18), or None. A legacy bare
    id list normalizes to an entry with EMPTY provenance — the sealed-run staleness check
    then refuses it (no recipe_hash == stale by definition)."""
    e = load_overrides().get(ticker)
    if e is None:
        return None
    if isinstance(e, dict):
        return {"selected_optional_ids": list(e.get("selected_optional_ids", [])),
                "provenance": dict(e.get("provenance", {}))}
    return {"selected_optional_ids": list(e), "provenance": {}}


def resolve_manifest(ticker):
    """The per-asset model manifest: CORE (always) + the OPTIONAL/PROPOSED ids selected for
    this ticker in per_asset_feature_overrides.json, ordered by id. Default = CORE only, so
    a run with no overrides is byte-identical to the core-only pipeline. Fail-closed: an
    override id the catalogue no longer knows raises instead of silently shrinking the
    manifest into a DIFFERENT sealed row (parity with xgb's _override_selection)."""
    core = list(features.CORE_FEATURE_NAMES)
    rec = override_record(ticker)
    ids = [] if rec is None else rec["selected_optional_ids"]
    id2name = optional_id_to_name()
    unknown = sorted(int(i) for i in ids if i not in id2name)
    if unknown:
        raise RuntimeError(f"per-asset override for {ticker} names unknown feature ids: {unknown}")
    return core + [id2name[i] for i in sorted(ids)]


def resolve_superset_manifest():
    """CORE + every OPTIONAL + every PROPOSED feature name, ordered by id. The feature search
    fixes its event set to what is finite under THIS superset, so every candidate subset is
    scored on one common set of events and folds (no long-warmup feature can win selection by
    deleting hard early samples)."""
    id2name = optional_id_to_name()
    return list(features.CORE_FEATURE_NAMES) + [id2name[i] for i in sorted(id2name)]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def session_date(df, t):
    """Daily bars: every ledger timestamp is a session date (decision = close of day t)."""
    return str(pd.Timestamp(df["date"].iloc[int(t)]).date())


# ============================ D2 — load + source QC ============================

def load_bars(ticker):
    import duckdb
    con = duckdb.connect(f"{DB_PATH}", read_only=True)
    try:
        df = con.execute("select date, open, high, low, close, volume from bars_1d "
                         "where symbol=? order by date", [ticker]).fetchdf()
    finally:
        con.close()
    if df.empty:
        raise RuntimeError(f"D2: no rows for {ticker} in {DB_PATH.name}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.astype({"open": float, "high": float, "low": float, "close": float,
                    "volume": float}).reset_index(drop=True)
    errs = []
    if int(df["date"].isna().sum()):
        errs.append("unparseable date(s)")
    if int(df["date"].duplicated().sum()):
        errs.append("duplicate date(s)")
    if not df["date"].is_monotonic_increasing:
        errs.append("dates are not strictly increasing")
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    ohlc = df[["open", "high", "low", "close"]].to_numpy()
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        errs.append("non-finite or <= 0 OHLC value(s)")
    if (h < l).any():
        errs.append("bar(s) with high < low")
    if (h < np.maximum(o, c)).any():
        errs.append("bar(s) with high < max(open, close)")
    if (l > np.minimum(o, c)).any():
        errs.append("bar(s) with low > min(open, close)")
    vol = df["volume"].to_numpy(float)
    if not np.isfinite(vol).all() or (vol < 0).any():
        errs.append("non-finite or negative volume")
    if errs:
        raise RuntimeError(f"D2 source QC FAILED for {ticker}: " + "; ".join(errs))
    return df


# ============================ D3 — time split ============================

def split_masks(df):
    sp = CONFIG["splits"]
    d = df["date"]
    warmup = (d >= sp["warmup_start"]) & (d <= sp["warmup_end"])
    train = (d >= sp["train_start"]) & (d <= sp["train_end"])
    oos = (d >= sp["oos_start"]) & (d <= sp["oos_end"])
    tr_idx, oos_idx = np.where(train.to_numpy())[0], np.where(oos.to_numpy())[0]
    if not len(tr_idx) or not len(oos_idx):
        raise RuntimeError("D3: empty train or OOS window for this ticker")
    bounds = {"train_start_idx": int(tr_idx[0]), "train_end_idx": int(tr_idx[-1]),
              "oos_start_idx": int(oos_idx[0]), "oos_end_idx": int(oos_idx[-1])}
    return {"warmup": warmup.to_numpy(), "train": train.to_numpy(), "oos": oos.to_numpy()}, bounds


def purge_train_events(events, bounds):
    H, emb, oos0 = CONFIG["H"], CONFIG["EMBARGO_BARS"], bounds["oos_start_idx"]
    kept = [e for e in events if e["t0"] + H + emb <= oos0]
    assert all(e["t0"] + H < oos0 for e in kept), "purge boundary assertion failed: label crosses into OOS"
    return kept, len(events) - len(kept)


# ============================ D4 — features + TRAIN-only z-score ============================

CACHE_DIR = ROOT / "cache" / "features"
CACHE_VERSION = ROOT / "cache" / "VERSION"


def cache_token():
    """Invalidates the feature cache automatically if the CORE/OPTIONAL bank ever changes."""
    import hashlib
    key = ",".join(features.CORE_FEATURE_NAMES) + "|" + ",".join(features.OPTIONAL_FEATURE_NAMES)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _core_optional_frame(df, ticker):
    """The frozen CORE+OPTIONAL superset (+ __atr_abs). Loaded from cache/features/<T>.parquet
    when fresh (held in RAM page-cache; the OS keeps the ~503 small parquets resident) — else
    computed. Parquet round-trips float64 exactly, so the cached frame is byte-identical."""
    if ticker is not None and CACHE_VERSION.exists() and CACHE_VERSION.read_text().strip() == cache_token():
        cf = CACHE_DIR / f"{ticker}.parquet"
        if cf.exists():
            fr = pd.read_parquet(cf)
            if len(fr) == len(df):
                return fr.reset_index(drop=True)
    return pd.concat([features.core_frame(df, CONFIG), features.optional_frame(df)], axis=1)


def feature_frame(df, ticker=None):
    """The full causal feature SUPERSET (CORE 13 + __atr_abs) + OPTIONAL bank [+ Claude-PROPOSED].
    CORE+OPTIONAL are frozen and served from the RAM cache when built (`make build-cache`);
    PROPOSED (few, DSL) are always computed fresh, so the cache never goes stale when the Opus
    agent adds features. The per-asset manifest selects which columns feed the model; core-only
    is byte-identical to the original pipeline."""
    frame = _core_optional_frame(df, ticker)
    frame = pd.concat([frame, features.xas_frame(df, ticker, DB_PATH)], axis=1)   # XAS tier (fresh)
    if PROPOSED_PATH.exists():                              # Claude-proposed DSL features (fresh)
        frame = pd.concat([frame, features.proposed_frame(df, PROPOSED_PATH)], axis=1)
    return frame


def train_norm_stats(feature_df, train_mask, names, before_idx=None):
    """Per-asset μ/σ over TRAIN rows only for the manifest `names` (finite values only;
    σ floor 1e-8). With before_idx, only train rows STRICTLY BEFORE that bar index count —
    the fold-causal variant used inside CV, so a fold's scaler never sees its validation
    region or the future. Without it: the full-Train stats frozen into the artifact."""
    mask = np.asarray(train_mask, bool)
    if before_idx is not None:
        mask = mask & (np.arange(len(mask)) < int(before_idx))
    stats = {}
    for n in names:
        x = feature_df[n].to_numpy(float)[mask]
        x = x[np.isfinite(x)]
        mu = float(x.mean()) if len(x) else 0.0
        sd = float(x.std()) if len(x) else 1.0
        stats[n] = {"mean": mu, "std": max(sd, 1e-8)}
    return stats


def normalize(feature_df, stats, names):
    out = pd.DataFrame(index=feature_df.index)
    for n in names:
        s = stats[n]
        out[n] = (feature_df[n].to_numpy(float) - s["mean"]) / s["std"]
    return out


# ============================ D5 — candidates + Triple Barrier ============================

def generate_candidates(df, scan_mask, features):
    """Meta-labeling primary: the momentum-direction entry, with an asymmetric ATR Triple
    Barrier (TP = TB_ATR_TP*ATR, SL = TB_ATR_SL*ATR). Both widths are set from ATR(t0), i.e.
    data <= close[t0] — strictly causal. TB_ATR_TP==TB_ATR_SL reproduces the symmetric v1."""
    atr = features["__atr_abs"].to_numpy(float)
    momentum = features[CONFIG["SIGNAL_MOMENTUM_FEATURE"]].to_numpy(float)
    tp_mult, sl_mult = float(CONFIG["TB_ATR_TP"]), float(CONFIG["TB_ATR_SL"])
    # Meta-labeling PRIMARY gate (fixed, Train-only, causal at t0): keep only higher-conviction
    # setups. enabled=false => original every-bar candidate set.
    eg = CONFIG.get("ENTRY_GATE", {})
    gate_on = bool(eg.get("enabled", False))
    trend = (features[eg["trend_feature"]].to_numpy(float)
             if gate_on and eg.get("require_trend_agreement") else None)
    vfloor = (features[eg["vol_floor_feature"]].to_numpy(float)
              if gate_on and float(eg.get("vol_floor_k", 0.0)) > 0 else None)
    vk = float(eg.get("vol_floor_k", 0.0))
    events = []
    for t0 in np.where(scan_mask)[0]:
        if t0 + 1 >= len(df):
            continue
        mom, w = momentum[t0], atr[t0]
        width_tp, width_sl = w * tp_mult, w * sl_mult
        if not (math.isfinite(mom) and math.isfinite(w) and width_tp > EPS and width_sl > EPS):
            continue
        direction = 1 if mom > 0 else (-1 if mom < 0 else 0)
        if direction == 0:
            continue
        if gate_on:
            if trend is not None:                       # (a) momentum must agree with the trend proxy
                tr = trend[t0]
                if not math.isfinite(tr) or (1 if tr > 0 else (-1 if tr < 0 else 0)) != direction:
                    continue
            if vfloor is not None:                      # (b) the move must be significant vs local vol
                vf = vfloor[t0]
                if not (math.isfinite(vf) and abs(mom) >= vk * vf):
                    continue
        events.append({"direction": int(direction), "t0": int(t0),
                       "width_tp": float(width_tp), "width_sl": float(width_sl)})
    return events


def simulate_trade(df, event, end_idx):
    H = CONFIG["H"]
    fee = CONFIG["COMMISSION_BPS"] * 1e-4
    slip = CONFIG["SLIPPAGE_BPS"] * 1e-4
    s = event["direction"]
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    t0, t_fill = event["t0"], event["t0"] + 1
    if t_fill > end_idx:
        return {"skip": "GAP_INVALIDATED_SKIP"}
    entry_fill = o[t_fill] * (1 + s * slip)
    width_tp, width_sl = float(event["width_tp"]), float(event["width_sl"])
    if not (math.isfinite(width_tp) and width_tp > EPS and math.isfinite(width_sl) and width_sl > EPS):
        return {"skip": "INVALID_BARRIER_SKIP"}
    tp, sl = entry_fill + s * width_tp, entry_fill - s * width_sl
    t_sched = min(t0 + H, end_idx)
    exit_idx = exit_fill = reason = kind = trig = None
    for t in range(t_fill, t_sched):
        target_hit = s * (c[t] - tp) >= 0
        stop_hit = s * (c[t] - sl) <= 0
        if target_hit or stop_hit:
            if t + 1 > end_idx:
                break
            exit_idx, kind, trig = t + 1, "condition", t
            exit_fill = o[t + 1] * (1 - s * slip)
            reason = "TARGET_TRIGGER" if target_hit and not stop_hit else "STOP_TRIGGER"
            break
    if exit_idx is None:
        exit_idx, kind, trig = t_sched, "scheduled", t_sched
        exit_fill = c[t_sched] * (1 - s * slip)
        reason = "OOS_END_FORCED_EXIT" if t_sched == end_idx else "TIME_BARRIER"
    per_unit = (s * (exit_fill - entry_fill) - fee * (entry_fill + exit_fill)) / (entry_fill * (1 + fee))
    return {"skip": None, "t_fill": t_fill, "entry_fill": float(entry_fill), "exit_idx": int(exit_idx),
            "exit_fill": float(exit_fill), "market_exit_reason": reason, "exit_kind": kind,
            "trigger_idx": int(trig), "local_per_unit_net_return": float(per_unit),
            "target_level": float(tp), "stop_level": float(sl),
            "barrier_width": float(width_sl), "barrier_width_tp": float(width_tp)}


def _uniqueness_weights(actionable):
    if not actionable:
        return {}
    lo = min(a["t_fill"] for a in actionable)
    hi = max(a["t_end"] for a in actionable)
    conc = np.zeros(hi - lo + 2)
    for a in actionable:
        conc[a["t_fill"] - lo:a["t_end"] - lo + 1] += 1
    return {a["key"]: float(np.mean(1.0 / np.maximum(1.0, conc[a["t_fill"] - lo:a["t_end"] - lo + 1])))
            for a in actionable}


def label_events(df, events, end_idx):
    """Simulate every candidate: keep the labeled ones (Y = net>0) with their label spans.
    Uniqueness weights are NOT assigned here — call assign_uniqueness_weights on the FINAL
    training set (after sequence-eligibility filtering), so concurrency is measured over
    exactly the samples the model trains on."""
    labeled = []
    for ev in events:
        sim = simulate_trade(df, ev, end_idx)
        if sim["skip"] is not None:
            continue
        labeled.append(dict(ev, y=1 if sim["local_per_unit_net_return"] > 0 else 0,
                            net_return=sim["local_per_unit_net_return"],
                            t_fill=sim["t_fill"], t_end=min(ev["t0"] + CONFIG["H"], end_idx)))
    return labeled


def assign_uniqueness_weights(events):
    """Label-uniqueness weights (mean 1/concurrency over the holding window) computed over
    exactly the given event set — call it AFTER every eligibility filter."""
    weights = _uniqueness_weights([{"key": e["t0"], "t_fill": e["t_fill"], "t_end": e["t_end"]}
                                   for e in events])
    for e in events:
        e["weight"] = weights.get(e["t0"], 1.0)
    return events


# ============================ D6 — sequence tensor ============================

def build_sequences(events, norm_features, names):
    """X[i] = the SEQ_LEN×len(names) window of normalized manifest features ending at t0
    (inclusive). Only windows with every value finite are eligible — no imputation, ever."""
    W = CONFIG["SEQ_LEN"]
    arr = norm_features[names].to_numpy(np.float32)
    kept, X = [], []
    for e in events:
        t0 = e["t0"]
        if t0 < W - 1:
            continue
        win = arr[t0 - W + 1:t0 + 1]
        if not np.isfinite(win).all():
            continue
        kept.append(e)
        X.append(win)
    Xa = np.stack(X) if X else np.empty((0, W, len(names)), np.float32)
    return kept, Xa


# ============================ CV folds + metric (ports) ============================

def average_precision(y, score):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pos = int(y.sum())
    if pos == 0:
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    y, s = y[order], score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    grp = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    recall = tp[grp] / pos
    precision = tp[grp] / np.maximum(1, tp[grp] + fp[grp])
    rprev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - rprev) * precision))


def purged_wf_folds(t0s, train_start_idx, train_end_idx, k=None):
    """(train_indices, val_indices, val_lo) triples. val_lo is the fold's validation
    boundary bar index — fold-causal normalization stats must use rows < val_lo - EMBARGO."""
    H, emb = CONFIG["H"], CONFIG["EMBARGO_BARS"]
    k = k or CONFIG["CV_FOLDS"]
    edges = np.linspace(train_start_idx, train_end_idx + 1, k + 2, dtype=int)
    folds = []
    for i in range(1, k + 1):
        val_lo, val_hi = int(edges[i]), int(edges[i + 1])
        val = [j for j, t0 in enumerate(t0s) if val_lo <= t0 < val_hi]
        cutoff = val_lo - emb
        tr = [j for j, t0 in enumerate(t0s) if t0 + H < cutoff]
        if len(val) >= 5 and len(tr) >= 10:
            folds.append((tr, val, val_lo))
    return folds


# ============================ the trade engine (port) ============================

def run_engine(df, scored, start_idx, end_idx, threshold, kelly_fraction=None):
    """Model-agnostic capital simulation over (event, p) pairs — a verbatim port of the
    parent project's engine (one open position, threshold gate, per-trade fractional Kelly,
    fees+slippage both sides, mark-to-close equity, capital-depletion halt)."""
    E0 = CONFIG["INITIAL_CAPITAL_USD"]
    fee = CONFIG["COMMISSION_BPS"] * 1e-4
    slip = CONFIG["SLIPPAGE_BPS"] * 1e-4
    kelly_cap = CONFIG["KELLY_CAP"]
    # Barrier reward:risk b = TP/SL width ratio → generalized fractional Kelly f = λ·(p − (1−p)/b).
    # For a symmetric barrier (b=1) this is exactly λ·(2p−1), the original even-money sizing.
    b_payoff = float(CONFIG["TB_ATR_TP"]) / float(CONFIG["TB_ATR_SL"])
    tie_eps = CONFIG["SIMULTANEOUS_SETUP_TIE_EPS"]
    c = df["close"].to_numpy()
    groups = {}
    for ev, p in scored:
        if start_idx <= ev["t0"] <= end_idx:
            groups.setdefault(ev["t0"], []).append((ev, p))
    counters = dict(signals_total=sum(len(v) for v in groups.values()), threshold_rejects=0, not_selected=0,
                    simultaneous_tie_skip=0, gap_invalidated_skip=0,
                    invalid_barrier_skip=0, ignored_while_open=0, entered=0)
    E = E0
    equity_events = [{"event_type": "initial_capital", "bar_index": -1, "trade_id": 0, "equity": E0}]
    ledger, exposure_bars, flat_from, halted, tid = [], 0, start_idx, False, 0
    for t0 in sorted(groups):
        if halted:
            break
        if t0 < flat_from:
            counters["ignored_while_open"] += len(groups[t0])
            continue
        cands = sorted(groups[t0], key=lambda x: -x[1])
        passing = [(ev, p) for ev, p in cands if p >= threshold]
        counters["threshold_rejects"] += len(cands) - len(passing)
        if not passing:
            continue
        if len(passing) >= 2 and abs(passing[0][1] - passing[1][1]) <= tie_eps:
            counters["simultaneous_tie_skip"] += len(passing)
            continue
        chosen, chosen_p = passing[0]
        counters["not_selected"] += len(passing) - 1
        sim = simulate_trade(df, chosen, end_idx)
        if sim["skip"] == "GAP_INVALIDATED_SKIP":
            counters["gap_invalidated_skip"] += 1
            continue
        if sim["skip"] == "INVALID_BARRIER_SKIP":
            counters["invalid_barrier_skip"] += 1
            continue
        s = chosen["direction"]
        entry_fill, exit_fill, exit_idx = sim["entry_fill"], sim["exit_fill"], sim["exit_idx"]
        kelly_edge = chosen_p - (1.0 - chosen_p) / b_payoff       # Kelly edge for a b:1 payoff (=2p−1 at b=1)
        f_size = 1.0 if kelly_fraction is None else min(max(kelly_fraction * kelly_edge, 0.0), kelly_cap)
        if f_size <= 0.0:
            counters["not_selected"] += 1
            continue
        q = f_size * E / (entry_fill * (1 + fee))
        entry_fee, exit_fee = q * entry_fill * fee, q * exit_fill * fee
        raw_net = s * q * (exit_fill - entry_fill) - entry_fee - exit_fee
        account_net = max(raw_net, -E)
        uncovered = max(-(E + raw_net), 0.0)
        E_before = E
        counters["entered"] += 1
        tid += 1
        equity_events.append({"event_type": "entry_fee_mark", "bar_index": int(sim["t_fill"]), "trade_id": tid,
                              "equity": max(0.0, E_before - entry_fee)})
        mark_end = exit_idx - 1 if sim["exit_kind"] == "condition" else exit_idx
        for t in range(sim["t_fill"], mark_end + 1):
            liq = c[t] * (1 - s * slip)
            equity_events.append({"event_type": "held_close_mark", "bar_index": int(t), "trade_id": tid,
                                  "equity": max(0.0, E_before + s * q * (liq - entry_fill) - entry_fee - q * liq * fee)})
        E = E + account_net
        cap_state = "ACTIVE"
        if E_before + raw_net <= 0:
            E, cap_state, halted = 0.0, "HALTED_CAPITAL_DEPLETED", True
        equity_events.append({"event_type": "exit_fill", "bar_index": int(exit_idx), "trade_id": tid, "equity": E})
        exposure_bars += (exit_idx - sim["t_fill"] + 1)
        flat_from = exit_idx if sim["exit_kind"] == "condition" else exit_idx + 1
        cond = sim["exit_kind"] == "condition"
        ledger.append({"trade_id": tid, "direction": s,
                       "setup_t0_index": int(t0), "entry_fill_index": int(sim["t_fill"]),
                       "exit_trigger_index": (int(sim["trigger_idx"]) if cond else -1),
                       "exit_fill_index": int(exit_idx),
                       "decision_date": session_date(df, t0),
                       "entry_fill_date": session_date(df, sim["t_fill"]),
                       "exit_fill_date": session_date(df, exit_idx),
                       "entry_fill": entry_fill, "exit_fill": exit_fill,
                       "target_level": sim["target_level"], "stop_level": sim["stop_level"],
                       "barrier_width_pct": 100.0 * sim["barrier_width"] / max(EPS, abs(entry_fill)),
                       "model_prob": float(chosen_p), "kelly_fraction_applied": float(f_size),
                       "quantity": q, "market_exit_reason": sim["market_exit_reason"], "capital_state": cap_state,
                       "capital_before": E_before, "raw_net_pnl_usd": raw_net, "account_net_pnl_usd": account_net,
                       "uncovered_loss_usd": uncovered, "capital_after": E})
    total_bars = end_idx - start_idx + 1
    nets = np.array([t["account_net_pnl_usd"] for t in ledger]) if ledger else np.array([])
    gp = float(nets[nets > 0].sum()) if len(nets) else 0.0
    gl = float(-nets[nets < 0].sum()) if len(nets) else 0.0
    pf = None if (not len(nets) or gl == 0) else gp / gl
    eq = np.array([ev["equity"] for ev in equity_events])
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max((peak - eq) / np.maximum(EPS, peak)) * 100) if len(eq) > 1 else 0.0
    wins, losses = int((nets > 0).sum()), int((nets < 0).sum())
    summary = {"start_capital": E0, "end_capital": float(E), "net_pnl_usd": float(E - E0),
               "return_pct": float((E / E0 - 1) * 100), "profit_factor": pf, "max_drawdown_pct": mdd,
               "win_rate_pct": (wins / len(nets) * 100) if len(nets) else 0.0, "trades": len(ledger),
               "model_trades": len(ledger), "benchmark_trades": 0,
               "wins": wins, "losses": losses,
               "time_in_market_pct": round(100.0 * exposure_bars / max(1, total_bars), 4),
               "forced_oos_exits": int(sum(1 for t in ledger if t["market_exit_reason"] == "OOS_END_FORCED_EXIT")),
               "capital_depleted": bool(halted),
               "uncovered_loss_total_usd": float(sum(t["uncovered_loss_usd"] for t in ledger)),
               "max_uncovered_loss_usd": float(max((t["uncovered_loss_usd"] for t in ledger), default=0.0)),
               **counters}
    return summary, ledger, equity_events


def hodl_fallback(df, start_idx, end_idx):
    """The HODL-fallback EXECUTION MODE (#36-40) — when the model produced ZERO trades the
    executed OOS path is one long buy-and-hold benchmark trade with the same fill/cost
    model. The benchmark trade is NEVER counted as a model trade: trades == model_trades
    == 0, benchmark_trades == 1, wins/losses/win_rate are model stats (all zero) and
    profit_factor is None (0 model trades — unrankable, #41). return_pct/end_capital
    remain the EXECUTED result of the hold."""
    E0 = CONFIG["INITIAL_CAPITAL_USD"]
    fee = CONFIG["COMMISSION_BPS"] * 1e-4
    slip = CONFIG["SLIPPAGE_BPS"] * 1e-4
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    entry_fill = float(o[start_idx] * (1 + slip))
    exit_fill = float(c[end_idx] * (1 - slip))
    q = E0 / (entry_fill * (1 + fee))
    entry_fee, exit_fee = q * entry_fill * fee, q * exit_fill * fee
    net = q * (exit_fill - entry_fill) - entry_fee - exit_fee
    E = E0 + net
    equity_events = [{"event_type": "initial_capital", "bar_index": -1, "trade_id": 0, "equity": E0},
                     {"event_type": "entry_fee_mark", "bar_index": int(start_idx), "trade_id": 1,
                      "equity": max(0.0, E0 - entry_fee)}]
    for t in range(start_idx, end_idx + 1):
        liq = c[t] * (1 - slip)
        equity_events.append({"event_type": "held_close_mark", "bar_index": int(t), "trade_id": 1,
                              "equity": max(0.0, E0 + q * (liq - entry_fill) - entry_fee - q * liq * fee)})
    equity_events.append({"event_type": "exit_fill", "bar_index": int(end_idx), "trade_id": 1, "equity": float(E)})
    nan = float("nan")
    ledger = [{"trade_id": 1, "direction": 1,
               "setup_t0_index": int(start_idx), "entry_fill_index": int(start_idx),
               "exit_trigger_index": -1, "exit_fill_index": int(end_idx),
               "decision_date": session_date(df, start_idx),
               "entry_fill_date": session_date(df, start_idx),
               "exit_fill_date": session_date(df, end_idx),
               "entry_fill": entry_fill, "exit_fill": exit_fill,
               "target_level": nan, "stop_level": nan, "barrier_width_pct": nan,
               "model_prob": nan, "kelly_fraction_applied": 1.0,
               "quantity": float(q), "market_exit_reason": "HODL_FALLBACK_EXIT",
               "capital_state": "ACTIVE", "capital_before": E0,
               "raw_net_pnl_usd": float(net), "account_net_pnl_usd": float(net),
               "uncovered_loss_usd": 0.0, "capital_after": float(E)}]
    eq = np.array([ev["equity"] for ev in equity_events])
    peak = np.maximum.accumulate(eq)
    mdd = float(np.max((peak - eq) / np.maximum(EPS, peak)) * 100)
    summary = {"start_capital": E0, "end_capital": float(E), "net_pnl_usd": float(net),
               "return_pct": float((E / E0 - 1) * 100),
               "profit_factor": None,                      # 0 model trades -> unrankable (#41)
               "max_drawdown_pct": mdd,
               "win_rate_pct": 0.0, "trades": 0,            # trades = MODEL trades (#38)
               "model_trades": 0, "benchmark_trades": 1,
               "wins": 0, "losses": 0,
               "time_in_market_pct": 100.0, "forced_oos_exits": 0,
               "capital_depleted": bool(E <= 0),
               "uncovered_loss_total_usd": 0.0, "max_uncovered_loss_usd": 0.0,
               "signals_total": 0, "threshold_rejects": 0, "not_selected": 0,
               "simultaneous_tie_skip": 0, "gap_invalidated_skip": 0,
               "invalid_barrier_skip": 0, "ignored_while_open": 0, "entered": 0,
               "capital_mode": "hodl_fallback_no_model_trades", "hodl_fallback": True}
    return summary, ledger, equity_events


RESULT_MODES = ("ML_MULTI_TRADE", "ML_ONE_TRADE_LOW_EVIDENCE",
                "HODL_FALLBACK_NO_MODEL_TRADES", "TRAIN_OOF_FLOOR_NOT_MET")


def result_mode(model_trades, trade_floor_met=True):
    """The result taxonomy (#1-5, #36-41; mirrors xgb pipeline.result_mode). Precedence: a
    Train-OOF trade-floor failure is decided BEFORE the OOS read (OOS only reports, never
    decides ML-vs-HODL) and blocks promotion (#9). Otherwise: 0 model trades -> the
    HODL-fallback EXECUTION mode; 1 -> an ML result with low evidence (NEVER converted to
    HODL, #3); >= 2 -> multi-trade."""
    if not trade_floor_met:
        return "TRAIN_OOF_FLOOR_NOT_MET"
    n = int(model_trades)
    if n == 0:
        return "HODL_FALLBACK_NO_MODEL_TRADES"
    return "ML_ONE_TRADE_LOW_EVIDENCE" if n == 1 else "ML_MULTI_TRADE"
