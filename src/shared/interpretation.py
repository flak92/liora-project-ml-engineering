"""Feature VALUE interpretation — pure math, no I/O (SSOT like op_select.py).

Read-side description of the SEALED models on their own Train window:
- XGB: the axis of every feature is cut at the model's own split thresholds (the
  only points where the decision function can change along that feature) and
  merged into SEGMENTS; ALL segments with enough support are recorded — also the
  ones where the model does NOT enter. `candidate_entry_region` is a highlight
  flag, never a storage filter (no cherry-picking of the Train distribution).
- LSTM: no thresholds (weights, not trees) — trajectory bands + t0 quantiles +
  deterministic occlusion live in the drivers; this module provides the shared
  numeric helpers and the provenance identity (interpretation_recipe_hash).

Interpretation semantics: "model-derived conditional ENTRY regions" — a
projection of WHOLE-MODEL behaviour onto one feature, never standalone causal
rules. TRAIN-DERIVED INTERPRETATION / NOT AN OOS RESULT / NOT A LIVE TRADING SIGNAL.

Hard limits (work order): no training, no OOS rows, no writes into decision
artifacts. Frozen parameters below are part of interpretation_recipe_hash.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "feature_ranges.v2"
EXTRACTOR_VERSION = "1.0.0"
ALGORITHM = "model_grid_bins.segments.v2"

# Frozen BEFORE the full universe run (work order §parameters; never tuned to charts):
MIN_INTERVAL_ROWS = 20          # segment recorded when it holds >= this many TRAIN rows
MIN_ENTRY_EVENTS = 10           # candidate_entry_region needs >= this many ENTRY rows
LOW_EVIDENCE_ENTRY_EVENTS = 10  # below: low_evidence flag on tp/occlusion stats
LIFT_HIGHLIGHT = 1.25           # highlight threshold ONLY (candidate_entry_region), never storage
ROUNDING_SIG = 5                # rates/lifts/quantiles/trajectories; NEVER segment bounds/thresholds

PARAMETER_CONTRACT = {
    "min_interval_rows": MIN_INTERVAL_ROWS,
    "min_entry_events": MIN_ENTRY_EVENTS,
    "low_evidence_entry_events": LOW_EVIDENCE_ENTRY_EVENTS,
    "lift_highlight": LIFT_HIGHLIGHT,
    "rounding_sig": ROUNDING_SIG,
}

DEFAULT_QS = (0.05, 0.25, 0.50, 0.75, 0.95)


# ---------------------------------------------------------------- split harvest

def harvest_splits(trees_df):
    """booster.trees_to_dataframe() -> {feature: {thresholds, split_count, total_gain}}.
    total_gain = SUM of the gains of every split of the feature (loss reduction it
    accounts for) — never a mean per split."""
    out = {}
    nodes = trees_df[trees_df["Feature"] != "Leaf"]
    for feat, grp in nodes.groupby("Feature"):
        thresholds = sorted({float(s) for s in grp["Split"].tolist() if math.isfinite(float(s))})
        out[str(feat)] = {"thresholds": thresholds,
                          "split_count": int(len(grp)),
                          "total_gain": float(grp["Gain"].sum())}
    return out


def total_gain_share(splits):
    """{feature: total_gain / sum(total_gain)}; sums to 1.0 when any split exists.
    Presentation name: 'XGB split total-gain contribution'."""
    total = sum(v["total_gain"] for v in splits.values())
    if total <= 0:
        return {k: 0.0 for k in splits}
    return {k: v["total_gain"] / total for k, v in splits.items()}


def model_grid_edges(thresholds):
    """[-inf, t1..tk, +inf] — the model-native bin grid."""
    return [-math.inf] + [float(t) for t in thresholds] + [math.inf]


# ---------------------------------------------------------------- segmentation

def _bin_stats(values, entry_mask, edges, y=None):
    """float32 digitize (XGBoost casts inputs to float32 in DMatrix — casting both
    sides removes the half-ulp routing mismatch). Returns per-bin columnar stats."""
    values = np.asarray(values, float)
    entry_mask = np.asarray(entry_mask, bool)
    finite = np.isfinite(values)
    v, e = values[finite], entry_mask[finite]
    yv = np.asarray(y, int)[finite] if y is not None else None
    inner = np.asarray(edges[1:-1], np.float32)
    bin_idx = np.digitize(v.astype(np.float32), inner, right=False)
    nb = len(inner) + 1
    per_bin = {"lo": [float(edges[b]) for b in range(nb)],
               "hi": [float(edges[b + 1]) for b in range(nb)],
               "n": [], "n_entry": [], "entry_rate": [], "lift": []}
    base_rate = (e.sum() / v.size) if v.size else 0.0
    masks = []
    for b in range(nb):
        m = bin_idx == b
        n, ne = int(m.sum()), int(e[m].sum())
        rate = (ne / n) if n else 0.0
        per_bin["n"].append(n)
        per_bin["n_entry"].append(ne)
        per_bin["entry_rate"].append(rate)
        per_bin["lift"].append((rate / base_rate) if base_rate > 0 else 0.0)
        masks.append(m)
    return v, e, yv, per_bin, masks, base_rate, int((~finite).sum())


def entry_segments(values, entry_mask, edges, y=None,
                   min_interval_rows=MIN_INTERVAL_ROWS,
                   min_entry_events=MIN_ENTRY_EVENTS,
                   low_evidence_entry_events=LOW_EVIDENCE_ENTRY_EVENTS,
                   lift_highlight=LIFT_HIGHLIGHT):
    """Cut the feature axis at the model's split thresholds, classify each bin
    (hot: entry-rate lift >= lift_highlight / cold / empty), merge adjacent bins of
    the same class (empty bins bridge into the running segment), and record ALL
    segments with n >= min_interval_rows — the full Train distribution, not only
    the favourable fragments. Optional y (TripleBarrier Y_outcome, train labels):
    adds tp_before_sl_rate over ENTRY rows per segment + lift vs the direction
    baseline mean(y) over all candidate rows."""
    v, e, yv, per_bin, masks, base_rate, n_nan = _bin_stats(values, entry_mask, edges, y)
    n_all, n_entry_all = int(v.size), int(e.sum())
    tp_baseline = float(yv.mean()) if (yv is not None and yv.size) else None

    def classify(b):
        if per_bin["n"][b] == 0:
            return "empty"
        return "hot" if (base_rate > 0 and per_bin["lift"][b] >= lift_highlight) else "cold"

    runs, cur = [], None
    for b in range(len(masks)):
        cls = classify(b)
        if cls == "empty":
            if cur is not None:
                cur["hi_bin"] = b                      # bridge: extend span, no counts
            continue
        if cur is not None and cur["class"] == cls:
            cur["hi_bin"], cur["bins"] = b, cur["bins"] + [b]
        else:
            if cur is not None:
                runs.append(cur)
            cur = {"class": cls, "lo_bin": b, "hi_bin": b, "bins": [b]}
    if cur is not None:
        runs.append(cur)

    segments = []
    for r in runs:
        m = np.zeros(v.size, bool)
        for b in r["bins"]:
            m |= masks[b]
        n = int(m.sum())
        if n < min_interval_rows:
            continue
        ne = int(e[m].sum())
        entry_share = ne / n
        entry_lift = (entry_share / base_rate) if base_rate > 0 else 0.0
        seg = {"lo": per_bin["lo"][r["lo_bin"]], "hi": per_bin["hi"][r["hi_bin"]],
               "n_rows": n, "n_entry_events": ne,
               "entry_share": round_sig(entry_share), "entry_lift": round_sig(entry_lift),
               "candidate_entry_region": bool(entry_lift >= lift_highlight and ne >= min_entry_events),
               "low_evidence": bool(ne < low_evidence_entry_events)}
        if yv is not None:
            ent_y = yv[m & e]
            tp_rate = float(ent_y.mean()) if ent_y.size else None
            seg["tp_events"] = int(ent_y.sum()) if ent_y.size else 0
            seg["tp_before_sl_rate"] = round_sig(tp_rate) if tp_rate is not None else None
            seg["lift_vs_train_baseline"] = (round_sig(tp_rate / tp_baseline)
                                             if (tp_rate is not None and tp_baseline) else None)
        segments.append(seg)

    for k in ("entry_rate", "lift"):
        per_bin[k] = [round_sig(x) for x in per_bin[k]]
    return {"base_entry_rate": round_sig(base_rate), "tp_baseline": round_sig(tp_baseline) if tp_baseline is not None else None,
            "n": n_all, "n_entry": n_entry_all, "n_nan": n_nan,
            "per_bin": per_bin, "segments": segments}


# ---------------------------------------------------------------- shared helpers

def quantiles(values, mask=None, qs=DEFAULT_QS):
    """Quantiles of values[mask] (NaN skipped); None when empty. Key "n" carries the
    support — a consumer MUST show it (n=1 quantiles are a point, not a distribution)."""
    v = np.asarray(values, float)
    if mask is not None:
        v = v[np.asarray(mask, bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    out = {f"q{int(q * 100):02d}": round_sig(float(np.quantile(v, q))) for q in qs}
    out["n"] = int(v.size)
    return out


def round_sig(x, sig=ROUNDING_SIG):
    if not isinstance(x, float) or x == 0.0 or not math.isfinite(x):
        return x
    return float(f"%.{sig}g" % x)


def sigma_bounds(lo, hi, mu, sigma):
    """(bound - mu)/sigma for finite bounds; None for unbounded or sigma<=0."""
    if mu is None or sigma is None or not sigma or sigma <= 0:
        return None, None
    z = lambda b: round_sig((b - mu) / sigma) if math.isfinite(b) else None
    return z(lo), z(hi)


def trajectory_bands(windows, entry_mask, feature_names):
    """windows: (n, seq_len, f) RAW; per feature mean/std trajectories for ENTRY
    windows vs all candidates. NaN-safe; rounded to ROUNDING_SIG."""
    import warnings
    w = np.asarray(windows, float)
    e = np.asarray(entry_mask, bool)
    out = {}

    def band(arr):
        return [round_sig(float(x)) for x in np.asarray(arr)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j, name in enumerate(feature_names):
            base = w[:, :, j]
            ent = base[e]
            out[str(name)] = {
                "base_mean": band(np.nanmean(base, axis=0)),
                "base_std": band(np.nanstd(base, axis=0)),
                "entry_mean": band(np.nanmean(ent, axis=0)) if ent.size else None,
                "entry_std": band(np.nanstd(ent, axis=0)) if ent.size else None,
                "n_base": int(base.shape[0]),
                "n_entry": int(ent.shape[0]),
            }
    return out


def interval_overlap(a, b):
    """Scale-invariant Jaccard of two segment lists [{lo,hi},...] (stability
    hook for future cross-run diffs; currently unused)."""
    def spans(iv):
        return [(float(x["lo"]), float(x["hi"])) for x in iv]

    sa, sb = spans(a), spans(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    finite = [x for lo, hi in sa + sb for x in (lo, hi) if math.isfinite(x)]
    if not finite:
        return 1.0
    lo_h, hi_h = min(finite), max(finite)
    span = hi_h - lo_h
    scale = max(span, abs(lo_h), abs(hi_h), 1e-12)
    lo_h, hi_h = lo_h - 0.5 * scale, hi_h + 0.5 * scale

    def measure(s):
        s = sorted((max(lo, lo_h), min(hi, hi_h)) for lo, hi in s)
        merged = []
        for lo, hi in s:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        return merged, sum(hi - lo for lo, hi in merged)

    ma, la = measure(sa)
    mb, lb = measure(sb)
    inter = sum(max(0.0, min(ahi, bhi) - max(alo, blo)) for alo, ahi in ma for blo, bhi in mb)
    union = la + lb - inter
    return inter / union if union > 0 else 1.0


# ---------------------------------------------------------------- determinism + identity

def sanitize(obj):
    """inf -> "inf", -inf -> "-inf", NaN -> None, -0.0 -> 0.0, numpy -> native."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [sanitize(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.floating,)):
        return sanitize(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj + 0.0 if obj == 0.0 else obj
    return obj


def canonical_json(payload):
    return json.dumps(sanitize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)


def payload_sha256(payload):
    """Hash of the canonical JSON — no timestamps in payloads => byte-identical re-runs."""
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def interpretation_recipe_hash():
    """Identity of the METHODOLOGY (not just the file): segmentation algorithm +
    frozen thresholds + sigma methods + LSTM window/occlusion definition +
    contribution normalization + rounding policy + this module's source hash."""
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "parameter_contract": PARAMETER_CONTRACT,
        "methods": {
            "xgb_sigma": "descriptive_standardization_train_candidate_rows",
            "lstm_sigma": "artifact_norm_stats_model_input_transform",
            "xgb_contribution": "split_total_gain_share",
            "lstm_contribution": "deterministic_occlusion_train_mean_entry_conditioned_primary",
            "lstm_occlusion_sample": "one_frozen_set_of_all_purged_train_windows_fixed_order",
            "tp_rate": "share_of_Y1_among_entry_rows_in_segment",
            "rounding": {"sig": ROUNDING_SIG, "exempt": ["segment_bounds", "split_thresholds"]},
        },
        "source_sha256": source_sha256(),
    }
    return hashlib.sha256(canonical_json(recipe).encode()).hexdigest()
