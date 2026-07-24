#!/usr/bin/env python3
"""Rung 8 — Family Transfer Across Assets. A pure, deterministic panel-level aggregator.

    frozen artifacts + family registry + frozen contract  ->  family_transfer.json

This script NEVER trains, NEVER opens the bar store (liora.duckdb), NEVER reads OOS, and NEVER imports
a model-training module. It reads only the frozen per-asset result artifacts and the static config
maps, and asks one panel-level question: *which OHLCV feature families carry stable information across
assets?* The scientific proof standard (acceptance statistic, the three nulls, alpha, the OOS boundary,
the viability floor) lives in config/contract/* and is untouched here — this is a reporting layer.

Two sources, one code path (mirrors engine/report.py):

    python3 scripts/family_transfer.py --snapshot
        -> results/methodology_snapshot/family_transfer.json
    python3 scripts/family_transfer.py --run-dir runs/<id>
        -> runs/<id>/results/panels/family_transfer.json
    python3 scripts/family_transfer.py --snapshot --panel config/panel_6.json --out /tmp/ft.json

The global funnel counters reuse engine/report.py's exact primitives (`_accepted_arms`, and the
canonical rung5_verdict.passed_arms), so the aggregator and the funnel share one definition of who
survived — snapshot parity 26 -> 11 -> 9 -> 2 is a property of that shared reader, not a constant.

Serialization is canonical: UTF-8, sort_keys, stable list order, NO timestamps in the hashed science
artifact. The same inputs produce a byte-identical file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rung5_verdict as RV                                                    # noqa: E402  (science layer)

SNAPSHOT = ROOT / "results" / "methodology_snapshot"
SCHEMA_VERSION = "family_transfer.v1"
SEED = 42                                                                     # the lab seed (config/contract)

FAMILIES_PATH = ROOT / "config" / "feature_families_xgb.json"
REGISTRY_DIR = ROOT / "config" / "feature_registries"
CONTRACT_PATH = ROOT / "config" / "feature_discovery_contract.json"
REPORTING_PATH = ROOT / "config" / "family_transfer_reporting.json"

# The seven per-asset science artifacts. feature_utility is OPTIONAL: it lives in run panels but is not
# shipped in the frozen snapshot, so its absence degrades single-utility metrics to null (reported in
# integrity.missing_inputs) rather than crashing.
REQUIRED = ["crossfit_selection.json", "procedure_null_a1.json", "procedure_null_a2.json",
            "procedure_null_b.json", "rung6_survivor_hpo.json"]
OPTIONAL = ["feature_utility.json", "cross_asset_matrix.json"]


# ── strict readers (missing -> reported; corrupt -> STOP) ───────────────────────────────────────────
class IntegrityStop(RuntimeError):
    """A corrupt or self-contradictory input halts the report instead of being silently skipped."""


def _read_strict(path, integrity, required):
    p = Path(path)
    if not p.exists():
        (integrity["missing_inputs"]).append(p.name)
        if required:
            integrity["status"] = "MISSING_REQUIRED"
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise IntegrityStop("corrupt artifact %s: %s" % (p.name, e))


def _read_config(path):
    p = Path(path)
    if not p.exists():
        raise IntegrityStop("missing required config %s" % p.name)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise IntegrityStop("corrupt config %s: %s" % (p.name, e))


# ── config maps (pure config; no private tree, no duckdb) ───────────────────────────────────────────
def load_family_map():
    fam = _read_config(FAMILIES_PATH)["families"]
    id_to_family, family_to_ids, dup = {}, {}, []
    for family, ids in fam.items():
        family_to_ids[family] = [int(i) for i in ids]
        for i in ids:
            i = int(i)
            if i in id_to_family and id_to_family[i] != family:
                dup.append({"feature_id": i, "families": sorted([id_to_family[i], family])})
            id_to_family[i] = family
    return id_to_family, family_to_ids, dup


def load_names():
    names = {}
    for p in sorted(REGISTRY_DIR.glob("*.json")):
        doc = _read_config(p)
        feats = doc.get("features") if isinstance(doc, dict) else doc
        for f in (feats or []):
            if isinstance(f, dict) and "id" in f:
                names[int(f["id"])] = f.get("name")
    return names


def load_complexity():
    """Complexity score per feature id, if the registry carries a formula; else name length as a
    stable, deterministic proxy. Only used for the minimal-panel tie-break, never for selection."""
    comp = {}
    for p in sorted(REGISTRY_DIR.glob("*.json")):
        doc = _read_config(p)
        feats = doc.get("features") if isinstance(doc, dict) else doc
        for f in (feats or []):
            if isinstance(f, dict) and "id" in f:
                formula = f.get("formula") or f.get("name") or ""
                comp[int(f["id"])] = len(str(formula))
    return comp


# ── normalization: every arm -> (ticker, fold, arm, feature_id, family, representative) ──────────────
def _arm_family(arm, unit, id_to_family):
    """flat unit is a feature id (int); hierarchical unit is a family name (str)."""
    if unit is None:
        return None, None
    s = str(unit)
    if s.isdigit():
        fid = int(s)
        return id_to_family.get(fid), fid
    return s, None                                    # hierarchical: unit already the family name


def build_representative_index(crossfit, rung6):
    """(ticker, fold, family) -> representative feature id, from the artifacts that name a rep.

    A hierarchical arm reports its family and a representative feature; a flat arm's unit already IS the
    feature. rung6 is authoritative (it carries `representative`); the crossfit fold verdict fills the
    rest. This is what lets a flat/112 arm and a hierarchical/oscillator_rsi arm collapse to one unique
    representative (rep 112) — two arms, one feature.
    """
    rep = {}
    if crossfit:
        for ticker, rec in crossfit["tables"].items():
            for f in rec.get("folds", []):
                fold = f.get("outer_fold")
                hv = (f.get("verdict", {}) or {}).get("hierarchical", {}) or {}
                fam = hv.get("unit")
                reps = hv.get("representatives") or []
                if fam and reps:
                    rep[(ticker, fold, fam)] = int(reps[0])
    if rung6:
        for r in rung6.get("results", []):
            if str(r.get("arm")) == "hierarchical" and r.get("representative") is not None:
                rep[(r.get("ticker"), r.get("outer_fold"), str(r.get("unit")))] = int(r["representative"])
    return rep


def representative_of(ticker, fold, arm, unit, rep_index):
    if unit is None:
        return None
    s = str(unit)
    if s.isdigit():
        return int(s)                                 # flat
    return rep_index.get((ticker, fold, s))           # hierarchical -> looked-up rep


# ── funnel primitives (identical semantics to engine/report.py) ─────────────────────────────────────
def accepted_arms_keys(crossfit, panel):
    """(ticker, fold, arm) cross-fit-accepted, and the family each was accepted under."""
    keys = {}
    if crossfit:
        for ticker, rec in crossfit["tables"].items():
            if panel and ticker not in panel:
                continue
            for f in rec.get("folds", []):
                fold = f.get("outer_fold")
                for arm in ("flat", "hierarchical"):
                    v = (f.get("verdict", {}) or {}).get(arm, {}) or {}
                    if v.get("accepted"):
                        fam, _ = _arm_family(arm, v.get("unit"), FAMILY_MAP)
                        keys[(ticker, fold, arm)] = fam
    return keys


def passed_keys(null_doc, panel):
    """(ticker, fold, arm, unit_str) that passed a null — via the canonical rung5_verdict.passed_arms."""
    keys = set()
    if null_doc:
        for ticker, rec in null_doc["tables"].items():
            if panel and ticker not in panel:
                continue
            for fold, arm, unit in RV.passed_arms(rec):
                keys.add((ticker, fold, arm, unit))
    return keys


def discovery_picks(crossfit, panel):
    """Per (family, arm) discovery pick counts from the rotations (before confirmation)."""
    flat, hier = {}, {}
    if crossfit:
        for ticker, rec in crossfit["tables"].items():
            if panel and ticker not in panel:
                continue
            for f in rec.get("folds", []):
                for rot in f.get("rotations", []):
                    arms = rot.get("arms", {}) or {}
                    fl = arms.get("flat") or {}
                    if fl.get("family"):
                        flat[fl["family"]] = flat.get(fl["family"], 0) + 1
                    hi = arms.get("hierarchical") or {}
                    if hi.get("family"):
                        hier[hi["family"]] = hier.get(hi["family"], 0) + 1
    return flat, hier


def confirmation_deltas(crossfit, panel):
    """Per family: the confirmation deltas of its arms across rotations (for median/IQR/sign)."""
    by_family = {}
    if crossfit:
        for ticker, rec in crossfit["tables"].items():
            if panel and ticker not in panel:
                continue
            for f in rec.get("folds", []):
                for rot in f.get("rotations", []):
                    for arm in ("flat", "hierarchical"):
                        a = (rot.get("arms", {}) or {}).get(arm) or {}
                        fam, d = a.get("family"), a.get("confirm_delta")
                        if fam is not None and d is not None:
                            by_family.setdefault(fam, []).append(float(d))
    return by_family


# ── small stats (stdlib only, deterministic) ────────────────────────────────────────────────────────
def _median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _iqr(xs):
    if len(xs) < 2:
        return None
    s = sorted(xs)

    def q(p):
        idx = p * (len(s) - 1)
        lo = int(idx)
        frac = idx - lo
        return s[lo] + frac * (s[min(lo + 1, len(s) - 1)] - s[lo])
    return q(0.75) - q(0.25)


def _sign_consistency(xs):
    if not xs:
        return None
    pos = sum(1 for x in xs if x > 0)
    neg = sum(1 for x in xs if x < 0)
    tot = pos + neg
    return None if tot == 0 else max(pos, neg) / tot


# ── the aggregation ─────────────────────────────────────────────────────────────────────────────────
FAMILY_MAP = {}   # module-level id->family, set in build()


def build(source, panel=None, source_label="snapshot"):
    global FAMILY_MAP
    integrity = {"missing_inputs": [], "unknown_feature_ids": [], "unmapped_feature_ids": [],
                 "duplicate_family_memberships": [], "status": "PASS"}

    id_to_family, family_to_ids, dup = load_family_map()
    FAMILY_MAP = id_to_family
    names = load_names()
    complexity = load_complexity()
    if dup:
        integrity["duplicate_family_memberships"] = dup
        integrity["status"] = "INTEGRITY_FAILED"

    src = Path(source)
    docs = {}
    for name in REQUIRED:
        docs[name] = _read_strict(src / name, integrity, required=True)
    for name in OPTIONAL:
        docs[name] = _read_strict(src / name, integrity, required=False)

    crossfit = docs["crossfit_selection.json"]
    a1, a2, b = docs["procedure_null_a1.json"], docs["procedure_null_a2.json"], docs["procedure_null_b.json"]
    r6 = docs["rung6_survivor_hpo.json"]
    feat_util = docs["feature_utility.json"]

    panel_set = set(panel) if panel else None
    rep_index = build_representative_index(crossfit, r6)

    # global funnel keys (report.py-identical), family-attributed
    acc = accepted_arms_keys(crossfit, panel_set)
    pa1, pa2, pb = passed_keys(a1, panel_set), passed_keys(a2, panel_set), passed_keys(b, panel_set)
    stable = pa1 & pa2 & pb
    r6_retained = {(r.get("ticker"), r.get("outer_fold"), str(r.get("arm")), str(r.get("unit")))
                   for r in (r6 or {}).get("results", []) if r.get("verdict") == "retained"
                   and (not panel_set or r.get("ticker") in panel_set)}
    r6_demoted = {(r.get("ticker"), r.get("outer_fold"), str(r.get("arm")), str(r.get("unit")))
                  for r in (r6 or {}).get("results", []) if str(r.get("verdict")).startswith("demoted")
                  and (not panel_set or r.get("ticker") in panel_set)}
    retained = stable & r6_retained

    flat_picks, hier_picks = discovery_picks(crossfit, panel_set)
    deltas = confirmation_deltas(crossfit, panel_set)

    def key_family(k):
        """(ticker, fold, arm, unit_str) -> family."""
        _, _, arm, unit = k
        fam, fid = _arm_family(arm, unit, id_to_family)
        if fam is None and str(unit).isdigit():
            integrity["unmapped_feature_ids"].append(int(unit))
        return fam

    def rep_family(k):
        """(ticker,fold,arm,unit) -> (family, representative_feature_id)."""
        t, fo, arm, unit = k
        fam = key_family(k)
        rep = representative_of(t, fo, arm, unit, rep_index)
        return fam, rep

    # bucket funnel keys by family
    def bucket(keyset):
        out = {}
        for k in keyset:
            out.setdefault(key_family(k), set()).add(k)
        return out
    acc_by_fam = {}
    for (t, fo, arm), fam in acc.items():
        acc_by_fam.setdefault(fam, set()).add((t, fo, arm))
    pa1_f, pa2_f, pb_f = bucket(pa1), bucket(pa2), bucket(pb)
    a1a2_f = bucket(pa1 & pa2)
    stable_f, retained_f, demoted_f = bucket(stable), bucket(retained), bucket(stable & r6_demoted)

    # asset universe actually present
    present = set()
    for doc in (crossfit, a1):
        if doc:
            present |= set(doc["tables"].keys())
    if crossfit:
        pass
    comp = {}
    d = src / "compiled"
    if d.is_dir():
        comp = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))}
    present |= set(comp.keys())
    if panel_set:
        present &= panel_set
        assets = sorted(panel_set)
    else:
        assets = sorted(present)

    # single-family utility (feature_utility.json — optional)
    single_pos = {}
    if feat_util:
        for ticker, rec in feat_util.get("tables", {}).items():
            if panel_set and ticker not in panel_set:
                continue
            for f in rec.get("folds", []):
                for cid, s in (f.get("singles", {}) or {}).items():
                    fam = id_to_family.get(int(cid))
                    gain = s.get("gain") if isinstance(s, dict) else s
                    if fam is not None and gain is not None and float(gain) > 0:
                        single_pos[fam] = single_pos.get(fam, 0) + 1

    families = []
    for family in sorted(family_to_ids):
        ids = family_to_ids[family]
        acc_k = acc_by_fam.get(family, set())
        a1_k, a2_k, b_k = pa1_f.get(family, set()), pa2_f.get(family, set()), pb_f.get(family, set())
        a1a2_k = a1a2_f.get(family, set())
        st_k, ret_k, dem_k = stable_f.get(family, set()), retained_f.get(family, set()), demoted_f.get(family, set())
        fam_deltas = deltas.get(family, [])
        ret_assets = sorted({k[0] for k in ret_k})
        ret_reps = sorted({rep_family(k)[1] for k in ret_k if rep_family(k)[1] is not None})
        fam_assets = sorted({k[0] for k in (acc_k | a1_k | st_k)})
        fam_folds = sorted({(k[0], k[1]) for k in (acc_k | a1_k | st_k)})

        status = classify_family(
            has_discovery=(flat_picks.get(family, 0) + hier_picks.get(family, 0)) > 0,
            has_crossfit=len(acc_k) > 0, has_a1=len(a1_k) > 0, has_a1a2=len(a1a2_k) > 0,
            has_stable=len(st_k) > 0, has_retained=len(ret_k) > 0, has_demoted=len(dem_k) > 0,
            n_retained_assets=len(ret_assets),
            feat_util_present=feat_util is not None, single_positive=single_pos.get(family, 0),
            reporting=REPORTING)

        families.append({
            "family": family,
            "feature_ids": ids,
            "feature_names": [names.get(i) for i in ids],
            "n_candidates": len(ids),
            "n_assets_evaluated": len(fam_assets),
            "n_outer_folds_evaluated": len(fam_folds),
            "asset_coverage": round(len(fam_assets) / len(assets), 6) if assets else 0.0,
            "single_positive_count": (single_pos.get(family, 0) if feat_util else None),
            "discovery_pick_count_flat": flat_picks.get(family, 0),
            "discovery_pick_count_hierarchical": hier_picks.get(family, 0),
            "crossfit_accepted_count": len(acc_k),
            "confirmation_positive_count": sum(1 for x in fam_deltas if x > 0),
            "median_confirmation_delta": _median(fam_deltas),
            "confirmation_delta_iqr": _iqr(fam_deltas),
            "confirmation_sign_consistency": _sign_consistency(fam_deltas),
            "a1_pass_count": len(a1_k),
            "a2_pass_count": len(a2_k),
            "b_pass_count": len(b_k),
            "stable_a1_a2_b_count": len(st_k),
            "rung6_retained_count": len(ret_k),
            "rung6_demoted_count": len(dem_k),
            "unique_retained_representatives": len(ret_reps),
            "retained_assets": ret_assets,
            "retained_representatives": ret_reps,
            "search_inflation_rate": _rate(len(acc_k) - len(a1_k), len(acc_k)),
            "regime_failure_rate": _rate(len(a1_k) - len(a1a2_k), len(a1_k)),
            "conditional_failure_rate": _rate(len(a1a2_k) - len(st_k), len(a1a2_k)),
            "tuning_failure_rate": _rate(len(dem_k), len(st_k)),
            "status": status,
        })

    minimal = minimal_panel_family_set(retained, retained_f, rep_index, complexity, id_to_family,
                                       deltas, family_to_ids, REPORTING)
    taxonomy = taxonomy_diagnostics(family_to_ids, crossfit, rep_index, deltas, complexity, REPORTING,
                                    panel_set)

    contract = _read_config(CONTRACT_PATH)
    report = {
        "schema_version": SCHEMA_VERSION,
        "run_identity": {
            "run_id": _run_id(src, source_label),
            "contract_hash": _contract_hash(contract),
            "contract_version": (contract.get("identity", {}) or {}).get("contract_version"),
            "seed": SEED,
            "source": source_label,
        },
        "panel": {
            "assets": assets, "n_assets": len(assets),
            "candidate_count": sum(len(v) for v in family_to_ids.values()),
            "family_count": len(family_to_ids),
        },
        "funnel": {
            "provisional_crossfit": len(acc),
            "passed_a1_marginal": len(pa1),
            "stable_a1_a2_b": len(stable),
            "retained_rung6": len(retained),
            "unique_retained_representatives": len(
                sorted({rep_family(k)[1] for k in retained if rep_family(k)[1] is not None})),
        },
        "families": families,
        "minimal_panel_family_set": minimal["minimal_panel_family_set"],
        "minimal_panel": minimal,
        "taxonomy_diagnostics": taxonomy,
        "integrity": integrity,
    }
    return report


def _rate(num, den):
    return None if not den else round(num / den, 6)


def classify_family(*, has_discovery, has_crossfit, has_a1, has_a1a2, has_stable, has_retained,
                    has_demoted, n_retained_assets, feat_util_present, single_positive, reporting):
    """Exactly one status per family, by the furthest funnel stage its arms reached."""
    if has_retained:
        min_assets = reporting["family_status"]["panel_stable_min_distinct_assets"]
        return "PANEL_STABLE" if n_retained_assets >= min_assets else "ASSET_CONDITIONAL"
    if has_stable:
        return "TUNING_DEPENDENT" if has_demoted else "ASSET_CONDITIONAL"
    if has_a1a2:
        return "CORE_CONDITIONAL_FAILURE"
    if has_a1:
        return "REGIME_DEPENDENT"
    if has_crossfit:
        return "SEARCH_INFLATED"
    if has_discovery:
        return "CROSSFIT_UNSTABLE"
    if feat_util_present and single_positive == 0:
        return "NO_POSITIVE_UTILITY"
    return "INSUFFICIENT_EVIDENCE"


def minimal_panel_family_set(retained, retained_f, rep_index, complexity, id_to_family, deltas,
                             family_to_ids, reporting):
    """Set-cover over confirmed (asset × outer_fold) units, NOT a ranking. Empty is a valid result."""
    trace = []
    # universe: retained (stable ∩ Rung-6-retained) asset-folds
    units = sorted({(k[0], k[1]) for k in retained})
    trace.append({"step": "confirmed_units", "count": len(units), "units": [f"{t}/{f}" for t, f in units]})
    if not units:
        return {"minimal_panel_family_set": [], "coverage": {
            "confirmed_units_total": 0, "confirmed_units_covered": 0, "asset_fold_coverage": 0.0},
            "representatives": [], "selection_trace": trace}

    # each retained family covers a set of asset-folds; pick a representative per family
    cover = {}
    fam_rep = {}
    for family, ks in retained_f.items():
        cover[family] = sorted({(k[0], k[1]) for k in ks})
        reps = sorted({representative_of(k[0], k[1], k[2], k[3], rep_index) for k in ks
                       if representative_of(k[0], k[1], k[2], k[3], rep_index) is not None})
        fam_rep[family] = reps[0] if reps else None
    trace.append({"step": "candidate_families", "families": sorted(cover)})

    remaining = set(units)
    chosen = []
    while remaining:
        best = None
        for family in sorted(cover):  # deterministic scan
            gain = len(set(cover[family]) & remaining)
            if gain == 0:
                continue
            comp = complexity.get(fam_rep.get(family), 0)
            sign = _sign_consistency(deltas.get(family, [])) or 0.0
            cand = (-gain, comp, -sign, family)  # more cover, less complexity, better sign, name
            if best is None or cand < best[0]:
                best = (cand, family)
        if best is None:
            break
        family = best[1]
        chosen.append(family)
        remaining -= set(cover[family])
        trace.append({"step": "pick", "family": family, "covers": [f"{t}/{f}" for t, f in cover[family]],
                      "representative": fam_rep.get(family)})

    covered = len(units) - len(remaining)
    return {
        "minimal_panel_family_set": sorted(chosen),
        "coverage": {"confirmed_units_total": len(units), "confirmed_units_covered": covered,
                     "asset_fold_coverage": round(covered / len(units), 6) if units else 0.0},
        "representatives": sorted({fam_rep[f] for f in chosen if fam_rep.get(f) is not None}),
        "selection_trace": trace,
    }


def taxonomy_diagnostics(family_to_ids, crossfit, rep_index, deltas, complexity, reporting, panel_set):
    """Is the current 12-family split right? Per-family width/consistency/heterogeneity + a suggestion."""
    # representatives chosen per family across folds (for switch rate / consistency)
    fam_reps = {f: [] for f in family_to_ids}
    if crossfit:
        for ticker, rec in crossfit["tables"].items():
            if panel_set and ticker not in panel_set:
                continue
            for f in rec.get("folds", []):
                hv = (f.get("verdict", {}) or {}).get("hierarchical", {}) or {}
                fam, reps = hv.get("unit"), hv.get("representatives") or []
                if fam in fam_reps and reps:
                    fam_reps[fam].append(int(reps[0]))
    tr = reporting["taxonomy"]
    out = []
    for family in sorted(family_to_ids):
        reps = fam_reps.get(family, [])
        d = deltas.get(family, [])
        distinct = len(set(reps))
        switch = None if len(reps) < 2 else round(distinct / len(reps), 6)
        sign = _sign_consistency(d)
        width = len(family_to_ids[family])
        heterog = (sign is not None and sign < 1.0)     # opposite-sign confirmation deltas present
        suggestion = _taxonomy_suggestion(reps, switch, width, heterog, tr)
        out.append({
            "family": family,
            "family_width": width,
            "within_family_representative_consistency": (None if not reps else round(1 - (distinct - 1) / max(1, len(reps) - 1), 6) if len(reps) > 1 else 1.0),
            "representative_switch_rate": switch,
            "family_effect_heterogeneity": (None if sign is None else round(1 - sign, 6)),
            "between_family_overlap": 0.0,               # ids are partitioned by construction (checked in integrity)
            "suggestion": suggestion,
        })
    return out


def _taxonomy_suggestion(reps, switch, width, heterog, tr):
    if len(reps) < tr["min_representatives_for_consistency"]:
        return "INSUFFICIENT_EVIDENCE"
    wide = width >= tr["family_width_review_split"]
    switchy = switch is not None and switch >= tr["representative_switch_rate_review_split"]
    if wide and (switchy or (heterog and tr["sign_heterogeneity_flags_split"])):
        return "REVIEW_SPLIT"
    return "KEEP"


# ── identity / hashing ──────────────────────────────────────────────────────────────────────────────
def _run_id(src, source_label):
    if source_label.startswith("run-dir"):
        # runs/<id>/results/panels -> <id>
        parts = src.parts
        return parts[parts.index("runs") + 1] if "runs" in parts else source_label
    return "snapshot"


def _contract_hash(contract):
    """Deterministic fingerprint of the frozen contract this report was read against."""
    payload = json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def serialize(report):
    return json.dumps(report, sort_keys=True, ensure_ascii=False, indent=1) + "\n"


# ── cli ─────────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Rung 8 Family Transfer aggregator (read-only, pure).")
    ap.add_argument("--snapshot", action="store_true")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--panel", default=None, help="json with an 'assets' list to restrict the panel")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.run_dir:
        source = Path(args.run_dir) / "results" / "panels"
        source_label = "run-dir:%s" % Path(args.run_dir).name
        default_out = source / "family_transfer.json"
    else:
        source = SNAPSHOT
        source_label = "snapshot"
        default_out = SNAPSHOT / "family_transfer.json"

    panel = None
    if args.panel:
        panel = _read_config(args.panel).get("assets")

    report = build(source, panel=panel, source_label=source_label)
    text = serialize(report)

    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    fn = report["funnel"]
    integ = report["integrity"]["status"]
    print("Rung 8 Family Transfer -> %s" % out)
    print("  panel: %d assets · %d families · %d candidates"
          % (report["panel"]["n_assets"], report["panel"]["family_count"], report["panel"]["candidate_count"]))
    print("  funnel: %d -> %d -> %d -> %d (unique rep %d)"
          % (fn["provisional_crossfit"], fn["passed_a1_marginal"], fn["stable_a1_a2_b"],
             fn["retained_rung6"], fn["unique_retained_representatives"]))
    print("  minimal panel family set: %s" % (report["minimal_panel_family_set"] or "[] (empty is valid)"))
    print("  integrity: %s%s" % (integ, ("  missing=" + ",".join(report["integrity"]["missing_inputs"]))
                                 if report["integrity"]["missing_inputs"] else ""))
    return 0 if integ in ("PASS",) else (0 if integ == "MISSING_REQUIRED" else 1)


REPORTING = None
if __name__ == "__main__":
    REPORTING = _read_config(REPORTING_PATH)
    raise SystemExit(main())
else:
    try:
        REPORTING = _read_config(REPORTING_PATH)
    except Exception:
        REPORTING = None
