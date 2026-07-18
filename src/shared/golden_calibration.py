#!/usr/bin/env python3
"""Golden-calibration search policy — shared, pure helpers for BOTH feature-search workers
(docs/METHODOLOGY.md). This module changes ONLY search policy: which
candidates are grouped/ordered/represented. Every acceptance gate (min gains, fold-win, trade
floor, complexity) stays in the workers, untouched.

Vocabulary (§2): a family's tested variants form its *spectrum*; the variants statistically
indistinguishable from the family best (within one standard error of the best's per-fold deltas)
form the *one-SE plateau*; the *representative* is the SIMPLEST member of that plateau —
`best point != golden calibration`.

All functions are deterministic and side-effect-free.
"""
import fcntl
import json
import math
import os
from pathlib import Path


def append_jsonl(path, record):
    """flock-serialized JSONL append (parallel search workers write the round report safely)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_families(path, candidate_ids):
    """{feature_id: family_name} validated exactly-once against the live candidate pool.
    Fail-closed: an unmapped candidate or an unknown/duplicate mapped id raises — the map is part
    of the recipe identity and must stay in lockstep with the pool."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    fam_of = {}
    for fam, ids in doc["families"].items():
        for i in ids:
            if i in fam_of:
                raise ValueError(f"feature id {i} mapped twice ({fam_of[i]!r} and {fam!r})")
            fam_of[int(i)] = fam
    cands = set(int(c) for c in candidate_ids)
    missing, extra = sorted(cands - set(fam_of)), sorted(set(fam_of) - cands)
    if missing or extra:
        raise ValueError(f"family map out of sync with pool: unmapped={missing} stale={extra}")
    return fam_of


def one_se_set(variants):
    """§2/§4: the one-SE plateau of a family. `variants` = [{'id', 'mean', 'folds': [per-fold
    deltas vs baseline]}, ...] (survivors of the existing marginal gates only). SE is the standard
    error of the BEST variant's per-fold deltas (SAMPLE std, ddof=1, / √k — #34); the plateau =
    every variant whose mean is within one SE of the best mean. Returns (plateau_ids, best_id, se)."""
    if not variants:
        return [], None, 0.0
    best = max(variants, key=lambda v: v["mean"])
    folds = [d for d in best["folds"] if d is not None]
    k = len(folds)
    se = (float(_std(folds)) / math.sqrt(k)) if k > 1 else 0.0
    plateau = [v["id"] for v in variants if v["mean"] >= best["mean"] - se]
    return sorted(plateau), best["id"], se


def _std(xs):
    """Sample standard deviation (ddof=1) — the unbiased one-SE estimator (#34); <2 points -> 0."""
    k = len(xs)
    if k < 2:
        return 0.0
    m = sum(xs) / k
    return (sum((x - m) ** 2 for x in xs) / (k - 1)) ** 0.5


def complexity_score(formula, fid=0):
    """#35 — multi-dimensional simplicity for pick_representative, replacing bare formula
    length: a variant is simpler when it needs LESS HISTORY (smaller max window/lag literal),
    FEWER OPERATORS, and a SHORTER causal definition; id breaks the final tie toward the
    earlier-registered variant. Purely lexical (no data) => deterministic across runs;
    covered by the golden-family-v2 policy tag in both recipe identities."""
    import re
    s = str(formula or "")
    nums = [int(x) for x in re.findall(r"\d+", s)]
    ops = sum(s.count(c) for c in "+-*/(),")
    return (max(nums) if nums else 0, ops, len(s), int(fid))


def pick_representative(plateau_ids, simplicity):
    """§4 step 3: the simplest member of the plateau — deterministic: lowest
    (simplicity_score, id). `simplicity` maps id -> score (formula length here: a shorter causal
    definition is the simpler configuration; id breaks ties toward the earlier/registered variant)."""
    if not plateau_ids:
        return None
    return min(plateau_ids, key=lambda i: (simplicity.get(i, 1 << 30), i))


def order_pool(survivor_stats, fam_of, simplicity):
    """§4 step 4: the greedy pool order under the golden policy. `survivor_stats` =
    [{'id','mean','folds'}] (post-gate survivors). Per family: one-SE plateau -> simplest
    representative. Pool = representatives ordered by their family's best mean (desc), then the
    remaining survivors (desc by mean) — those enter greedy last and, by the existing
    marginal-vs-current-subset rule, only if they still add value beyond their family's
    representative. Returns (ordered_ids, families_report)."""
    by_fam = {}
    for v in survivor_stats:
        by_fam.setdefault(fam_of[v["id"]], []).append(v)
    reps, report = [], {}
    for fam, variants in by_fam.items():
        plateau, best_id, se = one_se_set(variants)
        rep = pick_representative(plateau, simplicity)
        best_mean = max(v["mean"] for v in variants)
        reps.append((best_mean, rep))
        report[fam] = {"survivors": sorted(v["id"] for v in variants),
                       "one_se": plateau, "best": best_id, "rep": rep,
                       "se": round(se, 6), "best_mean": round(best_mean, 6)}
    reps.sort(key=lambda t: -t[0])
    rep_ids = [r for _, r in reps]
    rest = sorted((v for v in survivor_stats if v["id"] not in set(rep_ids)),
                  key=lambda v: -v["mean"])
    return rep_ids + [v["id"] for v in rest], report


def family_stage_pool(survivor_stats, fam_of, simplicity):
    """§4 under golden-family-v2 (TRUE two-stage family-first, #20-21). Returns
    (rep_ids, variants_by_family, families_report):
      rep_ids            — ONE representative per family (one-SE plateau -> simplest),
                           ordered by family strength desc; stage-1 greedy sees ONLY these;
      variants_by_family — family -> its remaining survivors (desc by mean); stage 2 probes
                           them ONLY for families whose representative was ACCEPTED (marginal
                           hypothesis vs the current subset, capped per family).
    Survivors of non-accepted families never enter the pool — near-duplicates are spectrum
    points, not alpha sources, and skipping them is the compute saving family-first promises."""
    by_fam = {}
    for v in survivor_stats:
        by_fam.setdefault(fam_of[v["id"]], []).append(v)
    reps, variants, report = [], {}, {}
    for fam, vs in by_fam.items():
        plateau, best_id, se = one_se_set(vs)
        rep = pick_representative(plateau, simplicity)
        best_mean = max(v["mean"] for v in vs)
        reps.append((best_mean, rep, fam))
        variants[fam] = [v["id"] for v in sorted(vs, key=lambda x: -x["mean"]) if v["id"] != rep]
        report[fam] = {"survivors": sorted(v["id"] for v in vs),
                       "one_se": plateau, "best": best_id, "rep": rep,
                       "se": round(se, 6), "best_mean": round(best_mean, 6)}
    reps.sort(key=lambda t: (-t[0], t[1]))
    return [r for _, r, _ in reps], variants, report


def round_report(*, ticker, model, recipe_hash, baseline_score, baseline_trades,
                 families_report, selected, final_gain, verdict, stop_reason):
    """§10 minimal per-round record. The wording is deliberate: this is `best_observed` for the
    exact Train scope and recipe hash — never a global optimum."""
    return {"ticker": ticker, "model": model, "recipe_hash": recipe_hash,
            "baseline_score": baseline_score, "baseline_trades": baseline_trades,
            "families": families_report, "selected": selected,
            "final_gain": final_gain, "verdict": verdict, "stop_reason": stop_reason,
            "claim": "best_observed for this Train scope and recipe hash"}
