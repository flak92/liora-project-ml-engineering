#!/usr/bin/env python3
"""The acceptance contract, in one place, so the null tests the procedure that actually ran.

A permutation test is only valid if each permutation reproduces the whole act of deciding. If the
real run and the null compute their statistic with two different pieces of code, the test measures
the difference between the two implementations as much as the effect. So the rule lives here and
both callers import it: `crossfit_selection.py` for the real run, `procedure_null.py` for every
permutation.

**The unit of the decision is the outer fold, not the rotation.** A rotation-level statistic
controls only "45 candidates, one maximum, one rotation". The procedure being defended is larger:
four rotations, a candidate must recur across them, its confirmation deltas must be positive in a
majority, and their median must clear the complexity charge. The null has to reproduce all of it,
which is why `T()` collapses four rotations into one scalar.

    selection_count_j   how many rotations picked j          (flat: the feature; hierarchical: the family)
    win_rate_j          share of its comparable confirmation deltas that are positive
    median_j            median of those deltas
    eligible_j          selection_count_j >= 2 AND win_rate_j > 0.5
    score_j             median_j - COMPLEXITY_PEN
    T                   max(score_j over eligible j), or 0.0 when nothing is eligible

`T > 0` is exactly the old acceptance: `median > MIN_MEDIAN_DELTA` and `MIN_MEDIAN_DELTA` and
`COMPLEXITY_PEN` are the same 0.004, so no threshold moved when the rule was rewritten.

**What changed and why.** The previous `accept()` examined only the modal pick
(`Counter(chosen).most_common(1)`) and ignored every other unit, however strong. That is arbitrary
in a way that matters: with two units each picked twice, the winner depended on Counter's insertion
order, which is not a reproducible property across Python versions. Worse, a null cannot honestly
reproduce "whichever one Counter happened to return first". Taking the maximum over everything
eligible is what a search actually does, it is reproducible, and it is the same rule on both sides
of the test. It can change which unit a fold reports, so the 3B headline counts are recomputed
rather than carried over.
"""
import statistics as st

COMPLEXITY_PEN = 0.004             # the same charge the search has levied since stage 2
MIN_ROTATIONS = 2                  # rotations overlap in their discovery sets: "at least 2", no more
MAJORITY = 0.5                     # strictly more than half


def unit_key(arm):
    """Flat is judged on the feature, hierarchical on the family: the same OHLCV relationship
    recurring across rotations counts even when a different variant of it wins each time."""
    return "picked" if arm == "flat" else "family"


def unit_stats(rotations, arm):
    """Per-unit evidence gathered across the rotations of ONE outer fold."""
    key = unit_key(arm)
    by = {}
    for r in rotations:
        if not r or arm not in r.get("arms", {}):
            continue
        a = r["arms"][arm]
        if a.get("confirm_delta") is None:                 # not comparable, contributes nothing
            continue
        u = by.setdefault(a[key], {"unit": a[key], "deltas": [], "representatives": set()})
        u["deltas"].append(float(a["confirm_delta"]))
        u["representatives"].add(a["picked"])

    out = {}
    for u, d in by.items():
        n = len(d["deltas"])
        wins = sum(1 for x in d["deltas"] if x > 0)
        med = st.median(d["deltas"])
        out[u] = {"unit": u, "selection_count": n, "deltas": d["deltas"],
                  "win_rate": wins / n, "wins": wins, "median_delta": med,
                  "score": med - COMPLEXITY_PEN,
                  "eligible": bool(n >= MIN_ROTATIONS and wins / n > MAJORITY),
                  "representatives": sorted(d["representatives"])}
    return out


def T(rotations, arm):
    """The procedure-level statistic for one outer fold and one arm.

    Zero when nothing is eligible — the honest encoding of "the procedure accepted nothing", which
    is a valid and complete outcome rather than a missing value.
    """
    stats = unit_stats(rotations, arm)
    elig = [s for s in stats.values() if s["eligible"]]
    if not elig:
        return 0.0, None
    best = max(elig, key=lambda s: s["score"])
    return float(best["score"]), best


def verdict(rotations):
    """Both arms of one outer fold, in the shape the artifacts and the chain's gates expect."""
    out = {}
    for arm in ("flat", "hierarchical"):
        stats = unit_stats(rotations, arm)
        t, best = T(rotations, arm)
        if best is None:
            why = ("no unit was picked in at least 2 rotations with a majority of positive "
                   "confirmations" if stats else "no comparable confirmation")
            out[arm] = {"T": t, "accepted": False, "unit": None, "reason": why,
                        "units_examined": len(stats)}
            continue
        out[arm] = {"T": round(t, 8), "accepted": bool(t > 0), "unit": best["unit"],
                    "rotations": best["selection_count"], "median_delta": round(best["median_delta"], 8),
                    "wins": best["wins"], "n_deltas": best["selection_count"],
                    "representatives": best["representatives"],
                    "units_examined": len(stats),
                    "units_eligible": sum(1 for s in stats.values() if s["eligible"]),
                    "reason": ("all clauses hold" if t > 0 else
                               f"median {best['median_delta']:+.4f} does not clear the "
                               f"complexity charge {COMPLEXITY_PEN}")}
    return out
