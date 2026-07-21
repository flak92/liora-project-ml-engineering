#!/usr/bin/env python3
"""Does the ladder reject what it should, and find what it should? Both are required.

A methodology that always rejects is exactly as useless as one that never rejects, and neither
failure is visible from a run that produced few survivors. So two controls, and they answer
different questions.

**Negative control — the false-acceptance rate.** Not "one noise feature must fail once", which
proves nothing, but an empirical rate over many independent synthetic nulls, with a confidence
interval, compared against the level the test declares.

This one is FREE. Every permutation of the procedure-level null already IS an independent
synthetic-null experiment: under it the optional features carry no information by construction, so
`T_null > 0` means the acceptance contract accepted something when there was nothing there. The
rate of that across all permutations is the type-I error of the four-rotation acceptance rule ON
ITS OWN — the quantity that tells you how much work the max-null still has to do. Reading it off
the artifact costs nothing, and it is measured on far more replicates than a bespoke experiment
could afford.

**Positive control — power.** Plant a feature of known strength and ask whether the ladder finds
it. Reported: the minimum detectable effect, the probability of recovery at each strength, the
cost, and how power decays as the candidate pool grows. The planting is deliberately artificial —
`z = a * standardised(y) + sqrt(1-a^2) * noise` puts outcome information directly into a column at
a controlled strength. It is not a realistic feature and is not meant to be; it is a ruler.

The plant REPLACES an existing candidate's column rather than adding a new id, so the name and
family machinery is untouched and the search cannot tell it apart from a real candidate.

    python3 scripts/null_controls.py --negative
    python3 scripts/null_controls.py --positive --jobs 2
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init  # noqa: E402,F401
runtime_init.apply()
sys.path.insert(0, str(XGB / "src"))
sys.path.insert(0, str(XGB / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

import acceptance as ACC                                                   # noqa: E402
from artifact_io import read_json, write_json_atomic                       # noqa: E402

DATA = XGB / "data"
ALPHA_DECLARED = 0.10
STRENGTHS = (0.02, 0.05, 0.10, 0.20, 0.40)
POOL_SIZES = (5, 15, 45)
REPLICATES = 5
SEED = 42


def wilson(k, n, z=1.96):
    """Wilson interval — the normal approximation is wrong exactly where these rates live (near 0)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ---------------------------------------------------------------------------------------------

def negative_control(paths):
    """Type-I error of the acceptance contract, read off the permutations already computed."""
    out = {"_what": "every permutation is an independent synthetic null; T > 0 means the "
                    "acceptance contract accepted something when nothing was there",
           "declared_alpha": ALPHA_DECLARED, "by_null": {}}
    for kind, path in paths.items():
        doc = read_json(path)
        if doc is None:
            continue
        per_arm = {}
        for t in doc["tables"].values():
            for f in t["folds"]:
                for arm, v in f["arms"].items():
                    s = per_arm.setdefault(arm, {"n": 0, "accepted": 0, "T": []})
                    for x in v["null_statistics"]:
                        s["n"] += 1
                        s["T"].append(x)
                        if x > 0:
                            s["accepted"] += 1
        block = {}
        for arm, s in per_arm.items():
            lo, hi = wilson(s["accepted"], s["n"])
            ts = sorted(s["T"])
            block[arm] = {
                "replicates": s["n"],
                "false_acceptances": s["accepted"],
                "rate": round(s["accepted"] / s["n"], 4) if s["n"] else None,
                "wilson_95": [round(lo, 4), round(hi, 4)],
                "median_T_under_null": round(ts[len(ts) // 2], 6) if ts else None,
                "q90_T_under_null": round(ts[int(0.9 * (len(ts) - 1))], 6) if ts else None,
                "_reading": ("this is the rate BEFORE the max-null filters anything; it is the "
                             "multiplicity the max-null exists to remove, not a defect")}
            block[arm]["exceeds_declared_alpha"] = bool(
                block[arm]["rate"] is not None and lo > ALPHA_DECLARED)
        out["by_null"][kind] = block
    return out


# ---------------------------------------------------------------------------------------------

def plant(y, rng, strength, n):
    """A column carrying `strength` worth of outcome information and nothing else.

    Standardised so the strength is comparable across assets, and the noise term is scaled to keep
    unit variance, so a stronger plant is not also a differently-scaled column — otherwise the tree
    would be reacting to scale rather than to signal.
    """
    import numpy as np
    ys = (y - y.mean()) / (y.std() + 1e-12)
    noise = rng.normal(size=n)
    return strength * ys + math.sqrt(max(0.0, 1.0 - strength ** 2)) * noise


def positive_for(job):
    """One (ticker, outer fold, strength, pool size, replicate): is the plant recovered?"""
    ticker, ofold, strength, pool, rep = job
    import numpy as np
    import crossfit_selection as CF
    import feature_search as FS
    import golden
    import pipeline as P

    t0 = time.time()
    reg = json.loads((DATA / "feature_utility.json").read_text(encoding="utf-8"))["tables"][ticker]
    scratch = runtime_init.scratch_dir(f"poscontrol_{strength}_{pool}", ticker)
    df = P.layer4_snapshot_to_parquet(P.bars_db(), ticker, scratch / f"{ticker}_1h.parquet")
    cands_all = FS.candidate_ids()
    rec = P.derive_output_b(df, ticker, FS._manifest(ticker, cands_all))
    dfx, dfb, tev, bnds = rec["df"], rec["df_b"], rec["train_events"], rec["bounds"]
    t0s = [int(s.split(":")[1]) for s in dfb["setup_id"]]

    fold = [f for f in reg["folds"] if f["outer_fold"] == ofold][0]
    bi = dict(bnds, train_end_idx=int(fold["inner_train_end_idx"]))
    inner = P.purged_wf_folds(t0s, bi["train_start_idx"], bi["train_end_idx"])
    params = fold["frozen_params"]

    rng = np.random.default_rng(SEED + rep * 7919 + int(strength * 1000) * 31 + pool)
    cands = list(cands_all[:pool])
    planted_id = int(cands[pool // 2])
    core = set(FS._names(ticker, []))
    col = [c for c in FS._names(ticker, [planted_id]) if c not in core][0]

    y = dfb["Y_outcome"].to_numpy(float)
    dfp = dfb.copy(deep=False)
    dfp[col] = plant(y, rng, strength, len(dfb))

    fam_of = golden.load_families(FS.FAMILIES_PATH, cands_all)
    simplicity = {int(f["id"]): golden.complexity_score(f.get("formula", ""), int(f["id"]))
                  for ns, rr in P.FEATURE_REGISTRIES.items() if ns != "1h"
                  for f in rr["features"] if bool(f.get("implemented", True))}

    rots = []
    for r in range(len(inner)):
        disc = [x for k, x in enumerate(inner) if k != r]
        conf = inner[r]
        base = CF.discover(dfx, dfp, tev, disc, params, FS._names(ticker, []), SEED)
        base_conf = CF.confirm(dfx, dfp, tev, conf, params, FS._names(ticker, []), SEED, base["q"])
        if base is None or base_conf is None:
            continue
        scores = {}
        for cid in cands:
            d = CF.discover(dfx, dfp, tev, disc, params, FS._names(ticker, [cid]), SEED)
            if d is not None:
                scores[int(cid)] = {"gain": d["mean"] - base["mean"], "q": d["q"],
                                    "folds": [c - b for c, b in
                                              zip(d["fold_growth"], base["fold_growth"])]}
        if not scores:
            continue
        top = max(scores, key=lambda i: scores[i]["gain"])
        c = CF.confirm(dfx, dfp, tev, conf, params, FS._names(ticker, [top]), SEED, scores[top]["q"])
        entry = {"confirmation_fold": r, "arms": {
            "flat": {"picked": top, "family": fam_of.get(top),
                     "confirm_delta": None if c is None else c["growth"] - base_conf["growth"]}}}
        byfam = {}
        for i, s in scores.items():
            byfam.setdefault(fam_of[i], []).append({"id": i, "mean": s["gain"], "folds": s["folds"]})
        fam = max(byfam, key=lambda f: max(x["mean"] for x in byfam[f]))
        plateau, best_id, _ = golden.one_se_set(byfam[fam])
        repr_id = golden.pick_representative(plateau, simplicity) or best_id
        c2 = CF.confirm(dfx, dfp, tev, conf, params, FS._names(ticker, [repr_id]), SEED,
                        scores[repr_id]["q"])
        entry["arms"]["hierarchical"] = {
            "picked": int(repr_id), "family": fam,
            "confirm_delta": None if c2 is None else c2["growth"] - base_conf["growth"]}
        rots.append(entry)

    v = ACC.verdict(rots)
    picked = [r["arms"]["flat"]["picked"] for r in rots]
    return {"ticker": ticker, "outer_fold": ofold, "strength": strength, "pool": pool,
            "replicate": rep, "planted_id": planted_id,
            "picked_in_rotations": sum(1 for p in picked if p == planted_id),
            "rotations": len(rots),
            "recovered_flat": bool(v["flat"]["accepted"] and v["flat"]["unit"] == planted_id),
            "accepted_anything_flat": bool(v["flat"]["accepted"]),
            "accepted_unit_flat": v["flat"]["unit"],
            "T_flat": v["flat"]["T"], "seconds": round(time.time() - t0, 1)}


def positive_control(jobs_n, ticker, ofold):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(ticker, ofold, s, p, r)
            for s in STRENGTHS for p in POOL_SIZES for r in range(REPLICATES)]
    print(f"kontrola pozytywna: {len(jobs)} eksperymentów "
          f"({len(STRENGTHS)} sił × {len(POOL_SIZES)} rozmiarów puli × {REPLICATES} powtórzeń)\n")
    rows = []
    if jobs_n > 1:
        with ProcessPoolExecutor(max_workers=jobs_n) as ex:
            for r in ex.map(positive_for, jobs):
                rows.append(r)
                print(f"  a={r['strength']:<5} pula={r['pool']:<3} rep={r['replicate']}  "
                      f"wybrana w {r['picked_in_rotations']}/{r['rotations']} rotacjach  "
                      f"odzyskana={r['recovered_flat']}  ({r['seconds']:.0f}s)", flush=True)
    else:
        for j in jobs:
            r = positive_for(j)
            rows.append(r)
            print(f"  a={r['strength']:<5} pula={r['pool']:<3} rep={r['replicate']}  "
                  f"odzyskana={r['recovered_flat']}  ({r['seconds']:.0f}s)", flush=True)

    power = {}
    for s in STRENGTHS:
        for p in POOL_SIZES:
            sel = [r for r in rows if r["strength"] == s and r["pool"] == p]
            k = sum(1 for r in sel if r["recovered_flat"])
            lo, hi = wilson(k, len(sel))
            power[f"a={s}|pool={p}"] = {
                "recovered": k, "of": len(sel), "power": round(k / len(sel), 3) if sel else None,
                "wilson_95": [round(lo, 3), round(hi, 3)],
                "core_seconds": round(sum(r["seconds"] for r in sel), 1)}
    detectable = [s for s in STRENGTHS
                  if all(power.get(f"a={s}|pool={p}", {}).get("power", 0) >= 0.8
                         for p in POOL_SIZES)]
    return {"_what": "a planted column carrying a known share of outcome information; the ladder "
                     "must find it, and the rate at which it does is the power",
            "construction": "z = a * standardised(y) + sqrt(1-a^2) * noise, replacing an existing "
                            "candidate's column so the name and family machinery is untouched",
            "_not_realistic": "this is a ruler, not a feature",
            "strengths": list(STRENGTHS), "pool_sizes": list(POOL_SIZES),
            "replicates": REPLICATES,
            "minimum_detectable_effect_at_power_0.8": (min(detectable) if detectable else None),
            "power": power, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--negative", action="store_true")
    ap.add_argument("--positive", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--ticker", default="ADBE")
    ap.add_argument("--outer-fold", type=int, default=2)
    ap.add_argument("--out", default=str(DATA / "null_controls.json"))
    args = ap.parse_args()

    doc = read_json(args.out, {}) or {}
    if args.negative or not (args.negative or args.positive):
        paths = {k: DATA / f"procedure_null_{k}.json" for k in ("a1", "a2", "b")}
        doc["negative_control"] = negative_control(paths)
        for kind, block in doc["negative_control"]["by_null"].items():
            for arm, s in block.items():
                print(f"  {kind}/{arm:<14} fałszywe akceptacje {s['false_acceptances']:>4}/"
                      f"{s['replicates']:<5} = {s['rate']:.3f}  95% CI {s['wilson_95']}")
    if args.positive:
        doc["positive_control"] = positive_control(args.jobs, args.ticker, args.outer_fold)
        p = doc["positive_control"]
        print(f"\n  minimalny wykrywalny efekt przy mocy 0,8: "
              f"{p['minimum_detectable_effect_at_power_0.8']}")
        for k, v in p["power"].items():
            print(f"    {k:<18} moc {v['power']} {v['wilson_95']}  ({v['core_seconds']:.0f} rdz-s)")

    sha = write_json_atomic(args.out, doc)
    print(f"\nwrote {args.out}  sha256 {sha[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
