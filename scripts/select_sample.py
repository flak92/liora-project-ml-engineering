#!/usr/bin/env python3
"""Pick the development sample by looking at each ticker as what it is here: a table of numbers.

A full research pass over 498 tables costs 22.4 core-hours; twenty tables cost 0.9, which is
the difference between a night and a coffee break. The danger in shrinking is fatal and
obvious: choose the twenty by how they PERFORMED and every number measured on them afterwards
is contaminated, because the test set would have reached the method through the selection.
So nothing here reads the OOS window, and nothing here reads a result. The rule asks one
question only — which tables are the least troublesome to build a methodology on.

No sector, no issuer, no company. A ticker is a column of names over a table of numbers.

The pipeline's own L4 gate (src/xgb/pipeline.py:223-244) already rejects tables that are
broken outright — NaN or duplicate timestamps, non-increasing time, high < max(open,close),
low > min(open,close), negative volume. Every sealed table passes it, so it separates
nothing. These thresholds are graded, and each one is a quantile of THIS store's own Train
window (see config/table_profile.json), not a number pulled from the air.

Two failure modes are filtered, not one:

  DAMAGED    holes in time, frozen stretches, impossible single-bar jumps
  DEGENERATE prices landing on so few distinct values that the table is a lattice, not a series

Ranking on damage alone would hand us the flattest, emptiest series in the store — a table
that never moves generates no label events and teaches the method nothing.

Finally, redundancy: two tables whose bar-to-bar returns correlate above 0.95 are the same
numbers twice, and a duplicate spends one of twenty slots for nothing. The threshold sits in
open space — in this store the highest pair reaches 0.9916 and the next reaches 0.7173.

    python3 scripts/select_sample.py            # writes config/sample_20.json
    python3 scripts/select_sample.py --check    # recompute and compare, write nothing

The output carries a SHA-256 of the ticker list so a reviewer can prove the sample was not
quietly edited afterwards.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "xgb" / "data" / "liora.duckdb"
PROFILE = ROOT / "config" / "table_profile.json"
OUT = ROOT / "config" / "sample_20.json"

TRAIN_START = "2016-10-17"
TRAIN_END = "2023-12-29"
SAMPLE_SIZE = 20
CORR_MAX = 0.95

# Each threshold, what it rejects, and where the number comes from in this store's own
# population of 498 tables. Order matters only for the reporting of how many each one drops.
FILTERS = [
    ("coverage >= 0.99", "bars present across the Train window",
     "population q05 is 0.772 and q25 is 0.998 — the tail below 0.99 is short history, not a hole",
     lambda r: r["coverage"] >= 0.99),
    ("gaps == 0", "no stretch longer than four days without a bar",
     "q00..q95 are all 0; only a handful of tables carry any",
     lambda r: r["gaps"] == 0),
    ("max_gap_hours <= 100", "the largest hole is a long weekend, not a hiatus",
     "q00..q95 sit at 90..93 h (a normal long weekend); q100 reaches 13,890 h",
     lambda r: r["max_gap_hours"] <= 100),
    ("violent_bars == 0", "no single bar moves more than 35% in log terms",
     "q00..q75 are 0; such a bar smells of an event the split adjustment did not close",
     lambda r: r["violent_bars"] == 0),
    ("max_abs_log_return <= 0.35", "the same condition stated on the maximum",
     "q100 reaches 1.78, i.e. +495% inside one hour",
     lambda r: r["max_abs_log_return"] <= 0.35),
    ("flat_share <= 0.0052", "few bars repeat the previous close",
     "population q25 — frozen bars break rolling statistics and z-scores",
     lambda r: r["flat_share"] <= 0.0052),
    ("price_resolution >= 0.7489", "prices resolve into many distinct values",
     "population q75 — a low-resolution table is a lattice, and its features quantise with it",
     lambda r: r["price_resolution"] >= 0.7489),
]

# zero_volume_share is deliberately NOT a filter: it is 0.0000 across the whole population,
# q100 included, so it separates nothing in this store. Recorded in the profile regardless.


def load_profile():
    if not PROFILE.exists():
        sys.exit(f"missing {PROFILE.relative_to(ROOT)} — run scripts/profile_tables.py first")
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def correlations(tickers):
    """Pairwise correlation of bar-to-bar log returns, on bars the two tables share."""
    con = duckdb.connect(str(STORE), read_only=True)
    try:
        rows = con.execute(
            f"""
            with w as (
                select ticker, timestamp,
                       ln(close / lag(close) over (partition by ticker order by timestamp)) as r
                from bars_1h
                where ticker in ({','.join('?' * len(tickers))})
                  and timestamp >= '{TRAIN_START}'::timestamptz
                  and timestamp <  ('{TRAIN_END}'::date + interval 1 day)
            )
            select a.ticker, b.ticker, corr(a.r, b.r)
            from w a join w b on a.timestamp = b.timestamp and a.ticker < b.ticker
            where a.r is not null and b.r is not null
            group by 1, 2 having count(*) > 5000
            """,
            tickers,
        ).fetchall()
    finally:
        con.close()
    out = {}
    for a, b, c in rows:
        if c is None:
            continue
        out[(a, b)] = out[(b, a)] = float(c)
    return out


def digest(tickers):
    return hashlib.sha256("\n".join(tickers).encode()).hexdigest()


def build():
    prof = load_profile()
    tables = prof["tables"]

    dropped = []
    survivors = tables
    for name, what, why, fn in FILTERS:
        before = len(survivors)
        survivors = [r for r in survivors if fn(r)]
        dropped.append({"filter": name, "rejects": what, "threshold_from": why,
                        "dropped": before - len(survivors), "remaining": len(survivors)})

    if len(survivors) < SAMPLE_SIZE:
        sys.exit(f"only {len(survivors)} tables clear the filters, fewer than {SAMPLE_SIZE}")

    # Cleanest first: the most resolved table, then the least frozen, then alphabetical so
    # two runs on the same store cannot disagree.
    survivors.sort(key=lambda r: (-r["price_resolution"], r["flat_share"], r["ticker"]))

    pool = [r["ticker"] for r in survivors[: max(SAMPLE_SIZE * 2, 40)]]
    corr = correlations(pool)

    picked, redundant = [], []
    for r in survivors:
        t = r["ticker"]
        clash = next(((p, corr[(t, p)]) for p in picked
                      if corr.get((t, p), 0.0) > CORR_MAX), None)
        if clash:
            redundant.append({"ticker": t, "duplicate_of": clash[0], "correlation": round(clash[1], 4)})
            continue
        picked.append(t)
        if len(picked) == SAMPLE_SIZE:
            break

    chosen = {r["ticker"]: r for r in tables}
    quarantined = sorted(r["ticker"] for r in tables if r["ticker"] not in picked)

    return {
        "schema_version": "sample.v2",
        "rule": (
            "Tables only — no sector, no issuer, no company. Filter the 498 Train-window "
            "tables on graded data quality (thresholds are this store's own quantiles), rank "
            "the survivors cleanest-first by price resolution then by frozen-bar share, and "
            f"walk down that ranking taking the first {SAMPLE_SIZE} that are not numerically "
            f"redundant (bar-return correlation > {CORR_MAX}) with something already taken. "
            "No OOS bar is read and no result is consulted."
        ),
        "train_window": {"start": TRAIN_START, "end": TRAIN_END},
        "oos_window_not_read": "2024-01-02 -> 2026-05-29",
        "source_store": str(STORE.relative_to(ROOT)),
        "source_profile": str(PROFILE.relative_to(ROOT)),
        "filters": dropped,
        "redundancy": {"max_correlation": CORR_MAX, "rejected": redundant},
        "sample_size": SAMPLE_SIZE,
        "sample": picked,
        "sample_sha256": digest(picked),
        "measurements": [
            {k: chosen[t][k] for k in
             ("ticker", "bars", "coverage", "price_resolution", "flat_share",
              "bar_return_sd", "max_abs_log_return", "gaps", "median_dollar_volume")}
            for t in picked
        ],
        "quarantine": {
            "note": (
                "Untouched until the methodology is frozen. This is the holdout the method is "
                "tested on afterwards; reading it during development would spend the evidence."
            ),
            "count": len(quarantined),
            "quarantine_sha256": digest(quarantined),
            "tickers": quarantined,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="recompute and compare against the committed file; write nothing")
    args = ap.parse_args()

    fresh = build()

    if args.check:
        if not OUT.exists():
            sys.exit(f"{OUT.relative_to(ROOT)} does not exist — run without --check first")
        held = json.loads(OUT.read_text(encoding="utf-8"))
        problems = [
            msg for cond, msg in (
                (held.get("sample") != fresh["sample"], "sample list differs from what the rule produces"),
                (held.get("sample_sha256") != fresh["sample_sha256"], "sample_sha256 differs"),
                (held.get("quarantine", {}).get("quarantine_sha256")
                 != fresh["quarantine"]["quarantine_sha256"], "quarantine_sha256 differs"),
            ) if cond
        ]
        if problems:
            for p in problems:
                print(f"FAIL  {p}")
            sys.exit(1)
        print(f"OK: the rule reproduces the committed sample of {len(fresh['sample'])} "
              f"({fresh['sample_sha256'][:12]}…) and the {fresh['quarantine']['count']}-table quarantine.")
        return

    OUT.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for f in fresh["filters"]:
        print(f"  {f['filter']:<30} -{f['dropped']:>3}  -> {f['remaining']:>3}")
    for r in fresh["redundancy"]["rejected"]:
        print(f"  redundant: {r['ticker']} ~ {r['duplicate_of']} (corr {r['correlation']})")
    print(f"\n  sample ({len(fresh['sample'])}): {' '.join(fresh['sample'])}")
    print(f"  sample_sha256: {fresh['sample_sha256']}")
    print(f"  quarantined:   {fresh['quarantine']['count']} tables")


if __name__ == "__main__":
    main()
