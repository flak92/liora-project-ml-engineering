#!/usr/bin/env python3
"""Profile every ticker as what it is to us: a table of numbers with a name on it.

No sector, no issuer, no domain story — the question is only which tables are the least
troublesome to build a methodology on. The pipeline's own L4 gate (src/xgb/pipeline.py:223-244)
already rejects tables that are broken outright: NaN or duplicate timestamps, non-increasing
time, high < max(open,close), low > min(open,close), negative or non-finite volume. Every
sealed ticker passes it, so it separates nothing. What follows are graded measures — how
much of each defect a table carries — computed on the Train window only.

Two failure modes get measured, not one. A table can be bad because it is DAMAGED (holes,
frozen stretches, dead bars, impossible jumps) or because it is DEGENERATE (never moves,
prices land on a handful of values). Ranking on damage alone would hand us the flattest,
emptiest series in the store, which is the opposite of useful: a table with no variation
generates no label events and teaches the method nothing.

    python3 scripts/profile_tables.py            # -> config/table_profile.json
    python3 scripts/profile_tables.py --top 40   # also print the cleanest 40
"""
import argparse
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "xgb" / "data" / "liora.duckdb"
OUT = ROOT / "config" / "table_profile.json"

TRAIN_START = "2016-10-17"
TRAIN_END = "2023-12-29"

# One bar is one hour; anything longer than this between consecutive rows is a hole in the
# table. Regular overnight and weekend closes are far larger than any intraday gap, so the
# threshold is set above a normal overnight break (17:00 -> 09:30 next session ~ 16.5 h) and
# below a long weekend, then gaps are counted rather than judged: the number is what ranks.
GAP_HOURS = 24 * 4

QUERY = f"""
with w as (
    select ticker, timestamp, open, high, low, close, volume,
           lag(close)     over (partition by ticker order by timestamp) as prev_close,
           lag(timestamp) over (partition by ticker order by timestamp) as prev_ts
    from bars_1h
    where timestamp >= '{TRAIN_START}'::timestamptz
      and timestamp <  ('{TRAIN_END}'::date + interval 1 day)
),
m as (
    select
        ticker,
        count(*)                                          as bars,
        min(timestamp)                                    as first_bar,
        max(timestamp)                                    as last_bar,
        count(distinct close)                             as distinct_close,
        sum(case when volume = 0 then 1 else 0 end)       as zero_volume_bars,
        sum(case when close = prev_close then 1 else 0 end) as flat_bars,
        sum(case when prev_ts is not null
                  and date_diff('hour', prev_ts, timestamp) > {GAP_HOURS}
                 then 1 else 0 end)                       as gaps,
        max(case when prev_ts is not null
                 then date_diff('hour', prev_ts, timestamp) else 0 end) as max_gap_hours,
        max(case when prev_close > 0 and close > 0
                 then abs(ln(close / prev_close)) else 0 end)  as max_abs_log_return,
        sum(case when prev_close > 0 and close > 0
                  and abs(ln(close / prev_close)) > 0.35
                 then 1 else 0 end)                       as violent_bars,
        stddev_samp(case when prev_close > 0 and close > 0
                         then ln(close / prev_close) end)  as bar_return_sd,
        median(close * volume)                            as median_dollar_volume,
        min(close)                                        as min_close,
        max(close)                                        as max_close
    from w
    group by ticker
)
select * from m order by ticker
"""


def profile():
    if not STORE.exists():
        raise SystemExit(f"missing bar store: {STORE}")
    con = duckdb.connect(str(STORE), read_only=True)
    try:
        cur = con.execute(QUERY)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()

    max_bars = max(r["bars"] for r in rows)
    for r in rows:
        r["first_bar"] = str(r["first_bar"])
        r["last_bar"] = str(r["last_bar"])
        r["coverage"] = round(r["bars"] / max_bars, 6)
        r["zero_volume_share"] = round(r["zero_volume_bars"] / r["bars"], 6)
        r["flat_share"] = round(r["flat_bars"] / r["bars"], 6)
        r["price_resolution"] = round(r["distinct_close"] / r["bars"], 6)
        for k in ("max_abs_log_return", "bar_return_sd", "median_dollar_volume",
                  "min_close", "max_close"):
            r[k] = None if r[k] is None else round(float(r[k]), 8)
    return rows, max_bars


def quantiles(values, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)):
    s = sorted(v for v in values if v is not None)
    if not s:
        return {}
    def at(q):
        i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[i]
    return {f"q{int(q*100):02d}": at(q) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    rows, max_bars = profile()
    dist = {k: quantiles([r[k] for r in rows])
            for k in ("coverage", "zero_volume_share", "flat_share", "price_resolution",
                      "gaps", "max_gap_hours", "max_abs_log_return", "violent_bars",
                      "bar_return_sd", "median_dollar_volume")}

    OUT.write_text(json.dumps({
        "schema_version": "table_profile.v1",
        "train_window": {"start": TRAIN_START, "end": TRAIN_END},
        "gap_threshold_hours": GAP_HOURS,
        "tickers": len(rows),
        "max_bars_observed": max_bars,
        "distributions": dist,
        "tables": rows,
    }, indent=1) + "\n", encoding="utf-8")

    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows)} tabel, max {max_bars:,} barów)\n")
    print(f"{'miara':<24}" + "".join(f"{q:>14}" for q in ("q00", "q05", "q25", "q50", "q75", "q95", "q100")))
    for k, d in dist.items():
        print(f"{k:<24}" + "".join(f"{d.get(q, 0):>14,.4f}" if isinstance(d.get(q), float)
                                   else f"{d.get(q, 0):>14,}" for q in ("q00", "q05", "q25", "q50", "q75", "q95", "q100")))

    if args.top:
        print(f"\nnajczystsze {args.top} wg sumy defektów (podgląd, nie reguła):")
        ranked = sorted(rows, key=lambda r: (r["gaps"], r["zero_volume_share"], r["flat_share"]))
        for r in ranked[:args.top]:
            print(f"  {r['ticker']:<7} bars={r['bars']:>6,} cov={r['coverage']:.3f} "
                  f"gaps={r['gaps']:>3} zerovol={r['zero_volume_share']:.4f} "
                  f"flat={r['flat_share']:.4f} res={r['price_resolution']:.3f} "
                  f"maxjump={r['max_abs_log_return']:.3f}")


if __name__ == "__main__":
    main()
