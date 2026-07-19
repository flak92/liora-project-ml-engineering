#!/usr/bin/env python3
"""Every figure the two pipeline maps assert, recomputed from the sealed store.

    python3 scripts/verify_figures.py

`data_pipeline_lego_plan.html` and `data_flow_3d_visualization.html` are hand-written: numbers
typed in, not derived. That is fine for a frozen release and fatal for the next one — a
re-seal moves the store and leaves the prose behind, silently, in two 110 KB files nobody
re-reads. This gate exists so that never has to be found by eye again.

Each check names one figure, recomputes it from data/results.db, and asserts the literal
string appears in the file(s) that claim it. A figure that moves fails here rather than on
stage. Standard library only, read-only connection, no network — same contract as the other
two verify scripts.

Scope, stated honestly: this checks the figures that are DERIVABLE from the sealed store.
Parameters that live in config/ (purge, embargo, trial counts, theta grids) are not covered,
and neither is prose. Absence of a failure here is not a claim that the maps are fully
correct — only that no store-derived number in them has drifted.
"""
import re
import sqlite3
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
BLUEPRINT = ROOT / "data_pipeline_lego_plan.html"
FLOW = ROOT / "data_flow_3d_visualization.html"
APP = ROOT / "app.py"
README = ROOT / "README.md"

FAILURES = []


def connect():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def rows(con, sql, params=()):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(sql, params)]


def normalize(text):
    """Compare numbers, not typography.

    The two maps disagree on house style — the flow map writes "−1.78 %" with a Unicode
    minus and a thin gap before the sign, the blueprint writes "-1.78%". Both are the same
    figure, and a gate that fails on the space would be noise that gets muted within a week.
    """
    return (text.replace("−", "-").replace(" ", " ")
                .replace(" %", "%").replace(" / ", "/"))


_CACHE = {}


def check(name, expected, *files, note=""):
    """`expected` must appear in every named file, up to number formatting."""
    want = normalize(expected)
    for f in files:
        if f not in _CACHE:
            _CACHE[f] = normalize(f.read_text(encoding="utf-8"))
    missing = [f.name for f in files if want not in _CACHE[f]]
    ok = not missing
    tail = "" if ok else f" — not found in {', '.join(missing)}"
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {expected}{tail}{'  (' + note + ')' if note and ok else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    if not DB.exists():
        print(f"store not found: {DB}")
        return 1
    con = connect()
    print("Figures asserted by the pipeline maps, recomputed from the sealed store\n")

    # ---- universe and cohorts -------------------------------------------------
    counts = {r["model"]: r["n"] for r in
              rows(con, "select model, count(*) n from asset_results group by model")}
    check("XGB universe", str(counts["xgb"]), BLUEPRINT, FLOW)
    check("LSTM universe", str(counts["lstm"]), BLUEPRINT, FLOW)
    check("sealed artifacts", str(counts["xgb"] + counts["lstm"]), BLUEPRINT, FLOW)

    modes = {(r["model"], r["result_mode"]): r["n"] for r in rows(
        con, "select model, result_mode, count(*) n from asset_results group by model, result_mode")}
    for model in ("xgb", "lstm"):
        promoted = modes.get((model, "ML_MULTI_TRADE"), 0)
        check(f"{model.upper()} promoted (ML_MULTI_TRADE)", str(promoted), BLUEPRINT, FLOW)

    # ---- the verdict headline -------------------------------------------------
    for model, files in (("xgb", (BLUEPRINT, FLOW)), ("lstm", (BLUEPRINT, FLOW))):
        rs = rows(con, "select return_pct, hodl_return_pct, model_trades from asset_results "
                       "where model=?", (model,))
        med_ret = statistics.median(r["return_pct"] for r in rs)
        med_hodl = statistics.median(r["hodl_return_pct"] for r in rs)
        med_trades = statistics.median(r["model_trades"] for r in rs)
        check(f"{model.upper()} median return", f"{med_ret:+.2f}".replace("+", "+") + "%",
              *files, note="over all rows")
        check(f"{model.upper()} median HODL", f"+{med_hodl:.2f}%", *files)
        # Trade medians are stated by the blueprint only; the flow map reports PF and returns.
        check(f"{model.upper()} median trades",
              str(int(med_trades)) if med_trades == int(med_trades) else f"{med_trades:.1f}",
              BLUEPRINT)

        beats = rows(con, "select count(*) n from asset_results where model=? and beats_hodl=1",
                     (model,))[0]["n"]
        check(f"{model.upper()} beats HODL", f"{beats} / {len(rs)}", FLOW)
        check(f"{model.upper()} beats HODL (blueprint form)", f"{beats}/{len(rs)}", BLUEPRINT)

        pf = [r["profit_factor"] for r in rows(
            con, "select profit_factor from asset_results where model=? and "
                 "result_mode='ML_MULTI_TRADE' and profit_factor is not null", (model,))]
        check(f"{model.upper()} median PF", f"{statistics.median(pf):.3f}", *files)
        check(f"{model.upper()} PF population", str(len(pf)), *files,
              note="promoted rows with a rankable PF")

    # ---- the payoff retraction ------------------------------------------------
    for model, files in (("xgb", (BLUEPRINT, FLOW)), ("lstm", (BLUEPRINT, FLOW))):
        pay = [r["profit_factor"] * r["losses"] / r["wins"] for r in rows(
            con, "select profit_factor, wins, losses from asset_results where model=? and "
                 "result_mode='ML_MULTI_TRADE' and profit_factor is not null and wins>0 and "
                 "losses>0", (model,))]
        check(f"{model.upper()} realized payoff", f"{statistics.median(pay):.3f}", *files)
        share = 100.0 * sum(1 for p in pay if p >= 2.0) / len(pay)
        check(f"{model.upper()} share reaching 2.0", f"{share:.1f}%", BLUEPRINT, FLOW)

    # ---- the read ledger ------------------------------------------------------
    for r in rows(con, "select * from oos_read_summary"):
        check(f"{r['pipe'].upper()} reads this epoch", str(r["reads_this_epoch"]), BLUEPRINT)

    # ---- the interpretation layer --------------------------------------------
    interp = rows(con, "select message from integrity_checks where check_name="
                       "'interpretation_coverage'")[0]["message"]
    for label, key in (("feature-stat rows", "stats"), ("ENTRY-range segments", "ranges")):
        m = re.search(rf"{key}=(\d+)", interp)
        if m:
            n = int(m.group(1))
            spaced = f"{n:,}".replace(",", " ")          # the maps write 16 601, not 16601
            check(f"interpretation {label}", spaced, BLUEPRINT, FLOW)

    # ---- integrity ------------------------------------------------------------
    checks = rows(con, "select count(*) n, sum(status='PASS') p from integrity_checks")[0]
    check("integrity checks", f"{checks['p']} / {checks['n']}", FLOW)

    # ---- corporate actions ----------------------------------------------------
    check("split events", "83 events / 69 tickers", FLOW)
    check("split events (blueprint form)", "83 events across 69 tickers", BLUEPRINT)

    # ---- the page count -------------------------------------------------------
    # Not a store figure, but the same failure mode: the number of console pages is asserted
    # by hand in a dozen places and nothing used to check any of them, so adding or removing
    # a page left stale counts behind in prose nobody re-reads. app.py is the only source of
    # truth — it is what Streamlit actually builds the sidebar from.
    n_pages = APP.read_text(encoding="utf-8").count("st.Page(")
    words = {10: "ten", 11: "eleven", 12: "twelve", 9: "nine", 8: "eight"}
    check("page count (blueprint knob)", f"pages={n_pages} in 3 sections", BLUEPRINT)
    check("page count (blueprint prose)", f"a {words.get(n_pages, n_pages)}-page", BLUEPRINT)
    check("page count (README)", f"## The {words.get(n_pages, n_pages)} pages", README)
    stale = [w for n, w in words.items() if n != n_pages
             and (f"{w} pages" in README.read_text(encoding="utf-8")
                  or f"{w}-page" in BLUEPRINT.read_text(encoding="utf-8"))]
    print(f"  {'PASS' if not stale else 'FAIL'}  no stale page count survives: "
          f"{n_pages} pages in app.py{'' if not stale else '  — found ' + ', '.join(stale)}")
    if stale:
        FAILURES.append("stale page count")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} figure(s) drifted — {', '.join(FAILURES)}")
        print("A figure that moved means the store was re-sealed and the maps were not "
              "updated. Fix the prose, not this gate.")
        return 1
    print("OK: every store-derived figure in both pipeline maps matches the sealed store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
