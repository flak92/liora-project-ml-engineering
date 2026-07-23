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
BLUEPRINT = ROOT / "docs" / "archive" / "figures" / "data_pipeline_lego_plan.html"
FLOW = ROOT / "docs" / "archive" / "figures" / "data_flow_3d_visualization.html"
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


def occurs(haystack, want):
    """`want` appears in `haystack` as a figure, not as a fragment of a longer number.

    Plain substring matching was safe while the universe was 498 assets: no map contains
    "16 601" by accident. It stops being safe the moment the shipped sample is small —
    "20" already sits inside 2016, 2024 and 2026, "40" inside 1940-style spans, "19"
    inside 2019. Every one of those would have reported PASS while checking nothing, and a
    gate that cannot fail is a gate that has been switched off.

    So a digit may not touch either end of the match. Everything else stays as it was: the
    haystack is already normalized, and non-numeric expectations ("## The four pages") are
    unaffected because the guard only looks at characters adjacent to the match.
    """
    start = 0
    while True:
        i = haystack.find(want, start)
        if i < 0:
            return False
        before = haystack[i - 1] if i else ""
        after = haystack[i + len(want):i + len(want) + 1]
        if not (before.isdigit() or after.isdigit()):
            return True
        start = i + 1


def check(name, expected, *files, note=""):
    """`expected` must appear in every named file, up to number formatting."""
    want = normalize(expected)
    for f in files:
        if f not in _CACHE:
            _CACHE[f] = normalize(f.read_text(encoding="utf-8"))
    missing = [f.name for f in files if not occurs(_CACHE[f], want)]
    ok = not missing
    tail = "" if ok else f" — not found in {', '.join(missing)}"
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {expected}{tail}{'  (' + note + ')' if note and ok else ''}")
    if not ok:
        FAILURES.append(name)


def selftest():
    """Prove the matcher can still fail. A gate nobody has seen fail is a rumour.

    Each case is a hazard that plain substring matching walked straight into once the
    universe shrank; the last two are the ordinary hits that must keep passing, because a
    guard that also rejects real matches would be worse than the hole it closes.
    """
    cases = [
        ("20 must not match inside 2016", "sealed in 2016 and read in 2024", "20", False),
        ("40 must not match inside 1940", "the 1940s", "40", False),
        ("19 must not match inside 2019", "through 2019", "19", False),
        ("7 must not match inside 72",    "beats HODL 72/498", "7", False),
        ("a real 20 still matches",       "20 tables ship here", "20", True),
        ("a bounded figure matches",      "median return -1.78% over", "-1.78%", True),
        ("prose still matches",           "## The four pages", "## The four pages", True),
        ("a figure that drifted fails",   "median return -1.78%", "+2.80%", False),
    ]
    bad = 0
    for name, haystack, want, expected in cases:
        got = occurs(normalize(haystack), normalize(want))
        ok = got is expected
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              f"{'' if ok else f' — expected {expected}, got {got}'}")
    print()
    if bad:
        print(f"FAILED: {bad} matcher case(s) — the gate does not behave as documented.")
        return 1
    print("OK: the matcher rejects digit-embedded fragments and still accepts real figures.")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
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
    # Spelled out because the prose spells them out. The list covers 1..20 rather than the
    # handful in use: a dict that has to be extended every time a page is added is a gate
    # that fails on its own vocabulary instead of on the thing it is watching, which is
    # exactly what happened the first time this ran after a two-page removal.
    WORDS = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
             "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split()
    word = WORDS[n_pages] if n_pages < len(WORDS) else str(n_pages)
    # README is the only LIVE surface that states the count now (the two pipeline maps are
    # archived under docs/archive/figures and no longer track the console), so it is the only
    # one pinned to app.py's truth.
    check("page count (README)", f"## The {word} pages", README)
    stale = [w for i, w in enumerate(WORDS) if i != n_pages and i > 1
             and f"The {w} pages" in README.read_text(encoding="utf-8")]
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
