#!/usr/bin/env python3
"""Cut the delivered store and artifact tree down to the development sample.

The repository shipped a 498-table study: 993 artifact folders and a 33 MB store, so a
clone took minutes and a full research pass took 22.4 core-hours. The work moved to a
twenty-table sample (config/sample_20.json, chosen by a rule that never read an OOS bar),
and this script makes the delivery match: what a reader finds in the clone is exactly the
twenty tables the methodology is being built on.

Three things here are easy to get wrong, and each one produces a repository that passes
every gate while lying:

  research_run.xgb_assets / lstm_assets — app/data.py:82 compares the row counts against
  these two numbers and returns DATASET STATUS: PARTIAL when they disagree. Cut the store
  without updating them and all three verifiers stay green while the console refuses to
  come up. `make verify` cannot see this; only `make on` can.

  integrity_checks — the table stores the RESULT of each check, it does not recompute it.
  Left alone, sixteen rows keep saying PASS while six of their messages describe a store
  that no longer exists ("xgb=498 lstm=495"). The messages are rebuilt here from the store
  as it stands after the cut, so they are true by construction rather than by editing.

  oos_read_summary — untouched, deliberately. The 588 XGB and 495 LSTM reads are a fact of
  the sealed epoch recorded in an append-only ledger. Trimming those rows to match the
  sample, or recomputing them, would be falsifying the history of OOS reads, which is the
  one thing this project treats as disqualifying. "reads=588 store=20" stays, and stays
  true: every sealed row was a counted read, and that remains so when the delivery shrinks.

research_status is likewise left alone: app/data.py:76 requires the literal
FROZEN_FINAL_RESEARCH_SNAPSHOT, and the snapshot really is frozen — what changed is how
much of it travels in the clone.

    python3 scripts/slim_to_sample.py --dry-run   # say what would happen, touch nothing
    python3 scripts/slim_to_sample.py             # do it

data/results.db and artifacts/ are tracked, so `git checkout -- data artifacts` undoes it.
"""
import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "results.db"
MANIFEST = ROOT / "artifacts" / "manifest.json"
SAMPLE = ROOT / "config" / "sample_20.json"

# Every table that carries a ticker. asset_features has no model column in some rows, so
# the filter is on ticker alone throughout — the sample is a set of tables, not of models.
TICKER_TABLES = [
    "asset_results",
    "asset_features",
    "feature_search_summary",
    "feature_contributions",
    "feature_train_stats",
    "xgb_entry_ranges",
]


def sample_tickers():
    if not SAMPLE.exists():
        sys.exit(f"missing {SAMPLE.relative_to(ROOT)} — run scripts/select_sample.py first")
    return sorted(json.loads(SAMPLE.read_text(encoding="utf-8"))["sample"])


def counts(con):
    return {t: con.execute(f"select count(*) from {t}").fetchone()[0] for t in TICKER_TABLES}


def rebuild_integrity_messages(con):
    """Restate the six messages that describe the store, from the store as it now stands.

    The remaining ten checks assert invariants (taxonomy, accounting, hash parity) whose
    messages carry no cardinality, so they survive the cut unchanged and are left alone.
    """
    n = {m: con.execute("select count(*) from asset_results where model=?", (m,)).fetchone()[0]
         for m in ("xgb", "lstm")}
    stats = con.execute("select count(*) from feature_train_stats").fetchone()[0]
    contribs = con.execute("select count(*) from feature_contributions").fetchone()[0]
    ranges = con.execute("select count(*) from xgb_entry_ranges").fetchone()[0]
    total = n["xgb"] + n["lstm"]

    # reads stay at the epoch's ledger values; only the store side of the sentence moves.
    def reread(msg, store_n):
        head = msg.split(" store=")[0]
        return f"{head} store={store_n} (every sealed row must be a counted read)"

    updates = {
        "store_counts": f"xgb={n['xgb']} lstm={n['lstm']}",
        "theta_parity_harvest_vs_store": f"{n['xgb']} harvested; mismatches=[]",
        "artifact_harvest_counts": (
            f"manifest={{'lstm': {n['lstm']}, 'total': {total}, 'xgb': {n['xgb']}}} "
            f"disk={{'xgb': {n['xgb']}, 'lstm': {n['lstm']}}} "
            f"sealed={{'xgb': {n['xgb']}, 'lstm': {n['lstm']}}}"
        ),
        "interpretation_coverage": (
            f"stats={stats} contribs={contribs} ranges={ranges} missing_payloads=[]"
        ),
    }
    for name, store_n in (("oos_reads_cover_store_xgb", n["xgb"]),
                          ("oos_reads_cover_store_lstm", n["lstm"])):
        row = con.execute("select message from integrity_checks where check_name=?", (name,)).fetchone()
        if row:
            updates[name] = reread(row[0], store_n)

    for name, msg in updates.items():
        con.execute("update integrity_checks set message=? where check_name=?", (msg, name))
    return updates


def slim_store(keep, dry):
    con = sqlite3.connect(DB)
    try:
        before = counts(con)
        marks = ",".join("?" * len(keep))
        after = {}
        for t in TICKER_TABLES:
            after[t] = con.execute(
                f"select count(*) from {t} where ticker in ({marks})", keep).fetchone()[0]

        print(f"  {'tabela':<24}{'przed':>8}{'zostaje':>9}{'usuwa':>8}")
        for t in TICKER_TABLES:
            print(f"  {t:<24}{before[t]:>8,}{after[t]:>9,}{before[t]-after[t]:>8,}")

        if dry:
            return

        for t in TICKER_TABLES:
            con.execute(f"delete from {t} where ticker not in ({marks})", keep)

        n = {m: con.execute("select count(*) from asset_results where model=?", (m,)).fetchone()[0]
             for m in ("xgb", "lstm")}
        con.execute("update research_run set xgb_assets=?, lstm_assets=?", (n["xgb"], n["lstm"]))
        print(f"  research_run: xgb_assets={n['xgb']} lstm_assets={n['lstm']}  "
              f"(bez tego health() zwraca PARTIAL i konsola nie wstaje)")

        for name, msg in rebuild_integrity_messages(con).items():
            print(f"  integrity_checks[{name}] -> {msg[:78]}")

        con.commit()
        con.execute("vacuum")
        con.commit()
    finally:
        con.close()


def slim_manifest(keep, dry):
    doc = json.loads(MANIFEST.read_text(encoding="utf-8"))
    meta = doc.get("_meta", {})
    entries = {k: v for k, v in doc.items() if k != "_meta"}

    def kept(key, payload):
        t = payload.get("ticker") if isinstance(payload, dict) else None
        return (t or key.split("/")[-1]) in set(keep)

    survivors = {k: v for k, v in entries.items() if kept(k, v)}
    per_model = {}
    for v in survivors.values():
        m = v.get("model") if isinstance(v, dict) else None
        if m:
            per_model[m] = per_model.get(m, 0) + 1

    print(f"  manifest: {len(entries)} wpisów -> {len(survivors)}  ({per_model})")
    if dry:
        return

    meta["counts"] = {**per_model, "total": len(survivors)}
    MANIFEST.write_text(json.dumps({"_meta": meta, **survivors}, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")


def slim_artifacts(keep, dry):
    keep = set(keep)
    doomed = []
    for model_dir in sorted((ROOT / "artifacts").glob("*")):
        if not model_dir.is_dir():
            continue
        for d in sorted(model_dir.iterdir()):
            if d.is_dir() and d.name not in keep:
                doomed.append(d)
    print(f"  artefakty: usuwa {len(doomed)} katalogów, zostaje "
          f"{sum(1 for m in (ROOT / 'artifacts').glob('*') if m.is_dir() for _ in m.iterdir()) - len(doomed)}")
    if dry:
        return
    for d in doomed:
        shutil.rmtree(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep = sample_tickers()
    print(f"próbka ({len(keep)}): {' '.join(keep)}\n")

    print("STORE")
    slim_store(keep, args.dry_run)
    print("\nMANIFEST")
    slim_manifest(keep, args.dry_run)
    print("\nARTEFAKTY")
    slim_artifacts(keep, args.dry_run)

    print("\n" + ("dry-run — nic nie zmieniono." if args.dry_run else
                  "zrobione. Teraz: make verify ORAZ make on (health() sprawdza tylko ta druga)."))


if __name__ == "__main__":
    main()
