#!/usr/bin/env python3
"""Self-test for the Rung 8 Family Transfer aggregator — proves the guarantees, reads no OOS.

Groups (zlecenie §13): registry integrity · determinism · fail-closed · deduplication · leakage ·
snapshot parity. Every check is fast and offline; the only real artifact touched is the frozen
snapshot, read-only. Exit 0 iff all pass.

    .venv/bin/python scripts/family_transfer_selftest.py         (or: make family-transfer-selftest)
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import family_transfer as FT                                                  # noqa: E402

SNAPSHOT = ROOT / "results" / "methodology_snapshot"
PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print("  [%s] %s%s" % (PASS if ok else FAIL, name, ("  — " + detail) if detail else ""))
    return ok


def _tmp_snapshot():
    """A writable copy of the frozen snapshot to mutate for fail-closed tests."""
    d = Path(tempfile.mkdtemp(prefix="ft_selftest_"))
    dst = d / "snap"
    shutil.copytree(SNAPSHOT, dst)
    return d, dst


# ── registry integrity ──────────────────────────────────────────────────────────────────────────────
def test_registry():
    id_to_family, family_to_ids, dup = FT.load_family_map()
    ids = [i for v in family_to_ids.values() for i in v]
    check("registry: exactly one family per searchable id", len(ids) == len(set(ids)) and not dup,
          "%d ids, %d dup" % (len(ids), len(dup)))
    check("registry: candidate count == 45", len(set(ids)) == 45, "%d" % len(set(ids)))
    check("registry: family count == 12", len(family_to_ids) == 12, "%d" % len(family_to_ids))
    core_leak = [i for i in range(1, 18) if i in id_to_family]
    check("registry: frozen 1h core 1-17 not searchable", not core_leak, "leaked %s" % core_leak)
    names = FT.load_names()
    unnamed = [i for i in set(ids) if names.get(i) is None]
    check("registry: every candidate id resolves to a name", not unnamed, "unnamed %s" % unnamed)


# ── determinism ─────────────────────────────────────────────────────────────────────────────────────
def test_determinism():
    a = FT.serialize(FT.build(SNAPSHOT, source_label="snapshot"))
    b = FT.serialize(FT.build(SNAPSHOT, source_label="snapshot"))
    check("determinism: two builds are byte-identical", a == b)
    check("determinism: canonical (sort_keys, no trailing noise)", a == FT.serialize(json.loads(a)))
    check("determinism: no timestamp key in hashed artifact",
          all(k not in a for k in ('"timestamp"', '"created_utc"', '"generated_at"', '"mtime"')))

    # mtime invariance: touch inputs, rebuild, still identical
    dtmp, snap = _tmp_snapshot()
    try:
        for p in snap.glob("*.json"):
            os.utime(p, (0, 0))
        c = FT.serialize(FT.build(snap, source_label="snapshot"))
        # panel/ids identical regardless of path label difference in run_identity.source only
        ca, cb = json.loads(a), json.loads(c)
        for k in ("families", "funnel", "minimal_panel_family_set", "taxonomy_diagnostics"):
            check("determinism: mtime-invariant (%s)" % k, ca[k] == cb[k])
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)

    # PYTHONHASHSEED invariance (subprocess, since it is read at interpreter start)
    out1 = _run_cli(["--snapshot", "--out", "-"], hashseed="0")
    out2 = _run_cli(["--snapshot", "--out", "-"], hashseed="12345")
    check("determinism: PYTHONHASHSEED-invariant", out1 == out2 and out1 != "")


def _run_cli(extra_args, hashseed=None):
    """Run the aggregator to a temp file and return its bytes (─ so we compare the written artifact)."""
    env = dict(os.environ)
    if hashseed is not None:
        env["PYTHONHASHSEED"] = hashseed
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out = tf.name
    args = [sys.executable, str(ROOT / "scripts" / "family_transfer.py")]
    args += [a if a != "-" else out for a in extra_args]
    subprocess.run(args, env=env, cwd=str(ROOT), capture_output=True, check=False)
    data = Path(out).read_text(encoding="utf-8")
    Path(out).unlink(missing_ok=True)
    return data


# ── fail-closed ─────────────────────────────────────────────────────────────────────────────────────
def test_fail_closed():
    # missing A2 or B → nothing can be stable
    dtmp, snap = _tmp_snapshot()
    try:
        (snap / "procedure_null_a2.json").unlink()
        (snap / "procedure_null_b.json").unlink()
        rep = FT.build(snap, source_label="snapshot")
        check("fail-closed: missing A2/B ⇒ no stable survivor", rep["funnel"]["stable_a1_a2_b"] == 0)
        check("fail-closed: missing A2/B ⇒ no retained", rep["funnel"]["retained_rung6"] == 0)
        check("fail-closed: missing inputs are reported",
              "procedure_null_a2.json" in rep["integrity"]["missing_inputs"])
        check("fail-closed: no family is PANEL_STABLE without the nulls",
              all(f["status"] != "PANEL_STABLE" for f in rep["families"]))
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)

    # missing Rung 6 → cannot be retained (retained ≠ default)
    dtmp, snap = _tmp_snapshot()
    try:
        (snap / "rung6_survivor_hpo.json").unlink()
        rep = FT.build(snap, source_label="snapshot")
        check("fail-closed: missing Rung 6 ⇒ retained == 0", rep["funnel"]["retained_rung6"] == 0)
        check("fail-closed: missing Rung 6 ⇒ no PANEL_STABLE",
              all(f["status"] != "PANEL_STABLE" for f in rep["families"]))
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)

    # unknown feature id → integrity failure surfaced
    dtmp, snap = _tmp_snapshot()
    try:
        a1 = json.loads((snap / "procedure_null_a1.json").read_text())
        a1["tables"]["ZZZZ"] = {"ticker": "ZZZZ", "folds": [
            {"outer_fold": 0, "arms": {"flat": {"verdict": "passed", "unit": 9999}}}]}
        (snap / "procedure_null_a1.json").write_text(json.dumps(a1))
        rep = FT.build(snap, source_label="snapshot")
        check("fail-closed: unknown feature id ⇒ unmapped_feature_ids",
              9999 in rep["integrity"]["unmapped_feature_ids"])
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)

    # duplicate family membership → integrity failure (crafted families config via monkeypatch path)
    dtmp = Path(tempfile.mkdtemp(prefix="ft_dupfam_"))
    try:
        bad = {"schema_version": "feature_families.v1", "families": {"a": [101, 102], "b": [102, 103]}}
        badp = dtmp / "families.json"
        badp.write_text(json.dumps(bad))
        orig = FT.FAMILIES_PATH
        FT.FAMILIES_PATH = badp
        try:
            _, _, dup = FT.load_family_map()
            check("fail-closed: duplicate family membership detected", bool(dup),
                  "%s" % (dup or "none"))
        finally:
            FT.FAMILIES_PATH = orig
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)

    # corrupt JSON → STOP (raise), not silent skip
    dtmp, snap = _tmp_snapshot()
    try:
        (snap / "crossfit_selection.json").write_text("{ this is not json ")
        raised = False
        try:
            FT.build(snap, source_label="snapshot")
        except FT.IntegrityStop:
            raised = True
        check("fail-closed: corrupt JSON halts the report", raised)
    finally:
        shutil.rmtree(dtmp, ignore_errors=True)


# ── deduplication ───────────────────────────────────────────────────────────────────────────────────
def test_dedup():
    rep = FT.build(SNAPSHOT, source_label="snapshot")
    check("dedup: two retained arms, one unique representative",
          rep["funnel"]["retained_rung6"] == 2 and rep["funnel"]["unique_retained_representatives"] == 1,
          "retained=%d rep=%d" % (rep["funnel"]["retained_rung6"],
                                  rep["funnel"]["unique_retained_representatives"]))
    osc = next(f for f in rep["families"] if f["family"] == "oscillator_rsi")
    check("dedup: retained family collapses flat+hierarchical to rep 112",
          osc["rung6_retained_count"] == 2 and osc["unique_retained_representatives"] == 1
          and osc["retained_representatives"] == [112], "%s" % osc["retained_representatives"])


# ── leakage (static: the aggregator cannot touch bar store / OOS / training) ────────────────────────
def _code_only(src):
    """Source with docstrings/comments/string-literals removed, so we scan CODE, not documentation.

    The aggregator's docstring deliberately says it 'NEVER opens liora.duckdb / reads OOS' — that word
    lives in a string, not in a call. We tokenize and drop COMMENT + STRING tokens before scanning."""
    import io
    import tokenize
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


def test_leakage():
    src = (ROOT / "scripts" / "family_transfer.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"duckdb", "sqlite3", "pipeline", "model", "torch", "xgboost", "optuna", "bars",
                 "asset_writers", "feature_search", "sklearn", "pandas", "numpy"}
    hit = imported & forbidden
    check("leakage: no bar-store / model-training import", not hit, "found %s" % sorted(hit))

    code = _code_only(src).lower()
    for term in ("duckdb", ".connect", ".execute", "read_parquet", "read_sql"):
        check("leakage: no %s in code (docstrings excluded)" % term, term not in code)
    check("leakage: no OOS read in code", "oos" not in code)
    # the science reader it DOES import is the read-only verdict layer
    check("leakage: imports only the read-only rung5_verdict science layer",
          "rung5_verdict" in imported and not (imported & forbidden))


# ── snapshot parity ─────────────────────────────────────────────────────────────────────────────────
def test_parity():
    rep = FT.build(SNAPSHOT, source_label="snapshot")
    fn = rep["funnel"]
    got = [fn["provisional_crossfit"], fn["passed_a1_marginal"], fn["stable_a1_a2_b"],
           fn["retained_rung6"], fn["unique_retained_representatives"]]
    check("snapshot parity: 26 → 11 → 9 → 2 → 1", got == [26, 11, 9, 2, 1], "got %s" % got)


def main():
    print("Family Transfer self-test\n")
    for group in (test_registry, test_determinism, test_fail_closed, test_dedup, test_leakage, test_parity):
        print("· %s" % group.__name__)
        group()
    n = len(_results)
    ok = sum(1 for _, o, _ in _results if o)
    print("\n%d/%d checks passed." % (ok, n))
    return 0 if ok == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
