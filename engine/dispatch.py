#!/usr/bin/env python3
"""Map one rung to an invocation of the EXISTING science runner for one asset, and pull that asset's
result back out. This is the whole adapter between the execution layer and the science layer.

The runners in `scripts/` are unchanged: each accepts a positional ticker (or a survivors file) and
writes a JSON keyed by ticker. Dispatch runs one for a single asset into throwaway files, then
extracts that asset's entry as the per-asset result. No science runner learns anything about the
queue, the state machine, or the contract snapshot — the worker enforces the contract before
dispatch, the planner owns the transitions. The executable path is 1 -> 3 -> 4 -> 5 -> 6; Rung 2's
operating-point check is folded into Rungs 3-4 (the quantile theta is chosen on discovery and applied
to confirmation there), so there is no standalone Rung 2 runner.

Cheap rungs (1, 3, 4) are one runner call plus an extract. Rung 5 is three null calls (a1, then a2/b
on that asset's survivors) collapsed into one per-asset artifact; the runner keeps its own internal
fold/permutation ledger, so the finest granularity resumes without the top-level queue knowing about
it. Rung 6 runs the survivor-HPO runner against a one-asset survivors file, so no `--asset` flag has
to be added to the science code.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python3")
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "xgb" / "data"


def _run(cmd, timeout):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    return p.returncode, p.stdout, p.stderr


def _tmp(suffix):
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, dir=str(DATA)) as tf:
        return tf.name


def _entry(out_path, asset):
    doc = json.loads(Path(out_path).read_text(encoding="utf-8"))
    return doc.get("tables", {}).get(asset)


# ---- cheap single-runner rungs -----------------------------------------------------------------

_SINGLE = {
    1: (SCRIPTS / "model_viability.py", 1800),
    3: (SCRIPTS / "feature_utility.py", 3600),
    4: (SCRIPTS / "crossfit_selection.py", 3600),
}


def _dispatch_single(task):
    script, timeout = _SINGLE[task["rung"]]
    asset, tmp = task["asset"], _tmp(".json")
    try:
        rc, _o, err = _run([PY, str(script), asset, "--out", tmp], timeout)
        if rc != 0:
            return None, rc, err[-400:]
        entry = _entry(tmp, asset)
        return (entry, 0, "") if entry is not None else (None, 91, f"brak wpisu {asset}")
    finally:
        Path(tmp).unlink(missing_ok=True)


# ---- Rung 5: three nulls per asset --------------------------------------------------------------

def _dispatch_null(task):
    asset = task["asset"]
    scratch = DATA / "runs" / f"engine_null_{asset}_{int(time.time())}"
    tmps = {}
    try:
        # A1 (marginal) over this asset's cross-fit-accepted folds.
        tmps["a1"] = _tmp(".json")
        rc, _o, err = _run([PY, str(SCRIPTS / "procedure_null.py"), asset, "--null", "a1",
                            "--jobs", "1", "--out", tmps["a1"], "--run-dir", str(scratch / "a1")], 5400)
        if rc != 0:
            return None, rc, err[-400:]
        a1 = _entry(tmps["a1"], asset)
        result = {"a1": a1, "a2": None, "b": None}
        passed = a1 and any(v.get("verdict") == "passed"
                            for f in a1.get("folds", []) for v in f.get("arms", {}).values())
        # A2 (regime) and B (conditional) only where A1 accepted something — a sensitivity check.
        for kind in ("a2", "b"):
            if not passed:
                continue
            tmps[kind] = _tmp(".json")
            rc, _o, err = _run([PY, str(SCRIPTS / "procedure_null.py"), asset, "--null", kind,
                                "--jobs", "1", "--survivors-from", tmps["a1"], "--out", tmps[kind],
                                "--run-dir", str(scratch / kind)], 5400)
            if rc != 0:
                return None, rc, f"{kind}: {err[-300:]}"
            result[kind] = _entry(tmps[kind], asset)
        return result, 0, ""
    finally:
        for p in tmps.values():
            Path(p).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


# ---- Rung 6: survivor-HPO against a one-asset survivors file ------------------------------------

def _dispatch_hpo(task):
    asset = task["asset"]
    a1_panel = DATA / "procedure_null_a1.json"                # assembled by the reducer
    try:
        panel = json.loads(a1_panel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, 93, "brak panelu procedure_null_a1.json (reducer nie złożył)"
    entry = panel.get("tables", {}).get(asset)
    if entry is None:
        return None, 94, f"asset {asset} nieobecny w panelu A1"
    one = _tmp(".json")
    out = _tmp(".json")
    scratch = DATA / "runs" / f"engine_hpo_{asset}_{int(time.time())}"
    try:
        Path(one).write_text(json.dumps({"contract": {"null": "a1"}, "tables": {asset: entry}}))
        rc, _o, err = _run([PY, str(SCRIPTS / "rung6_survivor_hpo.py"), "--survivors-from", one,
                            "--jobs", "1", "--out", out, "--run-dir", str(scratch)], 5400)
        if rc != 0:
            return None, rc, err[-400:]
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
        # rung6 output is already restricted to this asset's survivors (the survivors file had one).
        return {"results": doc.get("results", []), "retained": doc.get("retained"),
                "demoted": doc.get("demoted"), "budget_B": doc.get("budget_B")}, 0, ""
    finally:
        for p in (one, out):
            Path(p).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


DISPATCH = {1: _dispatch_single, 3: _dispatch_single, 4: _dispatch_single,
            5: _dispatch_null, 6: _dispatch_hpo}

# What panel register each rung's runner reads as input — the planner gates on the asset being present.
NEEDS = {1: [], 3: [], 4: ["feature_utility.json"],
         5: ["crossfit_selection.json", "feature_utility.json"],
         6: ["procedure_null_a1.json", "crossfit_selection.json", "feature_utility.json"]}


def dispatch(task):
    """Run the rung's runner for one asset; return (result_dict_or_None, exit_code, stderr_tail)."""
    fn = DISPATCH.get(task["rung"])
    if fn is None:
        return None, 90, f"rung {task['rung']} bez dispatchu"
    return fn(task)
