#!/usr/bin/env python3
"""Map one rung to an invocation of the EXISTING science runner for one asset, and pull that asset's
result back out. This is the whole adapter between the execution layer and the science layer.

The runners in `scripts/` are unchanged in their statistics; the only IO change is that they honour
`LIORA_RESEARCH_DATA_DIR` (default: the canonical `xgb/data`). Dispatch points them at the run's
private workspace, so a run's panel registers never touch the canonical tree or another run's. Bars
and the parquet scratch stay canonical (read-only, deterministic). Each runner writes a panel JSON
keyed by ticker into a throwaway file; dispatch extracts that asset's entry as the per-asset result.
No science runner learns anything about the queue, the state machine, or the contract snapshot.

The executable path is 1 -> 3 -> 4 -> 5 -> 6 (Rung 2's operating-point check is folded into 3-4).
Cheap rungs (1, 3, 4) are one runner call plus an extract. Rung 5 is three null calls (a1, then a2/b
on that asset's survivors) collapsed into one per-asset artifact; the runner keeps its own internal
fold/permutation ledger. Rung 6 runs the survivor-HPO runner against a one-asset survivors file.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
PY = str(ROOT / ".venv" / "bin" / "python3")
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import reducer as RD                                                        # noqa: E402
import rung5_verdict as RV                                                  # noqa: E402 — one definition of "stable"


def _run(cmd, timeout, ws):
    # PYTHONHASHSEED fixed so any set->list ordering in a runner's result is process-independent — a
    # runner's published bytes must be reproducible across re-runs, and str-set iteration order is the
    # one source of per-process variation that sort_keys (dict keys only) does not canonicalize.
    env = dict(os.environ, LIORA_RESEARCH_DATA_DIR=str(ws), PYTHONHASHSEED="0")
    # Point the runners at THIS run's frozen contract so a ladder epoch's admissible hypothesis
    # (operating_point grid, model_space) actually reaches the science. ws = run_dir/workspace/xgb/data,
    # so the snapshot is ws.parents[2]/contract.json. Absent -> runners keep their canonical default.
    contract = ws.parents[2] / "contract.json"
    if contract.exists():
        env["RESEARCH_CONTRACT"] = str(contract)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT), env=env)
    return p.returncode, p.stdout, p.stderr


def _tmp(ws, suffix=".json"):
    ws.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, dir=str(ws)) as tf:
        return tf.name


def _entry(out_path, asset):
    """Extract one asset's SCIENCE result. A sealed artifact holds only science; operational fields
    (timing, host, pid) live in the ledger and run_manifest — so `seconds` is dropped here. The hashed
    result must be byte-identical across re-runs, and wall-time never is; the per-unit cost it duplicated
    is already in the exec ledger (`per-unit cost for cost_report.py`)."""
    doc = json.loads(Path(out_path).read_text(encoding="utf-8"))
    e = doc.get("tables", {}).get(asset)
    if isinstance(e, dict):
        e.pop("seconds", None)          # operational, not science → never in the hashed seal
    return e


_SINGLE = {1: (SCRIPTS / "model_viability.py", 1800),
           3: (SCRIPTS / "feature_utility.py", 3600),
           4: (SCRIPTS / "crossfit_selection.py", 3600)}


def _dispatch_single(task, ws):
    script, timeout = _SINGLE[task["rung"]]
    asset, tmp = task["asset"], _tmp(ws)
    try:
        rc, _o, err = _run([PY, str(script), asset, "--out", tmp], timeout, ws)
        if rc != 0:
            return None, rc, err[-400:]
        entry = _entry(tmp, asset)
        return (entry, 0, "") if entry is not None else (None, 91, f"brak wpisu {asset}")
    finally:
        Path(tmp).unlink(missing_ok=True)


def _smoke_args():
    """Reduced null strength for a FAST dev gate: RESEARCH_SMOKE_PERMS/FOLDS -> --permutations/--folds.
    A run capped here records a low permutations_max in its artifact, and the confirmation guardrail
    (rung5_verdict.full_strength) then refuses to count it — a smoke validates the ORCHESTRATION, never
    the science. Unset -> full strength, so a real run is untouched."""
    a = []
    p, f = os.environ.get("RESEARCH_SMOKE_PERMS"), os.environ.get("RESEARCH_SMOKE_FOLDS")
    if p:
        a += ["--permutations", str(int(p))]
    if f:
        a += ["--folds", str(int(f))]
    return a


def _fold_args():
    """Within-ticker fold parallelism (#2, fork-after-prepare, byte-identical). RESEARCH_FOLD_JOBS ->
    --fold-jobs; unset -> 1 (serial). The engine dispatches one ticker per null task (--jobs 1), so
    the task's cores go to its folds. Byte-identical to serial by construction (folds independent)."""
    fj = os.environ.get("RESEARCH_FOLD_JOBS")
    return ["--fold-jobs", str(int(fj))] if fj else []


def _dispatch_null(task, ws):
    asset = task["asset"]
    scratch = ws / "scratch" / f"null_{asset}_{int(time.time())}"
    extra = _smoke_args() + _fold_args()
    tmps = {}
    try:
        tmps["a1"] = _tmp(ws)
        rc, _o, err = _run([PY, str(SCRIPTS / "procedure_null.py"), asset, "--null", "a1",
                            "--jobs", "1", "--out", tmps["a1"], "--run-dir", str(scratch / "a1")]
                           + extra, 5400, ws)
        if rc != 0:
            return None, rc, err[-400:]
        a1 = _entry(tmps["a1"], asset)
        result = {"a1": a1, "a2": None, "b": None}
        # ORCHESTRATION gate (which arms get a2/b), not a science verdict: accept smoke_pass too so a
        # reduced smoke still exercises a2/b. The SCIENCE selector (rung5_verdict.passed_arms) stays
        # strict "passed", so a smoke never reaches NULL_VALIDATED — only real strength confirms.
        passed = a1 and any(v.get("verdict") in ("passed", "smoke_pass")
                            for f in a1.get("folds", []) for v in f.get("arms", {}).values())
        for kind in ("a2", "b"):
            if not passed:
                continue
            tmps[kind] = _tmp(ws)
            rc, _o, err = _run([PY, str(SCRIPTS / "procedure_null.py"), asset, "--null", kind,
                                "--jobs", "1", "--survivors-from", tmps["a1"], "--out", tmps[kind],
                                "--run-dir", str(scratch / kind)] + extra, 5400, ws)
            if rc != 0:
                return None, rc, f"{kind}: {err[-300:]}"
            result[kind] = _entry(tmps[kind], asset)
        return result, 0, ""
    finally:
        for p in tmps.values():
            Path(p).unlink(missing_ok=True)
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)


def _panel_entry(path, asset):
    """One asset's entry from a null panel in the workspace, or None (file/asset absent). A2/B are
    absent when the asset never reached them (a1 produced no survivor) — None then makes
    stable_survivors treat that leg as failed, which is the correct fail-closed reading."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")).get("tables", {}).get(asset)
    except (OSError, json.JSONDecodeError):
        return None


def _stable_entry(a1_entry, stable):
    """The a1 entry filtered to ONLY its stable arms (A1∩A2∩B), each kept `passed` — the survivors
    file Rung 6 reads. rung6.survivors() then extracts exactly the stable arms, so Rung 6 optimizes
    the intersection, never A1 alone (else it would 'retain' features A2/B already rejected)."""
    folds = []
    for f in a1_entry.get("folds", []):
        of = f.get("outer_fold")
        arms = {arm: v for arm, v in (f.get("arms", {}) or {}).items()
                if v.get("verdict") == "passed" and (of, arm, str(v.get("unit"))) in stable}
        if arms:
            folds.append({**f, "arms": arms})
    return {**a1_entry, "folds": folds}


def _dispatch_hpo(task, ws):
    asset = task["asset"]
    a1e = _panel_entry(ws / "procedure_null_a1.json", asset)  # assembled by the reducer into the workspace
    if a1e is None:
        return None, 93, "brak panelu/wpisu procedure_null_a1.json w workspace (reducer nie złożył)"
    # Rung 6 optimizes ONLY the stable survivors A1∩A2∩B (canonical rung5_verdict.stable_survivors,
    # fail-closed when a2/b absent) — feeding the A1 panel alone lets Rung 6 retain arms A2/B rejected.
    a2e, be = _panel_entry(ws / "procedure_null_a2.json", asset), _panel_entry(ws / "procedure_null_b.json", asset)
    stable = RV.stable_survivors(a1e, a2e, be)
    if not stable:
        return {"results": [], "retained": 0, "demoted": 0, "budget_B": None}, 0, ""
    one, out = _tmp(ws), _tmp(ws)
    scratch = ws / "scratch" / f"hpo_{asset}_{int(time.time())}"
    try:
        Path(one).write_text(json.dumps({"contract": {"null": "a1a2b_stable"},
                                         "tables": {asset: _stable_entry(a1e, stable)}}))
        rc, _o, err = _run([PY, str(SCRIPTS / "rung6_survivor_hpo.py"), "--survivors-from", one,
                            "--jobs", "1", "--out", out, "--run-dir", str(scratch)], 5400, ws)
        if rc != 0:
            return None, rc, err[-400:]
        doc = json.loads(Path(out).read_text(encoding="utf-8"))
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


def dispatch(task, run_dir):
    """Run the rung's runner for one asset in the run's workspace; return (result|None, rc, stderr)."""
    fn = DISPATCH.get(task["rung"])
    if fn is None:
        return None, 90, f"rung {task['rung']} bez dispatchu"
    return fn(task, RD.workspace(run_dir))
