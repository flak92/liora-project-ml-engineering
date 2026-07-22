#!/usr/bin/env python3
"""Map one rung to an invocation of the EXISTING science runner for one asset, and pull that asset's
result back out. This is the whole adapter between the execution layer and the science layer.

The runners in `scripts/` are unchanged: each already accepts a positional ticker and writes a panel
JSON keyed by ticker. Dispatch runs one for a single asset into a throwaway file, then extracts that
asset's entry as the per-asset result. No science runner learns anything about the queue, the state
machine, or the contract snapshot — the worker enforces the contract before dispatch, and the planner
owns the transitions. Rung 2's operating-point check is folded into Rungs 3–4 on this panel (the
quantile theta is chosen on discovery and applied to confirmation there), so there is no standalone
Rung 2 runner; the DAG executes 1 -> 3 -> 4 -> 5 -> 6.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python3")
SCRIPTS = ROOT / "scripts"


def _run(cmd, timeout):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    return p.returncode, p.stdout, p.stderr


def _extract_ticker(out_path, asset):
    """Pull one asset's entry from a panel artifact keyed by ticker under 'tables'."""
    doc = json.loads(Path(out_path).read_text(encoding="utf-8"))
    tables = doc.get("tables", {})
    return tables.get(asset)


# rung -> (runner, builder(asset, tmp)->cmd, extractor(tmp, asset)->result, timeout_s)
def _rung1(asset, tmp):
    return [PY, str(SCRIPTS / "model_viability.py"), asset, "--out", tmp]


def _rung3(asset, tmp):
    return [PY, str(SCRIPTS / "feature_utility.py"), asset, "--out", tmp]


def _rung4(asset, tmp):
    return [PY, str(SCRIPTS / "crossfit_selection.py"), asset, "--out", tmp]


RUNG_SPEC = {
    1: {"name": "viability", "build": _rung1, "extract": _extract_ticker, "timeout": 1800,
        "needs": []},
    3: {"name": "utility", "build": _rung3, "extract": _extract_ticker, "timeout": 3600,
        "needs": []},
    4: {"name": "crossfit", "build": _rung4, "extract": _extract_ticker, "timeout": 3600,
        "needs": ["feature_utility.json"]},
    # Rung 5 (procedure_null) and Rung 6 (survivor HPO) are wired in phase 2 — they are per-asset but
    # read panel inputs the reducer assembles, and Rung 5 keeps its own internal fold/perm ledger.
}


def dispatch(task):
    """Run the rung's runner for one asset; return (result_dict_or_None, exit_code, stderr_tail)."""
    rung = task["rung"]
    asset = task["asset"]
    spec = RUNG_SPEC.get(rung)
    if spec is None:
        return None, 90, f"rung {rung} nie ma jeszcze dispatchu"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=str(ROOT / "xgb" / "data")) as tf:
        tmp = tf.name
    try:
        cmd = spec["build"](asset, tmp)
        rc, _out, err = _run(cmd, spec["timeout"])
        if rc != 0:
            return None, rc, err[-400:]
        result = spec["extract"](tmp, asset)
        if result is None:
            return None, 91, f"runner nie wyprodukował wpisu dla {asset}"
        return result, 0, ""
    finally:
        Path(tmp).unlink(missing_ok=True)
