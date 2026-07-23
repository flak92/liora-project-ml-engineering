#!/usr/bin/env python3
"""The single writer of panels. Workers only ever write their own immutable per-asset artifact; the
reducer is the one process that reads those and assembles the panel views — so there is no concurrent
write to a shared file and no `flock` on scientific results.

It serves two jobs. First, some science runners read a panel register as INPUT (cross-fit reads the
feature-utility register; the null reads the cross-fit register). The reducer rebuilds those panel
inputs under `xgb/data/` from the per-asset artifacts, so the next rung's runner finds what it needs
without the engine touching the runner's code. Second, it writes run-scoped panels under
`runs/<id>/results/panels/` for provenance and for the report.

Because it is the only writer and runs single-threaded (a planner pre-step / a scheduler cadence),
rebuilding is idempotent and cheap: it is always safe to run again, and a panel can always be
reconstructed from the per-asset artifacts that outlive it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
from artifact_io import write_json_atomic                                  # noqa: E402
import contract as CT                                                      # noqa: E402
import states as ST                                                        # noqa: E402


def workspace(run_dir, asset=None):
    """A run's private panel directory (via LIORA_RESEARCH_DATA_DIR), so a run's assembled panels never
    touch the canonical xgb/data or another run's. With `asset`, a PER-ASSET private workspace — each
    asset's registers live under workspace/<asset>/, so parallel-over-assets writes no shared file and
    cannot race (the class of bug the queue/guard existed to prevent). Without `asset`, the shared
    workspace (legacy queue-worker path). Bars/parquet scratch stay canonical (read-only, deterministic)."""
    d = Path(run_dir) / "workspace"
    if asset is not None:
        d = d / str(asset)
    d = d / "xgb" / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _asset_results(run_dir, rung):
    """{asset: result} for every asset with a valid artifact at this rung."""
    out = {}
    for a in CT.load(run_dir)["assets"]:
        art = ST.latest_artifact(run_dir, rung, a)
        if art is not None:
            out[a] = art["result"]
    return out


def assemble_feature_utility(run_dir):
    """xgb/data/feature_utility.json from per-asset Rung 3 artifacts — the register cross-fit reads."""
    snap = CT.load(run_dir)
    c = snap["contract"]
    tables = _asset_results(run_dir, 3)
    if not tables:
        return None
    doc = {"hpo_trials": c["model_space"]["hpo_trials"], "seed": snap.get("seed", 42),
           "viability": c["viability"], "operating_point": c["operating_point"],
           "_generated": "engine reducer from per-asset Rung 3 artifacts", "tables": tables}
    write_json_atomic(workspace(run_dir) / "feature_utility.json", doc)
    write_json_atomic(Path(run_dir) / "results" / "panels" / "feature_utility.json", doc)
    return len(tables)


def assemble_crossfit(run_dir):
    """xgb/data/crossfit_selection.json from per-asset Rung 4 artifacts — the register the null reads."""
    tables = _asset_results(run_dir, 4)
    if not tables:
        return None
    doc = {"contract": {"rule": "max-over-eligible", "_generated": "engine reducer"}, "tables": tables}
    write_json_atomic(workspace(run_dir) / "crossfit_selection.json", doc)
    write_json_atomic(Path(run_dir) / "results" / "panels" / "crossfit_selection.json", doc)
    return len(tables)


def assemble_null(run_dir, kind="a1"):
    """xgb/data/procedure_null_<kind>.json from per-asset Rung 5 artifacts (each holds a1/a2/b)."""
    src = _asset_results(run_dir, 5)
    tables = {a: r[kind] for a, r in src.items() if r.get(kind)}
    if not tables:
        return None
    doc = {"contract": {"null": kind, "_generated": "engine reducer"}, "tables": tables}
    write_json_atomic(workspace(run_dir) / f"procedure_null_{kind}.json", doc)
    write_json_atomic(Path(run_dir) / "results" / "panels" / f"procedure_null_{kind}.json", doc)
    return len(tables)


def assemble_inputs(run_dir):
    """Rebuild every panel a downstream runner reads. Idempotent; safe to call each planner cycle."""
    return {"feature_utility": assemble_feature_utility(run_dir),
            "crossfit_selection": assemble_crossfit(run_dir),
            "procedure_null_a1": assemble_null(run_dir, "a1"),
            "procedure_null_a2": assemble_null(run_dir, "a2"),
            "procedure_null_b": assemble_null(run_dir, "b")}


def assemble_asset_inputs(run_dir, asset):
    """Per-asset ONE-ROW registers into the asset's PRIVATE workspace, from that asset's OWN artifacts —
    the parallel-over-assets counterpart of assemble_inputs. The structure mirrors assemble_feature_utility
    / assemble_crossfit / assemble_null EXACTLY (same contract-derived fields, same `_generated`); only
    `tables` holds the single asset. A runner reads only tables[asset] plus the frozen contract-derived
    fields, so a one-row table is byte-identical INPUT to the shared-panel path — no other asset's row is
    ever read, so no shared write and no concurrency. Idempotent; called before each of the asset's rungs."""
    snap = CT.load(run_dir)
    c = snap["contract"]
    ws = workspace(run_dir, asset)
    a3 = ST.latest_artifact(run_dir, 3, asset)
    if a3 is not None:
        write_json_atomic(ws / "feature_utility.json", {
            "hpo_trials": c["model_space"]["hpo_trials"], "seed": snap.get("seed", 42),
            "viability": c["viability"], "operating_point": c["operating_point"],
            "_generated": "engine reducer from per-asset Rung 3 artifacts",
            "tables": {asset: a3["result"]}})
    a4 = ST.latest_artifact(run_dir, 4, asset)
    if a4 is not None:
        write_json_atomic(ws / "crossfit_selection.json", {
            "contract": {"rule": "max-over-eligible", "_generated": "engine reducer"},
            "tables": {asset: a4["result"]}})
    a5 = ST.latest_artifact(run_dir, 5, asset)
    if a5 is not None:
        for kind in ("a1", "a2", "b"):
            if a5["result"].get(kind):
                write_json_atomic(ws / f"procedure_null_{kind}.json", {
                    "contract": {"null": kind, "_generated": "engine reducer"},
                    "tables": {asset: a5["result"][kind]}})


def has_asset(panel_path, asset):
    """Is `asset` already present in a panel register? The planner's gate before a rung that needs it."""
    p = Path(panel_path)
    if not p.exists():
        return False
    try:
        return asset in json.loads(p.read_text(encoding="utf-8")).get("tables", {})
    except (json.JSONDecodeError, OSError):
        return False


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()
    print("reducer:", json.dumps(assemble_inputs(a.run_dir), ensure_ascii=False))
