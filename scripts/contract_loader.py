#!/usr/bin/env python3
"""The contract split into ten concern-scoped files, each tagged with when it may move and what it
breaks — and a loader that assembles them back, so nothing that reads the contract has to change.

The single 22 KB `feature_discovery_contract.json` was hard to reason about in one respect the owner
cares about: it said WHAT each value is but not WHEN it is allowed to change or WHAT goes stale when
it does. Splitting it by concern makes that explicit. Each file carries a `_meta` block with:

  variable_class   structural (frozen at Rung 0) | calibratable (data-fitted ranges) |
                   searched (the objects the ladder looks for) | methodology_governing (how much
                   proof is enough) | mixed
  rung             which rung(s) the file governs
  affects          what reads these values
  invalidates      which artifacts must be recomputed if this file changes

The split files are the SSOT. `feature_discovery_contract.json` becomes a GENERATED view assembled
from them, so `procedure_null` and `run_manifest` keep reading the exact same values they always did
— the split changes where a human EDITS, not what the code READS. `contract_lint.py` proves the two
stay in step and that no hardcoded constant has drifted from its file.

    python3 scripts/contract_loader.py --split        # monolith -> ten tagged files (one-time)
    python3 scripts/contract_loader.py --regenerate   # ten files -> monolith (after editing a split)
    from contract_loader import load; c = load()       # assemble in memory
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
MONOLITH = CONFIG / "feature_discovery_contract.json"
SPLIT_DIR = CONFIG / "contract"

# Meta fields the monolith carries at the top level; replicated into data_contract as the anchor.
_META_KEYS = ("schema_version", "methodology", "_status", "_purpose")

# Each entry: the monolith sections it owns, and its _meta tags. Order of the list defines assembly.
SPLIT_MAP = [
    ("data_contract.json", ["identity"], {
        "variable_class": "structural", "rung": 0,
        "affects": ["every rung — this file is the problem's identity"],
        "invalidates": ["ALL artifacts if the bar-store or sample hash changes"]}),
    ("label_contract.json", ["data_boundary"], {
        "variable_class": "structural", "rung": 0,
        "affects": ["event generation", "purge and embargo", "every fold boundary"],
        "invalidates": ["ALL artifacts — the label geometry defines the problem being studied"]}),
    ("model_space.json", ["model_space", "viability"], {
        "variable_class": "mixed (structural pointer + calibratable hpo_trials + governing viability floor)",
        "rung": [1, 3],
        "affects": ["model_viability", "feature_utility", "every trained booster"],
        "invalidates": ["model_viability.json", "feature_utility.json", "nested_*.json",
                        "crossfit_selection.json", "procedure_null_*.json", "rung6_survivor_hpo.json"]}),
    ("operating_space.json", ["operating_point"], {
        "variable_class": "mixed (structural mode + calibratable q-grid)", "rung": 2,
        "affects": ["the operating point in every rung that trades"],
        "invalidates": ["nested_*.json", "crossfit_selection.json", "procedure_null_*.json"]}),
    ("discovery_contract.json",
     ["arms", "cross_fitting", "acceptance", "stop_conditions", "rung_6_survivor_hpo",
      "rung_7_interactions"], {
        "variable_class": "methodology_governing", "rung": [3, 4, 6, 7],
        "affects": ["crossfit_selection", "procedure_null (acceptance.T)", "rung6", "rung7"],
        "invalidates": ["crossfit_selection.json", "procedure_null_*.json", "rung6_survivor_hpo.json"]}),
    ("multiplicity_contract.json", ["max_null", "rotation_level_null"], {
        "variable_class": "methodology_governing", "rung": 5,
        "affects": ["procedure_null"],
        "invalidates": ["procedure_null_*.json", "rung5_summary.json", "cross_asset_matrix.json",
                        "null_controls.json", "compiled/*.json"]}),
    ("compute_budget.json",
     ["runtime", "cost_model_core_seconds_per_ticker", "run_manifest_required_fields"], {
        "variable_class": "mixed (structural seed/threads + calibratable workers)", "rung": "infra",
        "affects": ["every run's parallelism, determinism and provenance"],
        "invalidates": ["cost_report.json"]}),
    ("certification_contract.json", ["certification"], {
        "variable_class": "methodology_governing / structural", "rung": 9,
        "affects": ["the certification run"], "invalidates": []}),
]

# Two reference pointers: the actual content lives in existing config files, not the monolith. They
# are part of the dependency graph, tagged, but contribute no sections to the assembled contract.
POINTERS = [
    ("feature_registry.json", "feature_namespaces_xgb.json", {
        "variable_class": "structural + searched", "rung": 3,
        "affects": ["which features exist to be searched"],
        "invalidates": ["feature_utility.json and everything downstream if a feature is added/removed"]}),
    ("feature_families.json", "feature_families_xgb.json", {
        "variable_class": "structural + searched", "rung": [3, 4, 8],
        "affects": ["family grouping — the hierarchical arm's stability unit and Rung 8"],
        "invalidates": ["crossfit_selection.json (hierarchical arm)", "cross_asset_matrix.json"]}),
]


def split():
    """Partition the monolith into the ten tagged files. One-time; idempotent."""
    mono = json.loads(MONOLITH.read_text(encoding="utf-8"))
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    meta_anchor = {k: mono[k] for k in _META_KEYS if k in mono}
    written = []
    for fname, sections, tags in SPLIT_MAP:
        body = {"_meta": {**tags, "sections": sections,
                          "_rule": "edit HERE, then run contract_loader.py --regenerate"}}
        if fname == "data_contract.json":
            body["_contract"] = meta_anchor            # the shared header lives with the identity
        for s in sections:
            if s in mono:
                body[s] = mono[s]
        (SPLIT_DIR / fname).write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
        written.append(fname)
    for fname, src, tags in POINTERS:
        src_path = CONFIG / src
        body = {"_meta": {**tags, "_rule": "reference pointer — the content lives in source_file"},
                "source_file": f"config/{src}",
                "exists": src_path.exists()}
        (SPLIT_DIR / fname).write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
        written.append(fname)
    return written


def assemble():
    """Read the split files and reconstruct the contract dict — values only, no _meta."""
    out = {}
    anchor = json.loads((SPLIT_DIR / "data_contract.json").read_text(encoding="utf-8"))
    out.update(anchor.get("_contract", {}))
    for fname, sections, _ in SPLIT_MAP:
        body = json.loads((SPLIT_DIR / fname).read_text(encoding="utf-8"))
        for s in sections:
            if s in body:
                out[s] = body[s]
    return out


def load():
    """The assembled contract dict, from the split SSOT."""
    return assemble()


def regenerate():
    """Write feature_discovery_contract.json from the split files. The monolith is a generated view;
    the split files are authoritative."""
    from artifact_io import write_json_atomic
    assembled = assemble()
    assembled["_generated"] = "assembled from config/contract/*.json — DO NOT EDIT; edit the splits"
    return write_json_atomic(MONOLITH, assembled)


def dependency_graph():
    """The affects/invalidates edges, for contract_lint and for a human reading what breaks what."""
    g = {}
    for fname, sections, tags in SPLIT_MAP:
        g[fname] = {"rung": tags["rung"], "variable_class": tags["variable_class"],
                    "sections": sections, "affects": tags["affects"],
                    "invalidates": tags["invalidates"]}
    for fname, src, tags in POINTERS:
        g[fname] = {"rung": tags["rung"], "variable_class": tags["variable_class"],
                    "source_file": f"config/{src}", "affects": tags["affects"],
                    "invalidates": tags["invalidates"]}
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--graph", action="store_true")
    args = ap.parse_args()
    if args.split:
        w = split()
        print(f"zapisano {len(w)} plików do {SPLIT_DIR}/:")
        for f in w:
            print(f"  {f}")
    if args.regenerate:
        sha = regenerate()
        print(f"zregenerowano {MONOLITH.name} z podziału  sha256 {sha[:16]}…")
    if args.graph:
        print(json.dumps(dependency_graph(), indent=1, ensure_ascii=False))
    if not (args.split or args.regenerate or args.graph):
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
