#!/usr/bin/env python3
"""Keep the split contract honest: every file tagged, no hardcoded constant drifted from it, the
generated monolith in step, and — given a diff — the artifacts that just went stale named.

The split into ten files only helps if the split stays true. Three checks make it so, and a fourth
turns the dependency graph into an operational tool.

  tags      every split file declares variable_class, rung, affects, invalidates — so a reader
            always knows when a value may move and what breaks when it does.
  round     the generated monolith equals assemble(splits), so nothing that reads the monolith is
            reading something the splits no longer say.
  drift     the constants still hardcoded in code (acceptance thresholds, the viability floor, a few
            null scalars) equal their contract values. This is the guard that lets the contract be
            the SSOT even where the code has not been rewired to read it: if the two disagree, lint
            fails rather than letting a silent divergence stand.
  stale     given `--since <ref>`, which split files changed, and therefore which artifacts must be
            recomputed — read straight off each file's `invalidates` edge.

    python3 scripts/contract_lint.py
    python3 scripts/contract_lint.py --since HEAD~1     # what a recent contract edit invalidated
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
SPLIT_DIR = CONFIG / "contract"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "xgb" / "tools"))
sys.path.insert(0, str(ROOT / "xgb" / "src"))
import runtime_init  # noqa: E402,F401
runtime_init.apply()
import contract_loader as CL                                               # noqa: E402

REQUIRED_TAGS = ("variable_class", "rung", "affects", "invalidates")


def check_tags():
    problems = []
    for f in sorted(SPLIT_DIR.glob("*.json")):
        meta = json.loads(f.read_text(encoding="utf-8")).get("_meta", {})
        missing = [t for t in REQUIRED_TAGS if t not in meta]
        if missing:
            problems.append(f"{f.name}: brak tagów {missing}")
    return problems


def check_round_trip():
    mono = json.loads(CL.MONOLITH.read_text(encoding="utf-8"))
    asm = CL.assemble()
    drop = {"_generated"}
    mk, ak = set(mono) - drop, set(asm) - drop
    if mk != ak:
        return [f"klucze monolitu != assemble(): {mk ^ ak}"]
    diff = [k for k in mk
            if json.dumps(mono[k], sort_keys=True) != json.dumps(asm[k], sort_keys=True)]
    return [f"sekcja {k} różni się między monolitem a podziałem" for k in diff]


def check_drift():
    """The load-bearing guard: hardcoded code constants must equal their contract values."""
    import acceptance as ACC
    import feature_utility as FU
    c = CL.assemble()
    acc = c["acceptance"]
    mn = c["max_null"]
    mv = c["viability"]
    vf_sn, vf_sd = FU.viability_floor()

    pairs = [
        ("acceptance.COMPLEXITY_PEN", ACC.COMPLEXITY_PEN, acc["complexity_penalty"]),
        ("acceptance.MIN_ROTATIONS", ACC.MIN_ROTATIONS, acc["min_rotations"]),
        ("viability_floor.min_split_nodes", vf_sn, mv["min_split_nodes"]),
        ("viability_floor.min_pred_std", vf_sd, mv["min_pred_std"]),
        ("max_null.min_displaced_fraction",
         _pn_const("MIN_DISPLACED"), mn["permutation_audit"]["min_displaced_fraction"]),
        ("max_null.a2.segments", _pn_const("G_SEGMENTS"),
         mn["null_constructions"]["a2"]["segments"]),
        ("max_null.permutations_max", _pn_const("M_MAX"), mn["permutations_max"]),
        ("max_null.block_length.value_bars", _pn_const("L_BLOCK"), mn["block_length"]["value_bars"]),
        ("runtime.seed", _pn_const("SEED"), c["runtime"]["seed"]),
    ]
    problems = []
    for name, code_val, contract_val in pairs:
        if code_val is None:
            problems.append(f"{name}: nie udało się odczytać stałej z kodu")
        elif code_val != contract_val:
            problems.append(f"{name}: kod={code_val} != kontrakt={contract_val} (DRYF)")
    return problems


def _pn_const(name):
    import procedure_null as PN
    return getattr(PN, name, None)


def stale_report(since):
    """Which split files a diff touched, and what each one's `invalidates` edge says is now stale."""
    try:
        out = subprocess.run(("git", "-C", str(ROOT), "diff", "--name-only", since, "--",
                              "config/contract"), capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError) as e:                     # noqa: BLE001
        return {"error": str(e)}
    changed = [Path(p).name for p in out.split() if p.strip()]
    graph = CL.dependency_graph()
    return {f: graph.get(f, {}).get("invalidates", []) for f in changed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    args = ap.parse_args()

    checks = [("tagi na plikach", check_tags()),
              ("monolit == assemble(podział)", check_round_trip()),
              ("brak dryfu kod<->kontrakt", check_drift())]
    ok = True
    print("contract_lint — spójność podzielonego kontraktu\n")
    for name, problems in checks:
        mark = "OK  " if not problems else "FAIL"
        print(f"  {mark}  {name}")
        for p in problems:
            print(f"        - {p}")
        ok = ok and not problems

    if args.since:
        print(f"\nartefakty nieważne od {args.since}:")
        rep = stale_report(args.since)
        if not rep:
            print("  (żaden plik kontraktu nie zmieniony)")
        for f, inv in rep.items():
            print(f"  {f} -> {inv}")

    print(f"\n{'WSZYSTKO SPÓJNE' if ok else 'NIESPÓJNE — patrz FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
