#!/usr/bin/env python3
"""The safety kernel of the Iterative Calibration Loop — the thing that makes autonomy honest.

The loop is allowed to walk a human-pre-authorized *ladder* of frozen contract versions on its own,
epoch by epoch, and stop when a new version adds no confirmed features. That autonomy is only
defensible under one invariant:

    the proof standard is FROZEN across the whole ladder; only the hypothesis space may vary.

A ladder rung may propose a different *hypothesis* — a different model search space, a different
operating point, different arms — but it may never touch the standard by which a feature is judged
real: viability thresholds, the acceptance rule, the cross-fitting protocol, the null / multiplicity
contract, the data boundary, or certification. `apply()` enforces this mechanically: a patch that
reaches into any frozen key is REJECTED, so even a mis-declared ladder cannot quietly loosen the
science. A negative result therefore never triggers loosening — it triggers "advance to the next
pre-declared hypothesis space", or stop.

Each admitted version is re-hashed from its own patched contract, so it is a genuinely distinct,
self-consistent frozen snapshot (`contract_hash == sha256(embedded contract)`), addressable and
auditable exactly like a hand-minted one. The engine still never invents a rung: the ladder is a
finite, human-authored list in `config/iteration_loop_policy.json`.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))
import contract as CT                                                       # noqa: E402
import contract_loader as CL                                               # noqa: E402
import runtime_init                                                        # noqa: E402
from artifact_io import sha256_of, write_json_atomic                        # noqa: E402

# A ladder patch may vary ONLY these top-level keys of the assembled contract — the hypothesis space.
ADMISSIBLE = {"model_space", "operating_point", "arms",
              "rung_6_survivor_hpo", "rung_7_interactions"}

# It may NEVER touch these — the frozen proof standard. Loosening any of them = a different science,
# which needs a human, a new contract version by hand, and a new run_id (never a loop side effect).
FROZEN = {"viability", "acceptance", "cross_fitting", "stop_conditions",
          "max_null", "rotation_level_null", "data_boundary", "certification"}

# Everything else (schema_version, methodology, identity, runtime, cost_model_*, _status, ...) is
# provenance / infrastructure — also off-limits to a hypothesis patch, caught by the allowlist below.


class PatchRejected(RuntimeError):
    """A ladder patch tried to touch something outside the admissible hypothesis space."""


def _hash(assembled):
    """The contract hash, computed exactly as contract.contract_fingerprint does (so a base version
    produced here is bit-identical to one snapshotted by contract.snapshot)."""
    import hashlib
    payload = json.dumps(assembled, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _deep_merge(base, patch, path=""):
    """Return base with patch overlaid. Dicts merge recursively; any other value replaces. Adding a
    key that is absent from base is allowed only below the top level — a patch cannot introduce a new
    top-level section (that is a new contract, not a variation of this one)."""
    out = dict(base)
    for k, v in patch.items():
        here = f"{path}.{k}" if path else k
        if not path and k not in base:
            raise PatchRejected(f"patch wprowadza nowy klucz najwyższego poziomu {here!r} — "
                                f"to nowy kontrakt, nie wariant tego")
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _deep_merge(base[k], v, here)
        else:
            out[k] = v
    return out


def guard(patch):
    """Reject a patch that reaches beyond the admissible hypothesis space. Returns the set of touched
    top-level keys on success; raises PatchRejected otherwise. This is the single chokepoint; every
    version the loop mints passes through it."""
    if not isinstance(patch, dict):
        raise PatchRejected(f"patch musi być obiektem, jest {type(patch).__name__}")
    touched = set(patch.keys())
    frozen_hit = touched & FROZEN
    if frozen_hit:
        raise PatchRejected(f"patch narusza ZAMROŻONY standard dowodu: {sorted(frozen_hit)} — "
                            f"luźniejszych kryteriów nie wolno wprowadzać automatycznie")
    outside = touched - ADMISSIBLE
    if outside:
        raise PatchRejected(f"patch poza dopuszczalną przestrzenią hipotez: {sorted(outside)} "
                            f"(dozwolone: {sorted(ADMISSIBLE)})")
    return touched


def apply(patch, base=None):
    """Produce a patched, guarded, self-consistently hashed assembled contract from a ladder patch.

    Returns (patched_contract, contract_hash, touched_keys). An empty patch reproduces the base
    contract and its canonical hash exactly, so the 'base' rung of a ladder is not a special case."""
    base = base if base is not None else CL.assemble()
    touched = guard(patch or {})
    patched = _deep_merge(base, patch or {})
    return patched, _hash(patched), touched


def snapshot_version(run_dir, assets, version_id, patch=None, seed=42, allow_dirty=False,
                     base_hash=None):
    """Freeze one ladder rung into runs/<epoch_id>/contract.json.

    Mirrors contract.snapshot, but embeds the *patched* contract and hashes THAT, so the snapshot is
    self-consistent regardless of the patch. Extra provenance ties the epoch to its ladder position
    and to the base it derives from, and records the exact patch that produced it — a frozen document
    that shows its own variation rather than hiding it.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dirty = bool(CT._git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("drzewo brudne — snapshot wersji niereprodukowalny; "
                           "użyj allow_dirty tylko w trybie development")
    base = CL.assemble()
    patched, chash, touched = apply(patch or {}, base)
    snap = {
        "run_id": run_dir.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "contract_hash": chash,
        "execution_head_sha": CT._git("rev-parse", "HEAD"),
        "code_dirty": dirty,
        "bar_store_sha256_prefix": sha256_of(CT.BARS)[:16] if CT.BARS.exists() else None,
        "sample_sha256_prefix": sha256_of(CT.SAMPLE)[:16] if CT.SAMPLE.exists() else None,
        "environment": runtime_init.env_report(),
        "data_boundary": patched.get("data_boundary"),
        "assets": list(assets),
        "seed": int(seed),
        "contract": patched,
        "iteration_version_id": version_id,
        "iteration_patch": patch or {},
        "iteration_patch_touched": sorted(touched),
        "iteration_base_contract_hash": base_hash if base_hash is not None else _hash(base),
        "_rule": "wersja drabiny: standard dowodu zamrożony, zmienia się tylko przestrzeń hipotez "
                 f"({sorted(touched) or 'baza — bez zmian'}). Nowe reguły dowodu = ręczny kontrakt.",
    }
    write_json_atomic(run_dir / "contract.json", snap)
    return snap


def self_consistent(snap):
    """The immutability check that works for base AND patched epochs: the declared hash must equal the
    hash of the snapshot's own embedded contract. (contract.matches_current compares to the *current*
    on-disk config, which a patched epoch deliberately differs from — so it is the wrong check here.)"""
    return _hash(snap["contract"]) == snap["contract_hash"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Guarded contract-version patcher (safety kernel).")
    ap.add_argument("--patch", help="ścieżka do JSON z patchem, albo '-' dla stdin")
    ap.add_argument("--check-only", action="store_true", help="tylko sprawdź guard, nie hashuj")
    args = ap.parse_args()
    patch = {}
    if args.patch:
        text = sys.stdin.read() if args.patch == "-" else Path(args.patch).read_text(encoding="utf-8")
        patch = json.loads(text)
    try:
        touched = guard(patch)
    except PatchRejected as e:
        print(f"ODRZUCONY: {e}")
        return 1
    if args.check_only:
        print(f"OK: patch dotyka {sorted(touched) or '∅'} — w dopuszczalnej przestrzeni hipotez")
        return 0
    _patched, chash, touched = apply(patch)
    print(f"OK: touched={sorted(touched) or '∅'}  contract_hash={chash[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
