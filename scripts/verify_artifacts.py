#!/usr/bin/env python3
"""Verify the sealed artifact tree against artifacts/manifest.json — offline, stdlib only.

Do not take the numbers on trust: this recomputes them. For every asset folder it
checks each hashed file's size and SHA-256, rebuilds `folder_sha256` from those
digests, and confirms the manifest's own count arithmetic. It then reads
data/results.db (read-only) and confirms every artifact_path resolves to a folder
the manifest knows.

    python3 scripts/verify_artifacts.py          # or: make verify
    python3 scripts/verify_artifacts.py --quiet  # only the verdict

Exit code 0 when everything matches, 1 on the first class of mismatch found (all
mismatches are listed, not just the first one).

`folder_sha256` is built from the four hashed files of the folder — the folder's own
manifest.json is excluded, since it is what carries the digests:

    sha256( "\\n".join(f"{name}:{sha256}" for name in sorted(files)).encode("utf-8") )

with no trailing newline. Here it is rebuilt from the digests measured on disk, so a
changed file fails both its own hash and its folder hash.

What this does and does not prove: it proves the tree on disk is the tree the manifest
describes, and that the manifest is internally consistent. The manifest is not signed,
so it cannot prove the tree is the one the research run produced — for that, compare
against the epoch's own committed manifest in the research tree.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MANIFEST = ARTIFACTS / "manifest.json"
DB_PATH = ROOT / "data" / "results.db"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def folder_sha256(digests):
    """digests: {filename: sha256_hex} of the folder's four hashed files."""
    pairs = "\n".join(f"{name}:{digest}" for name, digest in sorted(digests.items()))
    return hashlib.sha256(pairs.encode("utf-8")).hexdigest()


def verify_folder(model, ticker, entry, problems):
    base = ARTIFACTS / model / ticker
    if not base.is_dir():
        problems.append(f"{model}/{ticker}: folder missing")
        return False
    ok = True
    measured = {}
    for name, meta in entry["files"].items():
        path = base / name
        if not path.is_file():
            problems.append(f"{model}/{ticker}/{name}: file missing")
            ok = False
            continue
        size = path.stat().st_size
        if size != meta["bytes"]:
            problems.append(f"{model}/{ticker}/{name}: {size} bytes, manifest says {meta['bytes']}")
            ok = False
        digest = sha256_file(path)
        measured[name] = digest
        if digest != meta["sha256"]:
            problems.append(f"{model}/{ticker}/{name}: sha256 {digest[:12]}…, "
                            f"manifest says {meta['sha256'][:12]}…")
            ok = False
    if not ok and len(measured) != len(entry["files"]):
        return False  # a missing file makes the folder hash meaningless
    recomputed = folder_sha256(measured)
    if recomputed != entry["folder_sha256"]:
        problems.append(f"{model}/{ticker}: folder_sha256 {recomputed[:12]}…, "
                        f"manifest says {entry['folder_sha256'][:12]}…")
        ok = False
    return ok


def verify_paths(manifest, problems):
    """Every artifact_path in the sealed store must resolve to a folder the manifest knows."""
    if not DB_PATH.is_file():
        problems.append("data/results.db missing — cannot cross-check artifact paths")
        return 0
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        rows = con.execute("select ticker, model, artifact_path from asset_results").fetchall()
    finally:
        con.close()
    for ticker, model, path in rows:
        if path != f"artifacts/{model}/{ticker}":
            problems.append(f"{model}/{ticker}: artifact_path is {path!r}")
        elif ticker not in manifest.get(model, {}):
            problems.append(f"{model}/{ticker}: sealed row points at a folder the manifest omits")
        elif not (ROOT / path).is_dir():
            problems.append(f"{model}/{ticker}: artifact_path does not resolve on disk")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true", help="print only the verdict")
    args = ap.parse_args()

    if not MANIFEST.is_file():
        print(f"FAIL: {MANIFEST.relative_to(ROOT)} not found — is the clone complete?")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    counts = manifest.get("_meta", {}).get("counts", {})
    problems, verified = [], 0

    for model in ("xgb", "lstm"):
        entries = manifest.get(model, {})
        declared = counts.get(model)
        if declared is not None and declared != len(entries):
            problems.append(f"{model}: manifest declares {declared} assets, lists {len(entries)}")
        for ticker, entry in sorted(entries.items()):
            if verify_folder(model, ticker, entry, problems):
                verified += 1
        if not args.quiet:
            print(f"  {model}: {len(entries)} folders checked")

    total = counts.get("total")
    summed = counts.get("xgb", 0) + counts.get("lstm", 0)
    if total is not None and total != summed:
        problems.append(f"_meta.counts.total is {total}, xgb + lstm is {summed}")

    n_rows = verify_paths(manifest, problems)
    if not args.quiet:
        print(f"  store: {n_rows} sealed rows cross-checked against the manifest")

    expected = total if total is not None else summed
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s) found\n")
        for p in problems[:50]:
            print(f"  - {p}")
        if len(problems) > 50:
            print(f"  … and {len(problems) - 50} more")
        return 1
    print(f"\nOK: {verified}/{expected} artifact folders verified "
          f"(per-file SHA-256 + folder_sha256), {n_rows} sealed rows resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
