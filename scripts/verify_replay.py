#!/usr/bin/env python3
"""Seal-lint for methodology_replay.html — the guard that the built replay still tells the truth.

methodology_replay.html is a BUILT artifact carrying an embedded REPLAY-SEAL: the contract hash, the
funnel [26, 11, 9, 2], the retained feature, and the LIVE guard's actual rejection message. If the
contract, the snapshot, or the guard changes and the HTML is not rebuilt, that seal lies silently.
This makes the lie LOUD: it recomputes the canonical seal from contract_loader.assemble() +
report.funnel() + contract_patch.guard(), and fails unless that exact seal is embedded verbatim in the
HTML. It computes nothing scientific — it reads, recomputes the seal, and compares. It reuses
build_replay's own reconstruct()/seal()/_seal_line(), so the lint and the generator can never diverge.

    python3 scripts/verify_replay.py           # verify; exit 1 on drift
    python3 scripts/verify_replay.py --emit      # print the current seal line to paste / to diff
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_replay as B                                                       # noqa: E402


def main():
    line = B._seal_line(B.seal(B.reconstruct()))
    if "--emit" in sys.argv:
        print(line)
        return 0
    print("verify-replay — pieczęć methodology_replay.html vs bieżący kontrakt + snapshot + guard\n")
    ok = True
    if not B.OUT.exists():
        print(f"  BRAK    {B.OUT.name} — uruchom `make replay`")
        ok = False
    elif line in B.OUT.read_text(encoding="utf-8"):
        print(f"  OK      {B.OUT.name}: pieczęć aktualna")
    else:
        print(f"  STALE   {B.OUT.name}: pieczęć NIE zgadza się z kontraktem/snapshotem/guardem — "
              f"przebuduj: make replay")
        ok = False
    if not ok:
        print(f"\n  Oczekiwana pieczęć:\n  {line}")
    print("\n" + ("PIECZĘĆ ZGODNA" if ok else "PIECZĘĆ NIEAKTUALNA"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
