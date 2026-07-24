#!/usr/bin/env python3
"""Seal-lint for data_journey.html — the guard that the built data-journey board still tells the truth.

data_journey.html is a BUILT artifact carrying an embedded DATA-JOURNEY-SEAL: the contract hash, the
data boundary (train_end / oos_start), oos_reads, the arm funnel [26,11,9,2], the asset funnel
[20,12,7,5,1] and the assets shown. If the contract or the snapshot changes and the HTML is not rebuilt,
that seal lies silently. This makes the lie LOUD: it recomputes the canonical seal from the frozen
contract + snapshot and fails unless that exact seal is embedded verbatim in the HTML. It computes
nothing scientific — it reads, recomputes the seal, and compares. It reuses build_data_journey's own
reconstruct()/seal()/_seal_line(), so the lint and the generator can never diverge.

    python3 scripts/verify_data_journey.py           # verify; exit 1 on drift
    python3 scripts/verify_data_journey.py --emit      # print the current seal line to paste / to diff
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_data_journey as B                                               # noqa: E402


def main():
    line = B._seal_line(B.seal(B.reconstruct()))
    if "--emit" in sys.argv:
        print(line)
        return 0
    print("verify-data-journey — pieczęć data_journey.html vs bieżący kontrakt + snapshot\n")
    ok = True
    if not B.OUT.exists():
        print(f"  BRAK    {B.OUT.name} — uruchom `make data-journey`")
        ok = False
    elif line in B.OUT.read_text(encoding="utf-8"):
        print(f"  OK      {B.OUT.name}: pieczęć aktualna")
    else:
        print(f"  STALE   {B.OUT.name}: pieczęć NIE zgadza się z kontraktem/snapshotem — "
              f"przebuduj: make data-journey")
        ok = False
    if not ok:
        print(f"\n  Oczekiwana pieczęć:\n  {line}")
    print("\n" + ("PIECZĘĆ ZGODNA" if ok else "PIECZĘĆ NIEAKTUALNA"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
