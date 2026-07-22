#!/usr/bin/env python3
"""The methodology ledger — a copy of each decision, so the run records not only what was computed
but WHY the next experiment was run.

Every entry mirrors a verdict the planner derived from a result artifact: the question the rung
answered, the statistic, the verdict, the stop reason, and the next allowed experiment. Like the
execution ledger it is audit, not authority — the decision is re-derivable from `artifact + contract`
at any time, and if the two ever disagree, the artifact and contract win. Its value is legibility:
a human can read the sequence of scientific decisions without re-running the classifier.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ledger import Ledger                                                  # noqa: E402


class MethodLedger:
    def __init__(self, run_dir):
        self._led = Ledger(Path(run_dir) / "method_ledger.jsonl")

    def record(self, asset, rung, question, verdict, stop_reason, next_step, statistic=None):
        self._led.append("method", {"asset": asset, "rung": int(rung)}, "completed", payload={
            "question": question, "verdict": verdict, "stop_reason": stop_reason,
            "next_step": next_step, "statistic": statistic or {}})

    def verify(self):
        return self._led.verify_chain()

    def read_all(self):
        return self._led.read_all()
