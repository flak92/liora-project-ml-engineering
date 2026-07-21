#!/usr/bin/env python3
"""An append-only record of finished work, so a killed run resumes instead of restarting.

The methodology's remaining compute is roughly 28 core-hours. None of the runners in this tree can
survive an interruption: they accumulate every result in the parent process's memory and write one
JSON file after the last ticker returns. A run that dies at 95% has produced nothing.

This module is the missing half. Each worker appends one line the moment a unit of work is
finished; a resuming run reads the file and skips whatever is already there. The unit is chosen so
that losing one is cheap — for the procedure-level null it is a single
(ticker, outer fold, permutation), about 53 seconds.

Three properties are load-bearing:

**Append-only.** Nothing is ever rewritten or deleted, including failures. A crash is a result: if
a unit dies three times in a row that is something the run manifest should show, not something a
retry should quietly paper over.

**Crash-atomic per line.** Every append takes an exclusive `flock`, seeks to the end, writes one
line under 4096 bytes and fsyncs. Linux guarantees an O_APPEND write below PIPE_BUF is not
interleaved with another writer's, so four worker processes can share one ledger without a
coordinator. Records that would exceed that budget keep their bulky part in the artifact and only
the identity plus verdict in the ledger.

**Hash-chained.** Each line carries the checksum of the previous one, so a truncated tail is
detectable and a silently edited middle is not possible. `verify_chain()` is what makes the ledger
evidence rather than a log file.

The chain is read backwards, not by reloading the file: only the last 8 KiB is touched per append,
so appending 800 units costs 800 short reads rather than 800 full-file parses.

    led = Ledger(run_dir / "ledger.jsonl")
    done = led.completed("null_a1")                       # resume
    for unit in units:
        if led.key(unit) in done:
            continue
        led.append("null_a1", unit, "running")
        ...
        led.append("null_a1", unit, "completed", payload=result)
"""
import fcntl
import json
import os
import time
from pathlib import Path

TAIL_BYTES = 8192
MAX_LINE = 4000                    # under PIPE_BUF (4096) so a concurrent append cannot interleave

TERMINAL = {"completed", "failed", "skipped"}
STATUSES = {"running"} | TERMINAL


class LedgerError(RuntimeError):
    pass


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _checksum(prev, payload):
    import hashlib
    return hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    @staticmethod
    def key(unit):
        """A stable identity for a unit of work. Sorted keys, so dict ordering cannot make the same
        unit look like two."""
        return json.dumps(unit, sort_keys=True, ensure_ascii=False)

    # ---- writing ----------------------------------------------------------------------------

    def append(self, stage, unit, status, payload=None, note=""):
        if status not in STATUSES:
            raise LedgerError(f"nieznany status {status!r}")
        rec = {"stage": stage, "unit": unit, "status": status, "note": note,
               "payload": payload if payload is not None else {},
               "ts_utc": _now(), "pid": os.getpid()}

        with open(self.path, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                seq, prev = self._tail(f)
                rec["sequence_number"] = seq
                rec["prev_checksum"] = prev
                body = json.dumps(rec, sort_keys=True, ensure_ascii=False)
                rec["checksum"] = _checksum(prev, body)
                line = json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n"
                if len(line.encode("utf-8")) > MAX_LINE:
                    raise LedgerError(
                        f"rekord {len(line)} B przekracza {MAX_LINE} B — trzymaj obszerne wyniki "
                        f"w artefakcie, w ledgerze tylko tożsamość i werdykt")
                f.seek(0, os.SEEK_END)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return rec["checksum"]

    def _tail(self, f):
        """Sequence number and checksum of the last record, read from the end of the file."""
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return 0, ""
        f.seek(max(0, size - TAIL_BYTES))
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if not lines:
            return 0, ""
        try:
            last = json.loads(lines[-1])
        except json.JSONDecodeError:
            raise LedgerError("ostatnia linia ledgera jest uszkodzona — napraw ręcznie, "
                              "nie dopisuj na uszkodzonym łańcuchu")
        return int(last["sequence_number"]) + 1, str(last["checksum"])

    # ---- reading ----------------------------------------------------------------------------

    def read_all(self):
        out = []
        for ln in self.path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                out.append(json.loads(ln))
        return out

    def completed(self, stage):
        """Unit keys that finished successfully — exactly what a resuming run must skip."""
        return {self.key(r["unit"]) for r in self.read_all()
                if r["stage"] == stage and r["status"] == "completed"}

    def payloads(self, stage):
        """Completed units with their results, in the order they were recorded."""
        return [(r["unit"], r["payload"]) for r in self.read_all()
                if r["stage"] == stage and r["status"] == "completed"]

    def latest(self, stage):
        """Last status per unit — the basis for spotting orphans."""
        out = {}
        for r in self.read_all():
            if r["stage"] == stage:
                out[self.key(r["unit"])] = r
        return out

    # ---- integrity --------------------------------------------------------------------------

    def verify_chain(self):
        """Recompute the whole chain. Returns (ok, first_bad_sequence_number)."""
        prev = ""
        for i, r in enumerate(self.read_all()):
            if int(r.get("sequence_number", -1)) != i or r.get("prev_checksum", None) != prev:
                return False, i
            body = json.dumps({k: v for k, v in r.items() if k != "checksum"},
                              sort_keys=True, ensure_ascii=False)
            if _checksum(prev, body) != r.get("checksum"):
                return False, i
            prev = r["checksum"]
        return True, None

    def reconcile_orphans(self, stage):
        """A unit left `running` by a killed worker is not evidence of anything — close it.

        Called at the top of every resume. Without it a unit whose worker was SIGKILLed looks
        started-but-unfinished forever, and the resuming run cannot tell it apart from one another
        live worker is holding right now. Closing it as `failed(orphan)` makes the retry explicit
        and leaves the failure in the record.
        """
        n = 0
        for _, r in self.latest(stage).items():
            if r["status"] == "running":
                self.append(stage, r["unit"], "failed", note="resume_orphan")
                n += 1
        return n
