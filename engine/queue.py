#!/usr/bin/env python3
"""A file-system task queue whose only coordination primitive is the atomic rename.

`os.rename` within one filesystem is atomic and mutually exclusive: if two workers try to move the
same pending file into `running/`, exactly one succeeds and the other gets `FileNotFoundError`. That
is the whole claim protocol — no lock server, no database, no shared mutable file. It works because
everything lives under one run directory on one filesystem; the engine selftest asserts that
precondition rather than assuming it.

    pending/  a task waiting to be claimed          (planner writes here)
    running/  a task a worker owns right now         (claimed by rename)
    done/     a task whose result was published
    failed/   a task that errored (kept for audit and requeue)
"""
import json
import os
from pathlib import Path

SUBDIRS = ("pending", "running", "done", "failed")


class Queue:
    def __init__(self, run_dir):
        self.root = Path(run_dir) / "queue"
        for s in SUBDIRS:
            (self.root / s).mkdir(parents=True, exist_ok=True)

    def _p(self, sub, name):
        return self.root / sub / name

    def enqueue(self, task):
        """Publish a task to pending. Idempotent on task_hash — re-enqueuing an already-pending or
        running task is a no-op, so the planner can run repeatedly without duplicating work."""
        name = f"{task['task_hash']}.json"
        if self._p("pending", name).exists() or self._p("running", name).exists():
            return False
        tmp = self._p("pending", name + ".tmp")
        tmp.write_text(json.dumps(task, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._p("pending", name))       # atomic publish into pending
        return True

    def claim(self):
        """Atomically take one pending task into running. Returns the task dict, or None if empty.

        The rename is the mutual exclusion: a losing worker sees FileNotFoundError and simply tries
        the next candidate. No task is ever handed to two workers.
        """
        for f in sorted((self.root / "pending").glob("*.json")):
            dst = self._p("running", f.name)
            try:
                os.rename(f, dst)                        # atomic claim; raises if already taken
            except FileNotFoundError:
                continue
            except OSError:
                continue
            return json.loads(dst.read_text(encoding="utf-8"))
        return None

    def finish(self, task, status):
        """Move a claimed task to done/ or failed/. status in {'done','failed'}."""
        name = f"{task['task_hash']}.json"
        src = self._p("running", name)
        if not src.exists():
            return False
        os.rename(src, self._p(status, name))
        return True

    def requeue_stale(self, running_names):
        """Move named running tasks back to pending for retry (the scheduler's stale-task sweep).
        `running_names` is the set of task_hash the guard judged stale (no heartbeat)."""
        moved = []
        for h in running_names:
            src = self._p("running", f"{h}.json")
            if src.exists():
                os.rename(src, self._p("pending", f"{h}.json"))
                moved.append(h)
        return moved

    def counts(self):
        return {s: sum(1 for _ in (self.root / s).glob("*.json")) for s in SUBDIRS}

    def list(self, sub):
        return [json.loads(f.read_text(encoding="utf-8"))
                for f in sorted((self.root / sub).glob("*.json"))]
