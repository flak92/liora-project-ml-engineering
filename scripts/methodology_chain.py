#!/usr/bin/env python3
"""The unattended chain — runs the remaining Rung 5 stages, gates each one, and stops on doubt.

This is the process the tmux supervisor keeps alive. It exists because the work is roughly twenty
core-hours and nobody is going to sit and watch it.

Two rules govern what it may do.

**It never makes a methodological decision.** It computes, it checks invariants, and it records. It
does not drop a candidate, move a threshold, choose a null, or decide what a result means. When a
gate fails it stops the chain rather than working around the failure, because a chain that repairs
its own inputs at three in the morning produces a number nobody can defend. The final verdict is
read by a human from the artifacts this produces.

**It is resumable at every level.** Stages are recorded in `stages.json` as they complete, and each
stage's runner keeps its own ledger of finished units, so a machine that dies mid-stage resumes
mid-stage. A stage is re-entered, not restarted.

Halting is cooperative. `control.json` is consulted between stages and each runner consults it
between units, so a stop always lands on a consistent state. The watchdog's deadline hardkill is
the backstop for the case where cooperation fails, not the normal path.

    python3 scripts/methodology_chain.py --run-dir xgb/data/runs/<id> --jobs 4
    python3 scripts/methodology_chain.py --run-dir <same> --resume
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
XGB = ROOT / "xgb"
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_init                                                        # noqa: E402
runtime_init.apply()
from artifact_io import read_json, write_json_atomic                       # noqa: E402
from ledger import Ledger                                                  # noqa: E402

PY = str(ROOT / ".venv" / "bin" / "python3")
DATA = XGB / "data"
HEARTBEAT_SEC = 30

SMOKE_FOLDS = 3


# ---------------------------------------------------------------------------------------------
# stage definitions
# ---------------------------------------------------------------------------------------------

def _null_cmd(kind, run_dir, jobs, out, folds=0, survivors=None, extra=()):
    cmd = [PY, str(ROOT / "scripts" / "procedure_null.py"), "--null", kind,
           "--jobs", str(jobs), "--out", str(out),
           "--run-dir", str(run_dir / f"null_{kind}"),
           "--control", str(run_dir / "control.json")]
    if folds:
        cmd += ["--folds", str(folds)]
    if survivors:
        cmd += ["--survivors-from", str(survivors)]
    return cmd + list(extra)


A1 = DATA / "procedure_null_a1.json"
A2 = DATA / "procedure_null_a2.json"
NB = DATA / "procedure_null_b.json"
SMOKE = DATA / "procedure_null_a1_smoke.json"


def stages(run_dir, jobs):
    """The chain. Each stage names its inputs, its command and the gate that must pass after it."""
    return [
        {"name": "S1_smoke_a1",
         "why": "prove the machinery on three outer folds before spending the full budget",
         "needs": [DATA / "crossfit_selection.json", DATA / "feature_utility.json"],
         "cmd": _null_cmd("a1", run_dir, jobs, SMOKE, folds=SMOKE_FOLDS),
         "out": SMOKE, "gate": "gate_null"},

        {"name": "S2_full_a1",
         "why": "the procedure-level multiplicity verdict",
         "needs": [SMOKE],
         "cmd": _null_cmd("a1", run_dir, jobs, A1),
         "out": A1, "gate": "gate_null"},

        {"name": "S3_a2_survivors",
         "why": "does the verdict survive keeping each block inside its own market regime",
         "needs": [A1],
         "cmd": _null_cmd("a2", run_dir, jobs, A2, survivors=A1),
         "out": A2, "gate": "gate_null_optional"},

        {"name": "S4_b_survivors",
         "why": "does the verdict survive a null that keeps the dependence on core",
         "needs": [A1],
         "cmd": _null_cmd("b", run_dir, jobs, NB, survivors=A1),
         "out": NB, "gate": "gate_null_optional"},

        {"name": "S5_summary",
         "why": "assemble the artifacts a human reads to close Rung 5",
         "needs": [A1],
         "cmd": [PY, str(ROOT / "scripts" / "rung5_summary.py"),
                 "--run-dir", str(run_dir), "--out", str(DATA / "rung5_summary.json")],
         "out": DATA / "rung5_summary.json", "gate": "gate_summary"},
    ]


# ---------------------------------------------------------------------------------------------
# gates — fail-closed, and deliberately about invariants rather than about results
# ---------------------------------------------------------------------------------------------

def gate_null(out, run_dir):
    """What must be true of a null artifact regardless of which way the verdict went.

    None of these check whether anything passed. A gate that fired on "too few survivors" would be
    the chain forming an opinion, which is exactly what it must not do.
    """
    doc = read_json(out)
    if doc is None:
        return False, "artefakt nieczytelny lub nieobecny"
    folds = [f for t in doc["tables"].values() for f in t["folds"]]
    if not folds:
        return False, "zero foldów w artefakcie"

    problems = []
    for f in folds:
        tag = f"{f['ticker']}/{f['outer_fold']}"
        h = f.get("index_hashes", [])
        if len(h) != len(set(h)):
            problems.append(f"{tag}: powtórzone permutacje")
        prov = f.get("provenance", {})
        if prov.get("displaced", 0) < 0.5 * prov.get("n_blocks", 1):
            problems.append(f"{tag}: permutacja przemieszcza mniej niż połowę bloków")
        if prov.get("n_blocks", 0) < 8:
            problems.append(f"{tag}: {prov.get('n_blocks')} bloków — przestrzeń permutacji za mała")
        for a, v in f["arms"].items():
            if v["verdict"] == "incomplete":
                problems.append(f"{tag}/{a}: niedokończony ({v['permutations_executed']} permutacji)")
            if v["verdict"] == "rejected_early" and v["exceedances"] < 5:
                problems.append(f"{tag}/{a}: rejected_early przy b={v['exceedances']}")
            if v["verdict"] == "passed" and v["exceedances"] > 4:
                problems.append(f"{tag}/{a}: passed przy b={v['exceedances']}")
            lb = v.get("final_p_lower_bound")
            # The reported bound is round((1+b)/51, 6); at the futility stop b = 5 exactly, so lb is
            # 6/51 rounded to six places = 0.117647, which sits ~5.9e-8 BELOW the unrounded 6/51.
            # The 1e-9 epsilon could not absorb that rounding gap, so a correct futility bound tripped
            # a guard meant only for a genuinely-too-low bound (a coding error producing, say, 0.10).
            # A 1e-6 tolerance clears the rounding by two orders of magnitude and still catches that.
            if lb is not None and lb < 6 / 51 - 1e-6:
                problems.append(f"{tag}/{a}: dolna granica p {lb} poniżej 6/51")

    man = run_dir / f"null_{doc['contract']['null']}" / "run_manifest.json"
    if man.exists():
        from run_manifest import check_required_fields
        chk = check_required_fields(man)
        if not chk["ok"]:
            problems.append(f"manifest niekompletny: {chk}")

    return (not problems), "; ".join(problems[:6]) or "wszystkie niezmienniki trzymają"


def gate_null_optional(out, run_dir):
    """A sensitivity stage with no survivors to test is a valid outcome, not a failure."""
    doc = read_json(out)
    if doc is None:
        return False, "artefakt nieczytelny lub nieobecny"
    if not [f for t in doc["tables"].values() for f in t["folds"]]:
        return True, "brak survivorów do przetestowania — poprawny i pełny wynik"
    return gate_null(out, run_dir)


def gate_summary(out, run_dir):
    doc = read_json(out)
    return (doc is not None), ("podsumowanie zapisane" if doc else "brak podsumowania")


GATES = {"gate_null": gate_null, "gate_null_optional": gate_null_optional,
         "gate_summary": gate_summary}


# ---------------------------------------------------------------------------------------------

class Chain:
    def __init__(self, run_dir, jobs):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jobs = jobs
        self.state_path = self.run_dir / "stages.json"
        self.control = self.run_dir / "control.json"
        self.heartbeat = self.run_dir / "heartbeat"
        self.log = self.run_dir / "chain.log"
        self.state = read_json(self.state_path, {}) or {}

    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def beat(self):
        self.heartbeat.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def halted(self):
        return bool((read_json(self.control, {}) or {}).get("halt"))

    def workers(self):
        """The watchdog may lower this when memory gets tight; honoured between stages."""
        return int((read_json(self.control, {}) or {}).get("workers") or self.jobs)

    def mark(self, name, status, detail=""):
        self.state[name] = {"status": status, "detail": detail,
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json_atomic(self.state_path, self.state)

    def run(self):
        self.beat()
        for st in stages(self.run_dir, self.jobs):
            name = st["name"]
            if self.state.get(name, {}).get("status") == "completed":
                self.say(f"{name}: już ukończony, pomijam")
                continue
            if self.halted():
                self.say("control.json: halt — zatrzymuję łańcuch między etapami")
                return 0

            missing = [str(p) for p in st["needs"] if not Path(p).exists()]
            if missing:
                self.mark(name, "failed", f"brak wejść: {missing}")
                self.say(f"{name}: BRAK WEJŚĆ {missing} — łańcuch zatrzymany (fail-closed)")
                return 3

            jobs = self.workers()
            cmd = [c.replace(str(self.jobs), str(jobs)) if c == str(self.jobs) else c
                   for c in st["cmd"]]
            self.say(f"{name}: start ({st['why']}), workerów={jobs}")
            self.mark(name, "running")
            rc = self._spawn(cmd)
            if rc != 0:
                self.mark(name, "failed", f"rc={rc}")
                self.say(f"{name}: rc={rc} — łańcuch zatrzymany (fail-closed)")
                return 4

            ok, why = GATES[st["gate"]](st["out"], self.run_dir)
            if not ok:
                self.mark(name, "failed", f"bramka: {why}")
                self.say(f"{name}: BRAMKA NIE PRZESZŁA — {why}")
                return 5
            self.mark(name, "completed", why)
            self.say(f"{name}: ukończony — {why}")

        self.say("łańcuch ukończony; werdykt Rung 5 czyta CZŁOWIEK z artefaktów")
        return 0

    def _spawn(self, cmd):
        """Run a stage, beating the heartbeat while it works so the watchdog knows we live."""
        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        with open(self.log, "a", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
            while True:
                try:
                    return p.wait(timeout=HEARTBEAT_SEC)
                except subprocess.TimeoutExpired:
                    self.beat()
                    if self.halted():
                        self.say("halt w trakcie etapu — runner dokończy bieżącą jednostkę")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="documentation only; resume is the default")
    args = ap.parse_args()
    return Chain(args.run_dir, args.jobs).run()


if __name__ == "__main__":
    raise SystemExit(main())
