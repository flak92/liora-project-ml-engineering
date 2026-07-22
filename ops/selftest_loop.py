#!/usr/bin/env python3
"""Prove the orchestration works — especially the parts that fail silently when they don't.

Every check here corresponds to a real failure mode, and most of them were observed in the repo
this pattern came from rather than imagined:

  1. A child that inherits the locked descriptor keeps the flock alive after the supervisor dies.
     Because a tmux server is long-lived, this bricks every future start while `ps` shows nothing
     running. The test asserts BOTH directions: without `9>&-` the lock is still held, with it the
     lock is free. A one-directional test would pass even if the fix did nothing.
  2. A non-atomic artifact write leaves truncated JSON that already overwrote the good file.
  3. A watchdog whose restart branch is unreachable looks healthy and never restarts anything. In
     the source repo no run log ever showed a restart. Here it must actually fire.
  4. A cooperative halt that is not honoured turns every stop into a hard kill.
  5. A second start while one run is live corrupts the ledger.

    make loop-selftest
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PY = str(ROOT / ".venv" / "bin" / "python3")

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def lock_held(lock):
    """True when someone still holds the flock."""
    rc = subprocess.run(["flock", "-n", str(lock), "-c", "true"]).returncode
    return rc != 0


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ---------------------------------------------------------------------------------------------

def test_fd_inheritance():
    """The bug that bricks the lock forever. Both directions, or the test proves nothing."""
    print("\n1. dziedziczenie deskryptora locka (9>&-)")
    d = Path(tempfile.mkdtemp())
    lock = d / "t.lock"

    def spawn(close_fd):
        """A parent takes the lock, forks a long-lived child, then exits. If the child inherited
        fd 9, the kernel keeps the lock alive for as long as the child runs.

        The child is a subshell writing its own pid to a file rather than a bare `sleep &` whose
        pid is echoed: under command substitution a simple background job does not outlive its
        parent here, which would make the test measure the harness instead of the code.
        """
        pidf = d / f"kid_{int(close_fd)}.pid"
        redirect = "9>&-" if close_fd else ""
        script = (f'exec 9>"{lock}"; flock -n 9 || exit 9; '
                  f'( echo $BASHPID > "{pidf}"; sleep 20 ) {redirect} & '
                  f'exit 0')
        subprocess.run(["bash", "-c", script], check=False)
        for _ in range(40):
            if pidf.exists() and pidf.read_text().strip():
                return int(pidf.read_text().strip())
            time.sleep(0.05)
        raise RuntimeError("dziecko nie wystartowało")

    def kill(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    kid = spawn(close_fd=False)
    time.sleep(0.3)
    alive_a = _pid_alive(kid)
    leaked = lock_held(lock)
    kill(kid)
    time.sleep(0.4)

    kid = spawn(close_fd=True)
    time.sleep(0.3)
    alive_b = _pid_alive(kid)
    freed = not lock_held(lock)
    kill(kid)

    check("oba dzieci faktycznie żyły w chwili pomiaru", alive_a and alive_b,
          "(gdyby nie żyły, test mierzyłby harness, nie kod)")

    check("bez 9>&- dziecko TRZYMA lock po śmierci rodzica", leaked,
          "(kontrola negatywna — gdyby to nie zachodziło, test niczego nie dowodzi)")
    check("z 9>&- lock jest zwolniony", freed)
    shutil.rmtree(d, ignore_errors=True)


def test_atomic_write():
    """SIGKILL during a write must never damage the previous good artifact."""
    print("\n2. atomowość zapisu artefaktu pod SIGKILL")
    d = Path(tempfile.mkdtemp())
    target = d / "a.json"
    subprocess.run([PY, "-c", f"""
import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r})
from artifact_io import write_json_atomic
write_json_atomic({str(target)!r}, {{"generation": 0, "payload": [1,2,3]}})
"""], check=True)

    writer = subprocess.Popen([PY, "-c", f"""
import sys; sys.path.insert(0, {str(ROOT / 'scripts')!r})
from artifact_io import write_json_atomic
big = list(range(200000))
for i in range(1, 100000):
    write_json_atomic({str(target)!r}, {{"generation": i, "payload": big}})
"""])
    time.sleep(1.2)
    writer.send_signal(signal.SIGKILL)
    writer.wait()

    try:
        doc = json.loads(target.read_text())
        valid = "generation" in doc
    except Exception as e:                                        # noqa: BLE001
        valid, doc = False, str(e)
    leftovers = list(d.glob("*.tmp*"))
    check("plik docelowy nadal jest poprawnym JSON-em", valid,
          f"generation={doc.get('generation') if isinstance(doc, dict) else doc}")
    check("brak osieroconych plików tymczasowych", not leftovers, str(leftovers))
    shutil.rmtree(d, ignore_errors=True)


def test_ledger_resume():
    """A killed worker leaves the ledger usable and its orphan visible."""
    print("\n3. ledger: wznowienie i domknięcie sieroty")
    from ledger import Ledger
    d = Path(tempfile.mkdtemp())
    led = Ledger(d / "l.jsonl")
    for i in range(3):
        led.append("s", {"i": i}, "completed", payload={"v": i})
    led.append("s", {"i": 3}, "running")

    fresh = Ledger(d / "l.jsonl")
    done_before = len(fresh.completed("s"))
    orphans = fresh.reconcile_orphans("s")
    ok, bad = fresh.verify_chain()
    check("ukończone jednostki widoczne po restarcie", done_before == 3, f"{done_before}/3")
    check("sierota domknięta", orphans == 1)
    check("łańcuch hash spójny po rekoncyliacji", ok, f"bad={bad}")
    shutil.rmtree(d, ignore_errors=True)


def _fake_root(d, chain_body):
    """A throwaway repo root whose chain is a stub, so the watchdog can be tested in seconds."""
    (d / "scripts").mkdir(parents=True, exist_ok=True)
    (d / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    (d / "scripts" / "methodology_chain.py").write_text(chain_body)
    os.symlink(PY, d / ".venv" / "bin" / "python3")
    return d


def test_guard_restart():
    """The branch that was dead in the source repo. It must fire here."""
    print("\n4. watchdog: restart martwego łańcucha")
    d = Path(tempfile.mkdtemp())
    root = _fake_root(d, """
import sys, time, pathlib
rd = pathlib.Path(sys.argv[sys.argv.index('--run-dir') + 1])
(rd / 'restarted').write_text((rd / 'restarted').read_text() + 'x'
                              if (rd / 'restarted').exists() else 'x')
(rd / 'heartbeat').write_text('now')
time.sleep(30)
""")
    run = d / "run"
    run.mkdir()
    (run / "control.json").write_text(json.dumps(
        {"halt": False, "workers": 4, "deadline_epoch": int(time.time()) + 3600,
         "deadline_hardkill": False}))
    (run / "stages.json").write_text(json.dumps({"S1": {"status": "running"}}))
    (run / "chain.pid").write_text("999999")               # a pid that is certainly dead
    (run / "heartbeat").write_text("stale")
    os.utime(run / "heartbeat", (time.time() - 600, time.time() - 600))

    env = dict(os.environ, GUARD_TICK_SEC="1", GUARD_STALE_SEC="1",
               GUARD_DEADLINE_GRACE_SEC="120", PY=PY)
    g = subprocess.Popen(["bash", str(ROOT / "ops" / "guard.sh"),
                          "--run-dir", str(run), "--root", str(root)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    time.sleep(6)
    g.send_signal(signal.SIGTERM)
    try:
        out = g.communicate(timeout=5)[0]
    except subprocess.TimeoutExpired:
        g.kill()
        out = g.communicate()[0]

    restarted = (run / "restarted").exists()
    check("watchdog wskrzesił łańcuch", restarted, (out or "").strip().splitlines()[-1:] and
          (out.strip().splitlines()[-1][:70]))
    check("lease sprzątnięty po restarcie", not (run / ".restart.lease").exists())
    for f in run.glob("chain.pid"):
        try:
            os.kill(int(f.read_text()), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    shutil.rmtree(d, ignore_errors=True)


def test_guard_memory_degradation():
    """Zero swap on this machine: an OOM is a kill, so the watchdog must step down first."""
    print("\n5. watchdog: obniżenie liczby workerów przy niskiej pamięci")
    d = Path(tempfile.mkdtemp())
    root = _fake_root(d, "import time; time.sleep(30)")
    run = d / "run"
    run.mkdir()
    (run / "control.json").write_text(json.dumps(
        {"halt": False, "workers": 4, "deadline_epoch": int(time.time()) + 3600,
         "deadline_hardkill": False}))
    (run / "chain.pid").write_text(str(os.getpid()))       # alive, so no restart path
    (run / "heartbeat").write_text("now")

    env = dict(os.environ, GUARD_TICK_SEC="1", GUARD_MIN_AVAIL_MB="999999", PY=PY)
    g = subprocess.Popen(["bash", str(ROOT / "ops" / "guard.sh"),
                          "--run-dir", str(run), "--root", str(root)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    time.sleep(5)
    g.send_signal(signal.SIGTERM); g.wait(timeout=5)
    w = json.loads((run / "control.json").read_text())["workers"]
    check("workerzy obniżeni pod presją pamięci", w < 4, f"4 -> {w}")
    check("nie schodzi poniżej progu minimalnego", w >= 2, f"workers={w}")
    shutil.rmtree(d, ignore_errors=True)


def test_cooperative_halt():
    """A halt must be a flag the chain reads, not a signal that interrupts it mid-unit."""
    print("\n6. halt kooperacyjny")
    from methodology_chain import Chain
    d = Path(tempfile.mkdtemp())
    c = Chain(d, jobs=4)
    (d / "control.json").write_text(json.dumps({"halt": False, "workers": 3}))
    before = c.halted()
    (d / "control.json").write_text(json.dumps({"halt": True, "workers": 3}))
    check("halt=false nie zatrzymuje", not before)
    check("halt=true jest odczytany", c.halted())
    check("watchdog może obniżyć workerów przez control.json", c.workers() == 3, "4 -> 3")
    shutil.rmtree(d, ignore_errors=True)


def test_double_start():
    print("\n7. lock: drugi start jest odrzucany")
    d = Path(tempfile.mkdtemp())
    lock = d / "t.lock"
    p = subprocess.Popen(["flock", str(lock), "-c", "sleep 5"])
    time.sleep(0.5)
    rc = subprocess.run(["flock", "-n", str(lock), "-c", "true"]).returncode
    check("drugi start dostaje odmowę", rc != 0, f"rc={rc}")
    p.kill(); p.wait()
    shutil.rmtree(d, ignore_errors=True)


def test_gates_are_fail_closed():
    """A gate that passes on a missing or broken artifact is worse than no gate."""
    print("\n8. bramki są fail-closed")
    from methodology_chain import gate_null, gate_null_optional
    d = Path(tempfile.mkdtemp())
    missing = d / "nope.json"
    ok, _ = gate_null(missing, d)
    check("brak artefaktu nie przechodzi bramki", not ok)
    broken = d / "b.json"
    broken.write_text('{"tables": {')
    ok, _ = gate_null(broken, d)
    check("obcięty artefakt nie przechodzi bramki", not ok)
    empty = d / "e.json"
    empty.write_text(json.dumps({"contract": {"null": "a2"}, "tables": {}}))
    ok, why = gate_null_optional(empty, d)
    check("brak survivorów to POPRAWNY wynik, nie awaria", ok, why)
    bad = d / "x.json"
    bad.write_text(json.dumps({"contract": {"null": "a1"}, "tables": {"T": {"folds": [{
        "ticker": "T", "outer_fold": 0, "permutations_executed": 50,
        "index_hashes": ["a", "a"],
        "provenance": {"n_blocks": 100, "min_displaced_fraction": 0.30},
        "arms": {"flat": {"verdict": "passed", "exceedances": 9,
                          "permutations_executed": 50}}}]}}}))
    ok, why = gate_null(bad, d)
    check("dup permutacji, przemieszczenie 0.30<0.5 i b=9/passed są wyłapane", not ok, why[:90])

    # Regression: a legitimate futility stop (b=5) reports lb = round(6/51, 6) = 0.117647, which is
    # ~5.9e-8 below the unrounded 6/51. The gate must NOT read that rounding gap as an anomaly — this
    # is the bug that halted the real S1 smoke fail-closed on a correct result.
    fut = d / "fut.json"
    fut.write_text(json.dumps({"contract": {"null": "a1"}, "tables": {"T": {"folds": [{
        "ticker": "T", "outer_fold": 0, "permutations_executed": 38,
        "index_hashes": [f"h{i}" for i in range(38)],
        "provenance": {"n_blocks": 300, "displaced": 300},
        "arms": {"flat": {"verdict": "rejected_early", "exceedances": 5,
                          "permutations_executed": 38,
                          "final_p_lower_bound": round(6 / 51, 6)}}}]}}}))
    ok, why = gate_null(fut, d)
    check("poprawny futility stop (b=5, lb=6/51) PRZECHODZI bramkę", ok, why[:64])
    shutil.rmtree(d, ignore_errors=True)


def main():
    print("selftest orkiestracji — każdy przypadek odpowiada realnemu trybowi awarii")
    for t in (test_fd_inheritance, test_atomic_write, test_ledger_resume,
              test_guard_restart, test_guard_memory_degradation,
              test_cooperative_halt, test_double_start, test_gates_are_fail_closed):
        try:
            t()
        except Exception as e:                                      # noqa: BLE001
            check(f"{t.__name__} rzucił wyjątek", False, f"{type(e).__name__}: {e}")
    print(f"\n{'WSZYSTKO ZDANE' if not FAILS else 'NIEZDANE: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
