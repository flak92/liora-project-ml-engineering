#!/usr/bin/env bash
# loop.sh — supervisor for the unattended Golden Calibration chain.
#
# Takes a global lock, mints a run id, writes the manifest and the control channel, forks the
# watchdog as a sibling, runs the chain in the background, waits for it, and finalizes with a
# distinct exit code. Designed to be started detached by `make loop-start`, which puts it inside a
# tmux session so the tmux SERVER is the daemon and the terminal can be closed.
#
# THE ONE THING NOT TO GET WRONG: every child that can outlive this script is forked with `9>&-`.
# A child that inherits the locked file descriptor keeps the flock alive after the supervisor is
# gone — and because a tmux server is long-lived, a tmux session started under the lock holds it
# forever. The symptom is that every subsequent start reports "another run holds the lock" while
# `ps` shows nothing running. That failure cost the source repo real time; it is not repeated here.
#
# The chain's stdout is NOT piped through tee: a pipeline reports the exit status of its last
# command, so `chain | tee` would mask FAILED_SAFE. The chain writes its own log.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS="$ROOT/ops"
LOCK="$OPS/.golden.lock"
RUNS="$ROOT/xgb/data/runs"
PY="${PY:-$ROOT/.venv/bin/python3}"
JOBS="${JOBS:-4}"
HOURS="${HOURS:-12}"

EXIT_BAD_ARG=2
EXIT_LOCK=6
EXIT_PRECONDITION=7
EXIT_FAILED_SAFE=42

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { local rc=$1; shift; log "BŁĄD: $*"; exit "$rc"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)  JOBS="$2"; shift 2 ;;
    --hours) HOURS="$2"; shift 2 ;;
    --resume-run) RESUME_RUN="$2"; shift 2 ;;
    *) die $EXIT_BAD_ARG "nieznany argument: $1" ;;
  esac
done

# ---- lock ------------------------------------------------------------------------------------
exec 9>"$LOCK"
flock -n 9 || die $EXIT_LOCK "inny przebieg trzyma $LOCK (jeden przebieg globalnie)"

# ---- preconditions ---------------------------------------------------------------------------
[[ -x "$PY" ]] || die $EXIT_PRECONDITION "brak interpretera $PY"
for f in xgb/data/crossfit_selection.json xgb/data/feature_utility.json \
         config/feature_discovery_contract.json; do
  [[ -f "$ROOT/$f" ]] || die $EXIT_PRECONDITION "brak wejścia: $f"
done

# ---- run identity ----------------------------------------------------------------------------
HEAD_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo nogit)"
if [[ -n "${RESUME_RUN:-}" ]]; then
  RUN_ID="$RESUME_RUN"
else
  RUN_ID="golden_$(date -u +%Y%m%dT%H%M%SZ)_${HEAD_SHA:0:7}"
fi
RUNDIR="$RUNS/$RUN_ID"
mkdir -p "$RUNDIR"
DEADLINE=$(( $(date -u +%s) + HOURS * 3600 ))

# The control channel. Cooperative halt is read between units of work, so a stop always lands on a
# consistent ledger; the deadline is the watchdog's backstop for when cooperation fails.
[[ -f "$RUNDIR/control.json" ]] || cat > "$RUNDIR/control.json" <<JSON
{"halt": false, "finalize": false, "workers": $JOBS,
 "deadline_utc": "$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ)",
 "deadline_epoch": $DEADLINE, "deadline_hardkill": false}
JSON

cat > "$RUNDIR/supervisor.json" <<JSON
{"run_id": "$RUN_ID", "git_sha": "$HEAD_SHA", "jobs": $JOBS, "hours": $HOURS,
 "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "status": "RUNNING",
 "supervisor_pid": $$}
JSON

finalize() {
  local status="$1"
  local tmp="$RUNDIR/supervisor.json.tmp"
  "$PY" - "$RUNDIR/supervisor.json" "$status" > "$tmp" <<'PYEOF'
import json, sys, time
p, status = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["status"] = status
d["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
print(json.dumps(d, indent=1))
PYEOF
  mv "$tmp" "$RUNDIR/supervisor.json"
  [[ -n "${GUARD_PID:-}" ]] && kill "$GUARD_PID" 2>/dev/null
  log "FINALIZE $status  (run_dir: $RUNDIR)"
}

trap '[[ -n "${CHAIN_PID:-}" ]] && kill -TERM "$CHAIN_PID" 2>/dev/null; \
      finalize FAILED_SAFE; exit $EXIT_FAILED_SAFE' SIGTERM SIGINT

log "START $RUN_ID  sha=${HEAD_SHA:0:7}  workerów=$JOBS  deadline=+${HOURS}h"
log "run_dir: $RUNDIR"

# ---- watchdog, WITHOUT the lock fd -----------------------------------------------------------
"$OPS/guard.sh" --run-dir "$RUNDIR" --root "$ROOT" >>"$RUNDIR/guard.log" 2>&1 9>&- &
GUARD_PID=$!
echo "$GUARD_PID" > "$RUNDIR/guard.pid"
log "watchdog pid=$GUARD_PID (bez odziedziczonego fd 9)"

# ---- the chain -------------------------------------------------------------------------------
"$PY" "$ROOT/scripts/methodology_chain.py" --run-dir "$RUNDIR" --jobs "$JOBS" \
    >>"$RUNDIR/chain.stdout.log" 2>&1 9>&- &
CHAIN_PID=$!
echo "$CHAIN_PID" > "$RUNDIR/chain.pid"

wait "$CHAIN_PID"; rc=$?

if [[ $rc -eq 0 ]]; then
  finalize COMPLETED
elif grep -q '"deadline_hardkill": *true' "$RUNDIR/control.json" 2>/dev/null; then
  finalize DEADLINE_HARDKILL
  exit $EXIT_FAILED_SAFE
else
  finalize "FAILED_SAFE(rc=$rc)"
  exit $EXIT_FAILED_SAFE
fi
