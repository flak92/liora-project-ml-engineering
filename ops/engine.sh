#!/usr/bin/env bash
# engine.sh — supervisor for the methodology execution engine.
#
# It is the thin execution layer, not a new platform: it opens a tmux session whose windows are a
# planner, a pool of workers, a guard and a scheduler, and lets them realize the Calibration DAG that
# the science layer defines. It decides nothing scientific — every transition still comes from a
# result artifact and the frozen contract.
#
# Started detached by `make engine-start`: the tmux server is the daemon, so the terminal can be
# closed. Same hard-won lessons as the main loop: a global flock so two runs never collide, and
# `9>&-` on every child so a long-lived tmux server never inherits and pins the lock.
#
#   make engine-start ASSETS="AZO ADBE GOOG" WORKERS=3 HOURS=8
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS="$ROOT/ops"; ENG="$ROOT/engine"
PY="${PY:-$ROOT/.venv/bin/python3}"
LOCK="$OPS/.engine.lock"
WORKERS="${WORKERS:-4}"
HOURS="${HOURS:-8}"
ASSETS="${ASSETS:-}"
SEED="${SEED:-42}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "BŁĄD: $*"; exit 1; }

exec 9>"$LOCK"
flock -n 9 || die "inny engine trzyma $LOCK (jeden run globalnie)"

[[ -x "$PY" ]] || die "brak $PY"
command -v tmux >/dev/null || die "brak tmux"

HEAD="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
RUN_ID="${RESUME_RUN:-gc_$(date -u +%Y%m%dT%H%M%SZ)_${HEAD}}"
RUNDIR="$ROOT/runs/$RUN_ID"
mkdir -p "$RUNDIR"
DEADLINE=$(( $(date -u +%s) + HOURS * 3600 ))

# Rung 0 — freeze the contract snapshot for this run (unless resuming an existing one).
if [[ ! -f "$RUNDIR/contract.json" ]]; then
  DIRTY_FLAG=""; [[ "$ALLOW_DIRTY" == "1" ]] && DIRTY_FLAG="--allow-dirty"
  "$PY" - "$RUNDIR" "$SEED" "$ALLOW_DIRTY" $ASSETS <<'PYEOF' || die "snapshot kontraktu nieudany"
import sys; sys.path.insert(0, "engine"); sys.path.insert(0, "scripts")
import contract as CT
run, seed, dirty, *assets = sys.argv[1:]
import json
if not assets:
    assets = json.load(open("config/sample_20.json"))["sample"]
CT.snapshot(run, assets, seed=int(seed), allow_dirty=(dirty == "1"))
print("contract snapshot:", run)
PYEOF
fi

cat > "$RUNDIR/control.json" <<JSON
{"halt": false, "workers": $WORKERS, "deadline_epoch": $DEADLINE, "deadline_hardkill": false}
JSON
cat > "$RUNDIR/supervisor.json" <<JSON
{"run_id": "$RUN_ID", "git_sha": "$HEAD", "workers": $WORKERS, "hours": $HOURS,
 "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "status": "RUNNING", "pid": $$}
JSON

SESSION="golden-calibration"
tmux has-session -t "$SESSION" 2>/dev/null && die "sesja '$SESSION' już istnieje (tmux attach -t $SESSION)"

log "START engine $RUN_ID  workerów=$WORKERS  deadline=+${HOURS}h"
log "run_dir: $RUNDIR"

# One tmux session; each window a role. Workers pull from the queue, the scheduler paces the planner
# and reducer, the guard watches liveness. All started with 9>&- so none pins the flock.
tmux new-session -d -s "$SESSION" -c "$ROOT" -n planner \
  "PY=$PY $ENG/../ops/scheduler.sh --run-dir '$RUNDIR' 9>&-; bash"
for i in $(seq 1 "$WORKERS"); do
  w=$(printf 'worker-%02d' "$i")
  tmux new-window -t "$SESSION" -c "$ROOT" -n "$w" \
    "PY=$PY $OPS/worker.sh --run-dir '$RUNDIR' --worker '$w' 9>&-; bash"
done
tmux new-window -t "$SESSION" -c "$ROOT" -n guard \
  "PY=$PY $OPS/guard.sh --run-dir '$RUNDIR' --root '$ROOT' 9>&-; bash"

echo "$RUN_ID" > "$OPS/.engine.current"
log "sesja '$SESSION' uruchomiona; podgląd: tmux attach -t $SESSION  ·  stan: make engine-status"

# Supervisor waits until the run is done (all terminal) or the deadline; then tears the session down.
while true; do
  sleep 30
  DONE=$("$PY" - "$RUNDIR" <<'PYEOF' 2>/dev/null || echo 0
import sys; sys.path.insert(0,"engine"); sys.path.insert(0,"scripts")
import planner as PL, states as ST
try:
    acts = PL.plan(sys.argv[1], assemble=True)
    print(1 if all(ST.is_terminal(a["state"]) for a in acts) else 0)
except Exception:
    print(0)
PYEOF
)
  HALT=$("$PY" -c "import json,sys;print(1 if json.load(open(sys.argv[1])).get('halt') else 0)" "$RUNDIR/control.json" 2>/dev/null || echo 0)
  NOW=$(date -u +%s)
  if [[ "$DONE" == "1" ]]; then log "wszystkie assety terminalne — koniec runu"; break; fi
  if [[ "$HALT" == "1" ]]; then log "halt ustawiony — kończę"; break; fi
  if (( NOW >= DEADLINE )); then log "deadline — kończę"; break; fi
done

"$PY" - "$RUNDIR" <<'PYEOF'
import json,sys,time
p=sys.argv[1]+"/supervisor.json"; d=json.load(open(p))
d["status"]="COMPLETED"; d["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
open(p,"w").write(json.dumps(d,indent=1))
PYEOF
tmux kill-session -t "$SESSION" 2>/dev/null || true
log "FINALIZE COMPLETED ($RUN_ID)"
