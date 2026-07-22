#!/usr/bin/env bash
# iteration_loop.sh — detached entry for the Iterative Calibration Loop (the ladder orchestrator).
#
# It is the outer supervisor, and it is deliberately thin: it takes its own lock, opens its own tmux
# session, and runs `engine/iteration_planner.py --mode external`, which walks the pre-authorized
# ladder of frozen contract versions and shells the PROVEN per-epoch supervisor (`ops/engine.sh`) once
# per version. Nothing here is a second execution platform — the inner loop is the engine you already
# have; this only sequences epochs, gates on integrity, and stops at convergence.
#
# Detached like the engine: the tmux SERVER is the daemon, so closing the terminal leaves the loop
# running. A DISTINCT session name and lock let it coexist with a plain `engine-*` run. No SIGTERM
# trap — closing the terminal never tears anything down; stop is cooperative via control.json.
#
#   make iteration-start ASSETS="AZO ADBE GOOG" WORKERS=3
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPS="$ROOT/ops"; ENG="$ROOT/engine"
PY="${PY:-$ROOT/.venv/bin/python3}"
LOCK="$OPS/.iteration.lock"                       # distinct from .engine.lock — may run alongside
SESSION="iterative-calibration"                   # distinct from golden-calibration
WORKERS="${WORKERS:-4}"
ASSETS="${ASSETS:-}"
SEED="${SEED:-42}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
POLICY="${POLICY:-$ROOT/config/iteration_loop_policy.json}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "BŁĄD: $*"; exit 1; }

exec 9>"$LOCK"
flock -n 9 || die "inna pętla iteracji trzyma $LOCK (jedna drabina globalnie)"
[[ -x "$PY" ]] || die "brak $PY"
command -v tmux >/dev/null || die "brak tmux"

HEAD="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo nogit)"
LADDER_ID="${RESUME_LADDER:-icl_$(date -u +%Y%m%dT%H%M%SZ)_${HEAD}}"
LADDER_DIR="$ROOT/runs/$LADDER_ID"
mkdir -p "$LADDER_DIR"

# Cooperative-stop control file the orchestrator polls at each epoch boundary (make iteration-stop).
cat > "$LADDER_DIR/control.json" <<JSON
{"halt": false, "workers": $WORKERS}
JSON
cat > "$LADDER_DIR/supervisor.json" <<JSON
{"ladder_id": "$LADDER_ID", "git_sha": "$HEAD", "workers": $WORKERS,
 "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "status": "RUNNING", "pid": $$}
JSON

tmux has-session -t "$SESSION" 2>/dev/null && die "sesja '$SESSION' już istnieje (tmux attach -t $SESSION)"

DIRTY_FLAG=""; [[ "$ALLOW_DIRTY" == "1" ]] && DIRTY_FLAG="--allow-dirty"
ASSET_FLAG=""; [[ -n "$ASSETS" ]] && ASSET_FLAG="--assets $ASSETS"

# One durable window: walk the ladder (external backend), then generate the self-summary, then mark
# COMPLETED. `9>&-` so the tmux server never inherits/pins the lock. `; bash` keeps the pane for logs.
tmux new-session -d -s "$SESSION" -c "$ROOT" -n loop \
  "PY=$PY ALLOW_DIRTY=$ALLOW_DIRTY \
   '$PY' '$ENG/iteration_planner.py' --ladder-dir '$LADDER_DIR' --mode external \
        --seed $SEED --policy '$POLICY' $ASSET_FLAG $DIRTY_FLAG 9>&-; \
   '$PY' '$ENG/iteration_report.py' --ladder-dir '$LADDER_DIR'; \
   '$PY' -c \"import json,time; p='$LADDER_DIR/supervisor.json'; d=json.load(open(p)); \
d['status']='COMPLETED'; d['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()); \
open(p,'w').write(json.dumps(d,indent=1))\"; \
   echo '=== drabina zakończona; iteration_summary.md gotowe ==='; bash"

echo "$LADDER_ID" > "$OPS/.iteration.current"
log "drabina '$LADDER_ID' wystartowana w tmux '$SESSION' (WORKERS=$WORKERS)"
log "stan: make iteration-status  ·  podgląd: tmux attach -t $SESSION  ·  stop: make iteration-stop"
