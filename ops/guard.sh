#!/usr/bin/env bash
# guard.sh — watchdog sibling of the supervisor.
#
# It does exactly four things and deliberately nothing else. It does not choose hypotheses, write
# results, read the registry, decide what a verdict means, or edit the contract. A watchdog with
# opinions is a second experimenter nobody reviewed.
#
#   1. restart the chain (which resumes from its ledger) when its process is dead and the
#      heartbeat has gone stale, under a single-writer lease so two watchdogs cannot both do it;
#   2. lower the worker count in control.json when available memory gets tight — this machine has
#      7.7 GB and NO SWAP, so an OOM kill is a hard kill, not a slowdown;
#   3. hard-kill at the deadline, after a grace period, marking control.json so the supervisor can
#      report DEADLINE_HARDKILL rather than a mystery failure;
#   4. exit when the chain is gone or halt is set.
#
# In the repo this pattern came from, the restart branch was unreachable: the supervisor waited on
# the worker and killed the guard the instant it exited, long before the staleness threshold. Here
# the guard watches the CHAIN, and the supervisor's own wait is what keeps it alive, so the branch
# is live — and the verification suite kills a chain on purpose to prove it fires.
set -uo pipefail

TICK="${GUARD_TICK_SEC:-20}"
STALE="${GUARD_STALE_SEC:-180}"
GRACE="${GUARD_DEADLINE_GRACE_SEC:-120}"
MIN_AVAIL_MB="${GUARD_MIN_AVAIL_MB:-700}"
MIN_WORKERS="${GUARD_MIN_WORKERS:-2}"
MAX_RESTARTS="${GUARD_MAX_RESTARTS:-5}"

RUN_DIR=""; ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --root)    ROOT="$2";    shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$RUN_DIR" && -n "$ROOT" ]] || { echo "guard: brak --run-dir/--root"; exit 2; }

PY="${PY:-$ROOT/.venv/bin/python3}"
CONTROL="$RUN_DIR/control.json"
HEARTBEAT="$RUN_DIR/heartbeat"
CHAIN_PIDF="$RUN_DIR/chain.pid"
LEASE="$RUN_DIR/.restart.lease"
RESTARTS=0

log() { printf '[%s] guard: %s\n' "$(date -u +%H:%M:%S)" "$*"; }

_alive()   { [[ -n "${1:-}" ]] && kill -0 "$1" 2>/dev/null; }
_control() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" \
             "$CONTROL" "$1" 2>/dev/null; }

_set_control() {
  "$PY" - "$CONTROL" "$1" "$2" <<'PYEOF'
import json, sys
p, k, v = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(p))
try:
    v = json.loads(v)
except json.JSONDecodeError:
    pass
d[k] = v
tmp = p + ".guardtmp"
open(tmp, "w").write(json.dumps(d, indent=1))
import os; os.replace(tmp, p)
PYEOF
}

log "start (tick=${TICK}s stale=${STALE}s grace=${GRACE}s min_avail=${MIN_AVAIL_MB}MB)"

while true; do
  sleep "$TICK"
  [[ -f "$CONTROL" ]] || { log "brak control.json — kończę"; exit 0; }

  [[ "$(_control halt)" == "True" ]] && { log "halt ustawiony — kończę"; exit 0; }

  CHAIN_PID="$(cat "$CHAIN_PIDF" 2>/dev/null || echo)"
  NOW=$(date -u +%s)
  DEADLINE="$(_control deadline_epoch)"

  # --- 3. deadline hardkill -------------------------------------------------------------------
  if [[ -n "$DEADLINE" && "$DEADLINE" != "None" ]] && (( NOW >= DEADLINE + GRACE )); then
    log "deadline przekroczony o ${GRACE}s — twarde zatrzymanie"
    _set_control deadline_hardkill true
    _set_control halt true
    if _alive "$CHAIN_PID"; then
      kill -TERM "$CHAIN_PID" 2>/dev/null; sleep 5; kill -KILL "$CHAIN_PID" 2>/dev/null
    fi
    exit 0
  fi

  # --- 2. memory pressure ---------------------------------------------------------------------
  AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 99999)
  CUR="$(_control workers)"; CUR="${CUR:-4}"
  if (( AVAIL < MIN_AVAIL_MB )) && (( CUR > MIN_WORKERS )); then
    NEW=$(( CUR - 1 ))
    log "dostępne ${AVAIL}MB < ${MIN_AVAIL_MB}MB — obniżam workerów $CUR -> $NEW (zero swapu)"
    _set_control workers "$NEW"
  fi

  # --- 1. restart on a dead chain with a stale heartbeat ---------------------------------------
  if ! _alive "$CHAIN_PID"; then
    # A chain that finished normally leaves no pid to be alive either, so require the heartbeat to
    # be stale too: that is what distinguishes "died" from "done".
    if [[ -f "$HEARTBEAT" ]]; then
      HB_AGE=$(( NOW - $(stat -c %Y "$HEARTBEAT") ))
    else
      HB_AGE=$(( STALE + 1 ))
    fi
    DONE=$("$PY" -c "
import json,sys
try: s=json.load(open(sys.argv[1]))
except Exception: print(0); raise SystemExit
print(1 if any(v.get('status')=='failed' for v in s.values()) or
      len([v for v in s.values() if v.get('status')=='completed'])>=5 else 0)
" "$RUN_DIR/stages.json" 2>/dev/null || echo 0)
    if (( DONE == 1 )); then
      log "łańcuch zakończony (stages.json) — kończę"
      exit 0
    fi
    if (( HB_AGE >= STALE )); then
      if (( RESTARTS >= MAX_RESTARTS )); then
        log "wyczerpany limit restartów ($MAX_RESTARTS) — zatrzymuję i ustawiam halt"
        _set_control halt true
        exit 1
      fi
      if ( set -o noclobber; echo "$$" > "$LEASE" ) 2>/dev/null; then
        RESTARTS=$(( RESTARTS + 1 ))
        W="$(_control workers)"; W="${W:-4}"
        log "łańcuch martwy, heartbeat ${HB_AGE}s — restart $RESTARTS/$MAX_RESTARTS (workerów=$W)"
        "$PY" "$ROOT/scripts/methodology_chain.py" --run-dir "$RUN_DIR" --jobs "$W" --resume \
            >>"$RUN_DIR/chain.stdout.log" 2>&1 9>&- &
        echo "$!" > "$CHAIN_PIDF"
        rm -f "$LEASE"
      else
        log "lease zajęty — inny watchdog restartuje, pomijam"
      fi
    fi
  fi
done
