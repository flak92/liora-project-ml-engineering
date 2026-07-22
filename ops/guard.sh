#!/usr/bin/env bash
# guard.sh — the engine's watchdog. Technical only: it may move a stuck task back to the queue, lower
# the worker count under memory pressure, or hard-stop at the deadline. It may NOT touch anything
# scientific — not a threshold, not the HPO space, not the null, not the data boundary, not which
# asset runs. A task it requeues is retried UNCHANGED, under the same frozen contract.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python3}"

RUN_DIR=""
while [[ $# -gt 0 ]]; do case "$1" in --run-dir) RUN_DIR="$2"; shift 2 ;; --root) shift 2 ;; *) shift ;; esac; done

TICK="${GUARD_TICK_SEC:-30}"
GRACE="${GUARD_DEADLINE_GRACE_SEC:-120}"
MIN_AVAIL_MB="${GUARD_MIN_AVAIL_MB:-700}"
MIN_WORKERS="${GUARD_MIN_WORKERS:-2}"
STALE_TASK="${GUARD_STALE_TASK_SEC:-5400}"   # longer than any single task (Rung 5 ~35-40 min)

CONTROL="$RUN_DIR/control.json"
_ctl() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$CONTROL" "$1" 2>/dev/null; }
_set() { "$PY" - "$CONTROL" "$1" "$2" <<'PYEOF'
import json,sys,os
p,k,v=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(p))
try: v=json.loads(v)
except json.JSONDecodeError: pass
d[k]=v; open(p+".t","w").write(json.dumps(d,indent=1)); os.replace(p+".t",p)
PYEOF
}

log() { printf '[%s] guard: %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "start (tick=${TICK}s min_avail=${MIN_AVAIL_MB}MB stale_task=${STALE_TASK}s)"

while true; do
  sleep "$TICK"
  [[ -f "$CONTROL" ]] || { log "brak control.json — koniec"; exit 0; }
  [[ "$(_ctl halt)" == "True" ]] && { log "halt — koniec"; exit 0; }
  NOW=$(date -u +%s)

  # deadline
  DL="$(_ctl deadline_epoch)"
  if [[ -n "$DL" && "$DL" != "None" ]] && (( NOW >= DL + GRACE )); then
    log "deadline +${GRACE}s — halt"; _set deadline_hardkill true; _set halt true; exit 0
  fi

  # memory degrade (RAM + swap headroom, like the main guard)
  AVAIL=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 99999)
  SWAP=$(awk '/SwapFree/{print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 0)
  HEAD=$(( AVAIL + SWAP )); CUR="$(_ctl workers)"; CUR="${CUR:-4}"
  if (( HEAD < MIN_AVAIL_MB )) && (( CUR > MIN_WORKERS )); then
    log "zapas ${HEAD}MB < ${MIN_AVAIL_MB} — workerów $CUR -> $(( CUR - 1 ))"; _set workers $(( CUR - 1 ))
  fi

  # stale task requeue: a running task older than STALE_TASK is an orphan of a dead worker.
  for f in "$RUN_DIR"/queue/running/*.json; do
    [[ -e "$f" ]] || continue
    age=$(( NOW - $(stat -c %Y "$f") ))
    if (( age >= STALE_TASK )); then
      mv "$f" "$RUN_DIR/queue/pending/$(basename "$f")" && log "requeue zawieszonego zadania $(basename "$f" .json) (wiek ${age}s)"
    fi
  done
done
