#!/usr/bin/env bash
# worker.sh — one tmux window's worker loop: claim a task, run it, repeat.
#
# The loop is deliberately dumb. All the intelligence is in worker.py (which runs one task) and the
# planner (which decides what tasks exist). This just keeps pulling until the run halts, honours a
# lowered worker count from the guard (a window whose index exceeds the current worker count idles
# rather than competing for the last cores), and writes a heartbeat the guard reads for liveness.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python3}"

RUN_DIR=""; WORKER="worker-00"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --worker)  WORKER="$2";  shift 2 ;;
    *) shift ;;
  esac
done
IDX=$(( 10#$(printf '%s' "$WORKER" | grep -o '[0-9]*$') ))
HB="$RUN_DIR/heartbeat.$WORKER"

_ctl() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$RUN_DIR/control.json" "$1" 2>/dev/null; }

while true; do
  date -u +%s > "$HB"
  [[ "$(_ctl halt)" == "True" ]] && { echo "[$WORKER] halt — koniec"; break; }
  W="$(_ctl workers)"; W="${W:-4}"
  if (( IDX > W )); then sleep 10; continue; fi          # guard obniżył workerów — to okno idle
  # Claim and run exactly one task; worker.py prints the outcome. Empty queue -> short idle.
  OUT="$("$PY" "$ROOT/engine/worker.py" --run-dir "$RUN_DIR" --worker "$WORKER" 2>/dev/null)"
  echo "[$WORKER] $OUT"
  [[ "$OUT" == "brak zadań" ]] && sleep 5
done
