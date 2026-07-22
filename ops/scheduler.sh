#!/usr/bin/env bash
# scheduler.sh — a tmux loop with cron-like cadence. NOT system cron: a self-contained loop that
# survives with the tmux server, so nothing has to be installed into a user's crontab (the sibling
# hedging repo abandoned system cron for exactly this reason).
#
# It runs only TECHNICAL cadences — it decides nothing scientific. The planner it triggers is a pure
# function of artifacts and the contract; the scheduler just paces how often that function runs and
# how often the panels are rebuilt.
#
#   every 10 s   heartbeat (liveness for the guard)
#   every 10 min planner: derive states, enqueue the next allowed experiments
#   every 30 min reducer already runs inside the planner cycle; rebuild the run report
#   periodically nothing else — verify/backup is a human/make step, not a hot-loop concern
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python3}"

RUN_DIR=""
while [[ $# -gt 0 ]]; do case "$1" in --run-dir) RUN_DIR="$2"; shift 2 ;; *) shift ;; esac; done

PLAN_EVERY="${PLAN_EVERY:-600}"       # 10 min
REPORT_EVERY="${REPORT_EVERY:-1800}"  # 30 min
last_plan=0; last_report=0
HB="$RUN_DIR/heartbeat.scheduler"

_halted() { "$PY" -c "import json,sys;print(1 if json.load(open(sys.argv[1])).get('halt') else 0)" "$RUN_DIR/control.json" 2>/dev/null; }

echo "[scheduler] start (planner co ${PLAN_EVERY}s, report co ${REPORT_EVERY}s)"
# Plan once immediately so the first tasks appear without waiting a full cadence.
"$PY" "$ROOT/engine/planner.py" --run-dir "$RUN_DIR" --enqueue 2>/dev/null | tail -1
last_plan=$(date -u +%s)

while true; do
  sleep 10
  date -u +%s > "$HB"
  [[ "$(_halted)" == "1" ]] && { echo "[scheduler] halt — koniec"; break; }
  now=$(date -u +%s)
  if (( now - last_plan >= PLAN_EVERY )); then
    "$PY" "$ROOT/engine/planner.py" --run-dir "$RUN_DIR" --enqueue 2>/dev/null | tail -1
    last_plan=$now
  fi
  if (( now - last_report >= REPORT_EVERY )); then
    "$PY" "$ROOT/engine/report.py" --run-dir "$RUN_DIR" --out "$RUN_DIR/report.json" 2>/dev/null >/dev/null || true
    last_report=$now
  fi
done
