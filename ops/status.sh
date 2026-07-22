#!/usr/bin/env bash
# status.sh — what is the engine doing right now? Read-only.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python3}"
SESSION="golden-calibration"
hdr() { printf '\n\033[1m%s\033[0m\n' "$*"; }

hdr "sesja tmux"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "  '$SESSION' ŻYJE — okna: $(tmux list-windows -t "$SESSION" -F '#W' 2>/dev/null | tr '\n' ' ')"
  echo "  podgląd: tmux attach -t $SESSION (odczep: Ctrl-b d)"
else
  echo "  '$SESSION' nie istnieje"
fi

RUN_ID="$(cat "$ROOT/ops/.engine.current" 2>/dev/null || true)"
RUNDIR="$ROOT/runs/$RUN_ID"
[[ -n "$RUN_ID" && -d "$RUNDIR" ]] || { hdr "run"; echo "  brak aktywnego runu"; exit 0; }

hdr "run: $RUN_ID"
[[ -f "$RUNDIR/supervisor.json" ]] && "$PY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"  status={d.get('status')}  workerów={d.get('workers')}  start={d.get('started_utc')}\")" "$RUNDIR/supervisor.json"
[[ -f "$RUNDIR/control.json" ]] && sed 's/^/  control: /' "$RUNDIR/control.json" | tr -d '\n' | sed 's/$/\n/'

hdr "kolejka"
"$PY" - "$RUNDIR" <<'PYEOF' 2>/dev/null
import sys; sys.path.insert(0,"engine")
from taskqueue import Queue
print("  ", Queue(sys.argv[1]).counts())
PYEOF

hdr "stany assetów"
"$PY" "$ROOT/engine/planner.py" --run-dir "$RUNDIR" 2>/dev/null | grep -vE "dry-run|^asset" | sed 's/^/  /' | head -25

hdr "ledgery"
for l in exec_ledger method_ledger; do
  p="$RUNDIR/$l.jsonl"
  [[ -f "$p" ]] && "$PY" - "$p" <<'PYEOF' 2>/dev/null
import sys,collections; sys.path.insert(0,"scripts")
from ledger import Ledger
led=Ledger(sys.argv[1]); ok,bad=led.verify_chain()
c=collections.Counter(r['status'] for r in led.read_all())
print(f"  {sys.argv[1].split('/')[-1]:<20} {dict(c)}  łańcuch={'OK' if ok else f'ZŁY@{bad}'}")
PYEOF
done

hdr "maszyna"
free -m | awk 'NR<=2{printf "  %s\n",$0}'
echo
