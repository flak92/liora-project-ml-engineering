#!/usr/bin/env bash
# status.sh — what is the loop doing right now?
#
# Read-only. Answers, in one screen: is the tmux session up, is the lock held and by whom, what
# does the control channel say, which stages are done, how far the current stage has got through
# its ledger, and whether the machine has memory left. The source repo documented a status command
# that did not exist; this one is wired to `make loop-status`.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="$ROOT/xgb/data/runs"
LOCK="$ROOT/ops/.golden.lock"
PY="${PY:-$ROOT/.venv/bin/python3}"
SESSION="${SESSION:-liora-golden}"

hdr() { printf '\n\033[1m%s\033[0m\n' "$*"; }

hdr "sesja tmux"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "  '$SESSION' ŻYJE   (podgląd: tmux attach -t $SESSION, odczep: Ctrl-b d)"
else
  echo "  '$SESSION' nie istnieje"
fi

hdr "lock"
if command -v flock >/dev/null && [[ -f "$LOCK" ]]; then
  if flock -n 9 9>"$LOCK" 2>/dev/null; then
    echo "  wolny — żaden przebieg nie trwa"
  else
    echo "  ZAJĘTY przez: $(fuser "$LOCK" 2>/dev/null | tr -s ' ')"
  fi
else
  echo "  brak pliku lock (nigdy nie startował)"
fi

RUNDIR="$(ls -1dt "$RUNS"/golden_* 2>/dev/null | head -1)"
[[ -n "$RUNDIR" ]] || { hdr "przebiegi"; echo "  brak"; exit 0; }

hdr "ostatni przebieg: $(basename "$RUNDIR")"
[[ -f "$RUNDIR/supervisor.json" ]] && "$PY" -c "
import json,sys
d=json.load(open(sys.argv[1]))
print(f\"  status={d.get('status')}  sha={d.get('git_sha','')[:7]}  workerów={d.get('jobs')}  start={d.get('started_utc')}\")
" "$RUNDIR/supervisor.json"

for n in chain guard; do
  p="$(cat "$RUNDIR/$n.pid" 2>/dev/null || echo)"
  if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then echo "  $n: pid $p ŻYJE"
  else echo "  $n: pid ${p:-—} martwy"; fi
done
if [[ -f "$RUNDIR/heartbeat" ]]; then
  echo "  heartbeat: $(( $(date -u +%s) - $(stat -c %Y "$RUNDIR/heartbeat") ))s temu"
fi

hdr "control.json"
[[ -f "$RUNDIR/control.json" ]] && sed 's/^/  /' "$RUNDIR/control.json"

hdr "etapy"
if [[ -f "$RUNDIR/stages.json" ]]; then
  "$PY" -c "
import json,sys
for k,v in json.load(open(sys.argv[1])).items():
    print(f\"  {k:<20} {v['status']:<10} {v.get('detail','')[:60]}\")
" "$RUNDIR/stages.json"
else
  echo "  jeszcze żaden etap nie ruszył"
fi

hdr "postęp jednostek (ledgery)"
for l in "$RUNDIR"/*/ledger.jsonl; do
  [[ -f "$l" ]] || continue
  "$PY" -c "
import sys, collections
sys.path.insert(0,'$ROOT/scripts')
from ledger import Ledger
led=Ledger(sys.argv[1]); recs=led.read_all()
c=collections.Counter(r['status'] for r in recs)
ok,bad=led.verify_chain()
name=sys.argv[1].split('/')[-2]
print(f\"  {name:<12} {dict(c)}  łańcuch={'OK' if ok else f'USZKODZONY@{bad}'}\")
" "$l"
done

hdr "maszyna"
free -m | awk 'NR<=2{printf "  %s\n",$0}'
printf '  obciążenie:%s   rdzeni: %s\n' "$(cut -d' ' -f1-3 /proc/loadavg | sed 's/^/ /')" "$(nproc)"

hdr "ogon logu łańcucha"
tail -n 8 "$RUNDIR/chain.log" 2>/dev/null | sed 's/^/  /' || echo "  (brak)"
echo
