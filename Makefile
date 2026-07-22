.PHONY: setup on off verify clean help \
        methodology-report engine-plan engine-enqueue engine-start engine-smoke \
        engine-status engine-attach engine-stop engine-report engine-selftest \
        loop-start loop-status loop-attach loop-stop loop-kill loop-logs loop-selftest

PY := .venv/bin/python3
ST := .venv/bin/streamlit
PORT ?= 8503

OPS     := ops
SESSION ?= liora-golden
JOBS    ?= 4
HOURS   ?= 12

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

on:
	$(ST) run app.py --server.port $(PORT)

# Stops ONLY the process listening on our port, by pid. Never `pkill -f streamlit`:
# that pattern would also kill consoles belonging to other projects or other shells,
# and inside a compound command it can match — and kill — the caller's own shell.
off:
	@pid=$$(ss -ltnpH "sport = :$(PORT)" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2); \
	if [ -z "$$pid" ]; then \
		echo "nothing is listening on port $(PORT)"; \
	else \
		kill $$pid 2>/dev/null; \
		for i in 1 2 3 4 5; do \
			ss -ltnH "sport = :$(PORT)" 2>/dev/null | grep -q . || break; \
			sleep 1; \
		done; \
		if ss -ltnH "sport = :$(PORT)" 2>/dev/null | grep -q .; then \
			echo "pid $$pid did not release port $(PORT) — still listening"; exit 1; \
		else \
			echo "stopped the console on port $(PORT) (pid $$pid)"; \
		fi; \
	fi

# No virtualenv needed: both verifiers are stdlib only, so a reviewer can run them
# on a fresh clone before installing anything.
verify:
	python3 scripts/verify_artifacts.py
	python3 scripts/verify_notebooks.py
	python3 scripts/verify_figures.py --selftest
	python3 scripts/verify_figures.py

clean:
	rm -rf __pycache__ app/__pycache__ app/pages/__pycache__ .streamlit/cache

# --- the unattended Golden Calibration chain --------------------------------------------------
# Detached from the terminal by tmux: the tmux SERVER is the daemon, so closing the window, the
# ssh session or the editor leaves the run untouched. The supervisor takes a global lock and the
# chain resumes from its ledger, so a restart never repeats finished work.
loop-start:
	@tmux has-session -t $(SESSION) 2>/dev/null && \
		{ echo "sesja '$(SESSION)' już istnieje — podgląd: make loop-attach"; exit 1; } || true
	@tmux new-session -d -s $(SESSION) -c "$(CURDIR)" \
		"$(OPS)/loop.sh --jobs $(JOBS) --hours $(HOURS)"
	@echo "pętla wystartowana w tmux '$(SESSION)' (JOBS=$(JOBS) HOURS=$(HOURS))"
	@echo "stan: make loop-status   ·   podgląd: make loop-attach   ·   stop: make loop-stop"

# Resume the most recent run in place: reuses its stages.json + ledgers, so finished units are not
# recomputed. A failed or interrupted stage is re-entered; a fresh deadline is stamped.
loop-resume:
	@d=$$(ls -1dt xgb/data/runs/golden_* 2>/dev/null | head -1); \
	if [ -z "$$d" ]; then echo "brak przebiegu do wznowienia"; exit 1; fi; \
	id=$$(basename $$d); \
	tmux has-session -t $(SESSION) 2>/dev/null && \
		{ echo "sesja '$(SESSION)' już istnieje — make loop-attach"; exit 1; } || true; \
	tmux new-session -d -s $(SESSION) -c "$(CURDIR)" \
		"$(OPS)/loop.sh --resume-run $$id --jobs $(JOBS) --hours $(HOURS)"; \
	echo "wznowiono $$id w tmux '$(SESSION)' (JOBS=$(JOBS) HOURS=$(HOURS)); stan: make loop-status"

loop-status:
	@SESSION=$(SESSION) $(OPS)/status.sh

loop-attach:
	@tmux attach -t $(SESSION)

# Cooperative: sets halt in control.json. The current unit of work finishes, the ledger stays
# consistent, and the supervisor reports COMPLETED. Never `pkill` — it would match other projects'
# processes and, inside a compound command, the caller's own shell.
loop-stop:
	@d=$$(ls -1dt xgb/data/runs/golden_* 2>/dev/null | head -1); \
	if [ -z "$$d" ]; then echo "brak przebiegu do zatrzymania"; else \
		$(PY) -c "import json,os,sys; p=sys.argv[1]+'/control.json'; \
d=json.load(open(p)); d['halt']=True; d['halt_reason']='make loop-stop'; \
open(p+'.tmp','w').write(json.dumps(d,indent=1)); os.replace(p+'.tmp',p)" $$d; \
		echo "halt ustawiony w $$d/control.json — bieżąca jednostka zostanie dokończona"; \
	fi

loop-kill:
	@d=$$(ls -1dt xgb/data/runs/golden_* 2>/dev/null | head -1); \
	for f in chain guard; do \
		p=$$(cat $$d/$$f.pid 2>/dev/null); \
		[ -n "$$p" ] && kill -TERM $$p 2>/dev/null && echo "$$f (pid $$p) zatrzymany"; \
	done; \
	tmux kill-session -t $(SESSION) 2>/dev/null && echo "sesja '$(SESSION)' zamknięta" || true

loop-logs:
	@d=$$(ls -1dt xgb/data/runs/golden_* 2>/dev/null | head -1); \
	echo "== $$d/chain.log =="; tail -n 40 $$d/chain.log 2>/dev/null; \
	echo "== $$d/guard.log =="; tail -n 20 $$d/guard.log 2>/dev/null

loop-selftest:
	@$(PY) ops/selftest_loop.py

# Contract consistency: every split file tagged, the generated monolith in step with the splits,
# and no hardcoded constant drifted from its contract value. Run after editing config/contract/*.
lint-contract:
	@$(PY) scripts/contract_loader.py --regenerate
	@$(PY) scripts/contract_lint.py

# --- methodology execution engine (branch `methodology`) --------------------------------------
# Two ways to use the branch. PRESENTATION reads frozen artifacts and prints the funnel in a blink;
# REPRODUCTION runs the Calibration DAG per asset in a detached tmux session and produces new ones.
ASSETS  ?=
WORKERS ?= 4
HOURS   ?= 8

methodology-report:                 ## presentation: funnel + per-asset descriptions from the snapshot
	@$(PY) engine/report.py --snapshot --parity 26 11 9 4

engine-plan:                        ## deterministic plan (enqueue nothing); DRY_RUN=1 for explicit dry
	@d=$$(cat ops/.engine.current 2>/dev/null); [ -n "$$d" ] || { echo "brak runu; make engine-start"; exit 1; }; \
	$(PY) engine/planner.py --run-dir runs/$$d

engine-enqueue:                     ## plan, then write the next tasks to the queue
	@d=$$(cat ops/.engine.current 2>/dev/null); $(PY) engine/planner.py --run-dir runs/$$d --enqueue

engine-start:                       ## start the engine detached in tmux (ASSETS, WORKERS, HOURS)
	@ASSETS="$(ASSETS)" WORKERS=$(WORKERS) HOURS=$(HOURS) ALLOW_DIRTY=$${ALLOW_DIRTY:-0} \
		bash ops/engine.sh >runs/.engine-start.log 2>&1 & echo "engine startuje; make engine-status"

engine-smoke:                       ## full DAG on three assets, detached (validation run)
	@ASSETS="AZO ADBE GOOG" WORKERS=3 HOURS=6 ALLOW_DIRTY=1 \
		bash ops/engine.sh >runs/.engine-smoke.log 2>&1 & echo "smoke startuje; make engine-status"

engine-status:                      ## session, queue, per-asset states, ledgers, memory
	@bash ops/status.sh

engine-attach:                      ## watch the engine live (detach with Ctrl-b d)
	@tmux attach -t golden-calibration

engine-stop:                        ## cooperative halt — finishes the current tasks
	@d=$$(cat ops/.engine.current 2>/dev/null); \
	$(PY) -c "import json,os;p='runs/'+'$$d'+'/control.json';c=json.load(open(p));c['halt']=True;open(p+'.t','w').write(json.dumps(c));os.replace(p+'.t',p)" \
		&& echo "halt ustawiony (runs/$$d)"

engine-report:                      ## rebuild the run report from a live/finished run
	@d=$$(cat ops/.engine.current 2>/dev/null); $(PY) engine/report.py --run-dir runs/$$d

engine-selftest:                    ## prove queue claim atomicity, contract enforcement, resume, OOS
	@$(PY) engine/selftest.py

help:
	@echo "make setup        Install presentation dependencies"
	@echo "make on           Run the Streamlit presentation on port $(PORT)"
	@echo "make off          Stop whatever is listening on port $(PORT)"
	@echo "make verify       Recompute every artifact hash, and check the notebooks against the store"
	@echo "make clean        Remove local runtime cache"
	@echo ""
	@echo "Unattended research chain (survives closing the terminal):"
	@echo "  make loop-start    Start it detached in tmux '$(SESSION)'  (JOBS=$(JOBS) HOURS=$(HOURS))"
	@echo "  make loop-status   Session, lock, control, stages, ledger progress, memory"
	@echo "  make loop-attach   Watch it live            (detach with Ctrl-b d)"
	@echo "  make loop-stop     Cooperative halt — finishes the current unit of work"
	@echo "  make loop-kill     Hard stop and close the tmux session"
	@echo "  make loop-logs     Tail the chain and watchdog logs"
	@echo "  make loop-selftest Prove the lock, resume, atomicity and watchdog actually work"
	@echo ""
	@echo "  make lint-contract Regenerate the monolith from config/contract/*, then check consistency"
	@echo ""
	@echo "Another port: make on PORT=8600  ·  make off PORT=8600"
