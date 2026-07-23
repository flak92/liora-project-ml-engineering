.PHONY: setup on off verify clean help lint-contract verify-calibration-docs replay verify-replay \
        methodology-report engine-selftest \
        iteration-start iteration-status iteration-plan iteration-report iteration-stop \
        iteration-smoke iteration-selftest \
        loop-start loop-status loop-attach loop-stop loop-kill loop-logs loop-selftest

PY := .venv/bin/python3
ST := .venv/bin/streamlit
PORT ?= 8503

OPS     := ops
SESSION ?= liora-golden
JOBS    ?= 4
LOOP_HOURS ?= 12

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
		"$(OPS)/loop.sh --jobs $(JOBS) --hours $(LOOP_HOURS)"
	@echo "pętla wystartowana w tmux '$(SESSION)' (JOBS=$(JOBS) HOURS=$(LOOP_HOURS))"
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
		"$(OPS)/loop.sh --resume-run $$id --jobs $(JOBS) --hours $(LOOP_HOURS)"; \
	echo "wznowiono $$id w tmux '$(SESSION)' (JOBS=$(JOBS) HOURS=$(LOOP_HOURS)); stan: make loop-status"

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

verify-calibration-docs:            ## fail if the calibration docs' seal drifted from contract/snapshot
	@$(PY) scripts/verify_calibration_docs.py

# The methodology replay: a BUILT artifact reconstructed from the snapshot panels — never hand-edited.
# `replay` regenerates methodology_replay.html; `verify-replay` fails if its embedded seal (contract
# hash, funnel, live guard message) drifted from the current contract/snapshot/guard.
replay:                             ## build methodology_replay.html from the snapshot (never hand-edit it)
	@$(PY) scripts/build_replay.py

verify-replay:                      ## fail if the built replay's seal drifted from contract/snapshot/guard
	@$(PY) scripts/verify_replay.py

# --- methodology (branch `methodology`) --------------------------------------------------------
# Two ways to use the branch. PRESENTATION reads frozen artifacts and prints the funnel in a blink;
# REPRODUCTION walks the ladder per asset in a detached tmux session (make iteration-start), driving
# each frozen contract version to a fixpoint with the parallel-over-assets driver.
ASSETS  ?=
WORKERS ?= 4

methodology-report:                 ## presentation: funnel + per-asset descriptions from the snapshot
	@$(PY) engine/report.py --snapshot --parity 26 11 9 2

engine-selftest:                    ## prove execution guarantees: contract gate, idempotent publish, ledgers, OOS
	@$(PY) engine/selftest.py

# --- Iterative Calibration Loop (ladder of frozen contract versions) ---------------------------
# The outer loop: walk a human-pre-authorized ladder of frozen contract versions, drive each to a
# fixpoint by reusing the engine per epoch, stop when a new hypothesis adds no confirmed feature.
# The proof standard is frozen across the whole ladder (engine/contract_patch.py enforces it); only
# the hypothesis space varies. Detached like the engine; stop is cooperative, never pkill.
iteration-start:                    ## walk the pre-authorized ladder, detached in tmux (ASSETS, WORKERS)
	@ASSETS="$(ASSETS)" WORKERS=$(WORKERS) ALLOW_DIRTY=$${ALLOW_DIRTY:-0} bash ops/iteration_loop.sh

iteration-status:                   ## ladder progress (epochs, convergence, budget) + tmux liveness
	@d=$$(cat ops/.iteration.current 2>/dev/null); [ -n "$$d" ] || { echo "brak drabiny; make iteration-start"; exit 1; }; \
	tmux has-session -t iterative-calibration 2>/dev/null && echo "sesja iterative-calibration: ŻYWA" || echo "sesja iterative-calibration: brak"; \
	$(PY) engine/iteration_planner.py --ladder-dir runs/$$d --status

iteration-plan:                     ## print the ladder (guard-checked), compute nothing
	@$(PY) engine/iteration_planner.py --ladder-dir /tmp/iteration-plan --plan-only

iteration-report:                   ## (re)generate iteration_summary.md from the ladder's artifacts
	@d=$$(cat ops/.iteration.current 2>/dev/null); [ -n "$$d" ] || { echo "brak drabiny; make iteration-start"; exit 1; }; \
	$(PY) engine/iteration_report.py --ladder-dir runs/$$d

iteration-stop:                     ## cooperative halt — finishes the current epoch, stops the ladder
	@d=$$(cat ops/.iteration.current 2>/dev/null); [ -n "$$d" ] || { echo "brak drabiny do zatrzymania"; exit 1; }; \
	$(PY) -c "import json,os;p='runs/'+'$$d'+'/control.json';c=json.load(open(p));c['halt']=True;open(p+'.t','w').write(json.dumps(c));os.replace(p+'.t',p)" && echo "halt drabiny ustawiony (runs/$$d)"

iteration-smoke:                    ## FAST dev gate: reduced null strength (mechanics only, 0 confirmations)
	@id=icl_smoke_$$(git rev-parse --short HEAD); \
	RESEARCH_SMOKE_PERMS=5 RESEARCH_SMOKE_FOLDS=1 \
	$(PY) engine/iteration_planner.py --ladder-dir runs/$$id --assets AZO ADBE GOOG --allow-dirty && \
	$(PY) engine/iteration_report.py --ladder-dir runs/$$id; \
	echo "UWAGA: smoke = obniżona siła (perms=5, folds=1) → waliduje ORKIESTRACJĘ; guardrail wymusza 0 potwierdzeń. Pełna nauka: make iteration-start (bez RESEARCH_SMOKE_*)."

iteration-selftest:                 ## engine guarantees + ladder guard, convergence, repair, budget
	@$(PY) engine/iteration_selftest.py

help:
	@echo "make setup        Install presentation dependencies"
	@echo "make on           Run the Streamlit presentation on port $(PORT)"
	@echo "make off          Stop whatever is listening on port $(PORT)"
	@echo "make verify       Recompute every artifact hash, and check the notebooks against the store"
	@echo "make clean        Remove local runtime cache"
	@echo ""
	@echo "Unattended research chain (survives closing the terminal):"
	@echo "  make loop-start    Start it detached in tmux '$(SESSION)'  (JOBS=$(JOBS) HOURS=$(LOOP_HOURS))"
	@echo "  make loop-status   Session, lock, control, stages, ledger progress, memory"
	@echo "  make loop-attach   Watch it live            (detach with Ctrl-b d)"
	@echo "  make loop-stop     Cooperative halt — finishes the current unit of work"
	@echo "  make loop-kill     Hard stop and close the tmux session"
	@echo "  make loop-logs     Tail the chain and watchdog logs"
	@echo "  make loop-selftest Prove the lock, resume, atomicity and watchdog actually work"
	@echo ""
	@echo "  make lint-contract Regenerate the monolith from config/contract/*, then check consistency"
	@echo ""
	@echo "Iterative Calibration Loop (walks a pre-authorized ladder of frozen contract versions):"
	@echo "  make iteration-start    Detached ladder walk in tmux 'iterative-calibration' (ASSETS, WORKERS)"
	@echo "  make iteration-status   Epochs, convergence, budget, tmux liveness"
	@echo "  make iteration-plan     Print the guard-checked ladder; compute nothing"
	@echo "  make iteration-report   (Re)generate iteration_summary.md — methodology + corrections"
	@echo "  make iteration-stop     Cooperative halt — finishes the current epoch, stops the ladder"
	@echo "  make iteration-selftest Prove the safety kernel, convergence, repair and budget"
	@echo ""
	@echo "Another port: make on PORT=8600  ·  make off PORT=8600"
