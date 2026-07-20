.PHONY: setup on off verify clean help

PY := .venv/bin/python3
ST := .venv/bin/streamlit
PORT ?= 8503

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

help:
	@echo "make setup   Install presentation dependencies"
	@echo "make on      Run the Streamlit presentation on port $(PORT)"
	@echo "make off     Stop whatever is listening on port $(PORT)"
	@echo "make verify  Recompute every artifact hash, and check the notebooks against the store"
	@echo "make clean   Remove local runtime cache"
	@echo ""
	@echo "Another port: make on PORT=8600  ·  make off PORT=8600"
