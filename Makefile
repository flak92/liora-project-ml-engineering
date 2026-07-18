.PHONY: setup app clean help

PY := .venv/bin/python3
ST := .venv/bin/streamlit

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

app:
	$(ST) run app.py --server.port 8503

clean:
	rm -rf __pycache__ app/__pycache__ app/pages/__pycache__ .streamlit/cache

help:
	@echo "make setup   Install presentation dependencies"
	@echo "make app     Run the Streamlit presentation"
	@echo "make clean   Remove local runtime cache"
