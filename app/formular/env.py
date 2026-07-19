"""Read an API key from the environment or a local .env, on the standard library alone.

Deliberately NOT a dotenv implementation. No ${VAR} expansion, no multi-line values, no
inheritance, no export semantics beyond stripping the word. It reads `KEY=value` lines and
stops there, which is all a single API key needs — and it keeps the presentation branch's
dependency list honest, since `make setup` is meant to install a console, not a runtime.

Nothing here ever logs, prints or returns a value for display. api_key() returns the key
to the one caller that needs it; everything else asks whether a key EXISTS.
"""
import os
from pathlib import Path

VAR = "ANTHROPIC_API_KEY"

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Repo root first, because that is where .env.example sits and where people look. The
# folder-local path exists so the feature can be made entirely self-contained; both are
# covered by the unanchored .gitignore rules.
CANDIDATES = (ROOT / ".env", HERE / ".env")


def load(path):
    """Parse one .env file. A missing or unreadable file is an empty dict, not an error."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return {}

    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, value = line.partition("=")          # first "=" only: values may contain it
        key, value = key.strip(), value.strip()
        if not key.isidentifier():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def api_key():
    """The key, or "" if there is none. A real environment variable always wins, so
    `ANTHROPIC_API_KEY=… make on` works and no file has to exist."""
    from_env = os.environ.get(VAR, "").strip()
    if from_env:
        return from_env
    for path in CANDIDATES:
        value = load(path).get(VAR, "").strip()
        if value:
            return value
    return ""
