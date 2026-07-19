"""The one call to Claude, and everything that can go wrong with it.

This is the only module in the repository that touches the network, and the only one that
imports `anthropic` — lazily, inside the call, so that `import formular` still succeeds on
a clone where `make setup` has not run. Every other page, and `make verify`, stay exactly
as offline as they were.

Two rules hold everywhere below:

1. No exception text ever reaches the screen. Every failure maps to a fixed sentence from
   MESSAGES. `str(exc)` on an SDK error can carry request bodies and headers, and the one
   thing this feature must never do is print a secret while explaining a mistake.
2. Nothing the model returns reaches the basket unchecked. Symbols are intersected with
   the store's own universe before anything is selected.
"""
import json

from . import env, prompt, questions

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096          # headroom so adaptive thinking cannot truncate the JSON
TIMEOUT_S = 60.0           # a hung call must fail calmly, not wedge the dialog

# Every message the user can ever see from this module. Fixed sentences with named slots:
# only integers (a status code, a count, a retry delay) are ever interpolated.
MESSAGES = {
    "no_sdk": "The adviser package is not installed here. Run `make setup` (or "
              "`pip install -r requirements.txt`) to add it. Nothing else in this console "
              "depends on it.",
    "no_key": "No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env` and put your key "
              "in it. Every other page works without one.",
    "auth": "The API key was rejected (401). Check the key in your .env file. This app "
            "never displays, logs or stores the key.",
    "forbidden": "This key does not have access to the model this page uses (403).",
    "not_found": "The model this page uses was not found for this key (404).",
    "rate_limit": "Rate limited (429) — the client already retried. Give it a moment and "
                  "submit again. Nothing in the basket changed.",
    "server": "The API returned {code}, a server-side error. Nothing in the basket changed.",
    "status": "The API rejected the request ({code}). Nothing in the basket changed.",
    "connection": "Could not reach the API — no network, or the request timed out. "
                  "Nothing in the basket changed.",
    "refusal": "The model declined to answer this request. Nothing in the basket changed.",
    "truncated": "The adviser's answer was cut off before it finished. Nothing in the "
                 "basket changed.",
    "bad_json": "The adviser's answer did not parse into the expected shape. Nothing in "
                "the basket changed.",
    "no_valid_tickers": "The adviser named {n} symbol(s), none of which exist in this "
                        "study's universe. Nothing in the basket changed.",
}


class AdviserError(Exception):
    """Carries a sentence meant for a human, in .calm. Never carries a cause."""

    def __init__(self, calm):
        super().__init__(calm)
        self.calm = calm


def _sdk():
    """The anthropic module, or None. Imported here and nowhere else."""
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic


def available():
    """(ready, reason). Reason is "" when ready — otherwise a sentence for st.info."""
    if _sdk() is None:
        return False, MESSAGES["no_sdk"]
    if not env.api_key():
        return False, MESSAGES["no_key"]
    return True, ""


def _raise_for(anthropic, exc):
    """Map an SDK exception to a fixed sentence. Most specific first."""
    if isinstance(exc, anthropic.AuthenticationError):
        raise AdviserError(MESSAGES["auth"]) from None
    if isinstance(exc, anthropic.PermissionDeniedError):
        raise AdviserError(MESSAGES["forbidden"]) from None
    if isinstance(exc, anthropic.NotFoundError):
        raise AdviserError(MESSAGES["not_found"]) from None
    if isinstance(exc, anthropic.RateLimitError):
        raise AdviserError(MESSAGES["rate_limit"]) from None
    if isinstance(exc, anthropic.APIStatusError):
        code = int(getattr(exc, "status_code", 0) or 0)
        key = "server" if code >= 500 else "status"
        raise AdviserError(MESSAGES[key].format(code=code)) from None
    if isinstance(exc, anthropic.APIConnectionError):
        raise AdviserError(MESSAGES["connection"]) from None
    raise AdviserError(MESSAGES["connection"]) from None


def _request(answers, universe):
    """Send the questionnaire, return the parsed JSON document."""
    anthropic = _sdk()
    if anthropic is None:
        raise AdviserError(MESSAGES["no_sdk"])
    key = env.api_key()
    if not key:
        raise AdviserError(MESSAGES["no_key"])

    client = anthropic.Anthropic(api_key=key, timeout=TIMEOUT_S, max_retries=2)
    try:
        # No temperature/top_p/top_k and no thinking block: this model rejects explicit
        # thinking config and non-default sampling, and runs adaptive thinking by default.
        # effort="low" keeps the spinner short — the task is a pick from a list, not an
        # analysis — and output_config.format guarantees the first text block is JSON.
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=prompt.SYSTEM,
            messages=[{"role": "user",
                       "content": prompt.build_user_message(answers, universe)}],
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": prompt.RESPONSE_SCHEMA},
            },
        )
    except Exception as exc:                      # noqa: BLE001 — mapped, never re-raised
        _raise_for(anthropic, exc)

    # Checked before content is touched: on a refusal the content list can be empty, and
    # indexing it first would turn a policy stop into an IndexError on stage.
    if getattr(response, "stop_reason", None) == "refusal":
        raise AdviserError(MESSAGES["refusal"])
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise AdviserError(MESSAGES["truncated"])

    try:
        text = next(b.text for b in response.content if getattr(b, "type", "") == "text")
        return json.loads(text)
    except (StopIteration, AttributeError, ValueError):
        raise AdviserError(MESSAGES["bad_json"]) from None


def interpret(doc, answers, universe):
    """Turn a raw response into something safe to put on screen and into the basket.

    Split from _request so the whole validation path is testable without a network, a key
    or an SDK — selfcheck.py drives it with canned documents.
    """
    if not isinstance(doc, dict) or not isinstance(doc.get("picks"), list):
        raise AdviserError(MESSAGES["bad_json"])

    known = set(universe)
    seen, kept, dropped, mismatched = set(), [], [], []
    from . import sectors                                    # local: keeps the import graph flat

    for item in doc["picks"]:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("ticker", "")).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        if symbol not in known:
            dropped.append(symbol)
            continue
        kept.append({"ticker": symbol,
                     "sector": sectors.sector_of(symbol),
                     "reason": str(item.get("reason", "")).strip()})
        declared = str(item.get("sector", "")).strip()
        if declared and declared != sectors.sector_of(symbol):
            mismatched.append((symbol, declared, sectors.sector_of(symbol)))

    # The count is a hard constraint the schema cannot carry, so it is enforced here.
    _, hi = questions.POSITION_RANGE[answers["q8"]]
    trimmed = [p["ticker"] for p in kept[hi:]]
    kept = kept[:hi]

    if not kept:
        raise AdviserError(MESSAGES["no_valid_tickers"].format(n=len(dropped)))

    return {
        "picks": kept,
        "tickers": [p["ticker"] for p in kept],
        "dropped": dropped,
        "trimmed": trimmed,
        "mismatched": mismatched,
        "note": str(doc.get("note", "")).strip(),
        "profile_summary": str(doc.get("profile_summary", "")).strip(),
    }


def call_adviser(answers, universe):
    """Ask the adviser, validate the answer. The one entry point ui.py uses.

    ui.py reaches this as `advisor.call_adviser`, an attribute lookup at call time, so the
    offline checks can rebind it. Do not let a caller do `from .advisor import
    call_adviser` — that would bind the function object and make the page untestable
    without a key and a network.
    """
    return interpret(_request(answers, universe), answers, universe)
