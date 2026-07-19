"""Offline gates for the Formular add-on. No network, no key, no Streamlit runtime.

    .venv/bin/python3 app/formular/selfcheck.py

The load-bearing one is check_no_result_leak: it takes a fully built message, subtracts
every word that the fixed template, the questionnaire and the sector map could have
contributed, and requires the remainder to be nothing but ticker symbols. That is what
turns "the adviser does not see the results" from a claim in a docstring into something a
reviewer can re-run.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data                                                          # noqa: E402
from formular import advisor, env, prompt, questions, sectors        # noqa: E402

FAILURES = []

ANSWERS = {
    "q1": "Some experience",
    "q2": "1-12 months",
    "q3": "20%",
    "q4": "Hold",
    "q5": "5,000-20,000",
    "q6": ["Energy"],
    "q7": ["Information Technology", "Health Care"],
    "q8": "Moderate (4-7)",
    "q9": "Balance steady moderate gains against moderate risk",
}

# Column and concept names that only exist on the results side of this project. None of
# them may appear anywhere in what the adviser is sent.
RESULT_WORDS = ("hodl", "beats_hodl", "return_pct", "profit_factor", "end_capital",
                "oos_window", "max_drawdown", "model_trades", "trade_floor",
                "result_mode", "sharpe", "win_rate", "equity")


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'' if condition else f' — {detail}'}")
    if not condition:
        FAILURES.append(name)


def check_sector_coverage():
    universe = data.tickers()
    mapped, unmapped, extra = sectors.coverage(universe)
    check("sector map covers the universe 1:1",
          len(mapped) == len(universe) and not unmapped and not extra,
          f"unmapped={unmapped} extra={extra}")
    check("every sector label is one of the eleven",
          set(sectors.SECTOR_OF.values()) <= set(sectors.GICS_11))
    empty = [s for s in questions.SECTORS_OFFERED if not sectors.BY_SECTOR.get(s)]
    check("every offered sector has tickers behind it", not empty, f"empty={empty}")


def _walk_schema(node, path="root"):
    """Objects must close themselves off and declare every property required; array
    length keywords are silently unsupported by structured outputs and would 400."""
    problems = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            props = set(node.get("properties", {}))
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties is not False")
            if set(node.get("required", [])) != props:
                problems.append(f"{path}: required does not name every property")
        for bad in ("minItems", "maxItems", "minLength", "maxLength", "pattern"):
            if bad in node:
                problems.append(f"{path}: unsupported keyword {bad}")
        for key, value in node.items():
            problems += _walk_schema(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            problems += _walk_schema(value, f"{path}[{i}]")
    return problems


def check_schema():
    problems = _walk_schema(prompt.RESPONSE_SCHEMA)
    check("response schema is well formed for structured outputs",
          not problems, "; ".join(problems))
    enum = prompt.RESPONSE_SCHEMA["properties"]["picks"]["items"]["properties"]["sector"]
    check("sector is constrained to the map's own vocabulary",
          enum.get("enum") == list(sectors.GICS_11))


def check_prompt_clauses():
    required = ("You do not know what happened",
                "Do not claim, imply or hint",
                "Do not invent figures",
                "Do not justify a pick by its results",
                "first trading day of\nJanuary 2024")
    missing = [c for c in required if c not in prompt.SYSTEM]
    check("system prompt still carries every blindfold clause",
          not missing, f"missing={missing}")


def check_no_result_leak():
    universe = data.tickers()
    message = prompt.build_user_message(ANSWERS, universe)
    haystack = (prompt.SYSTEM + "\n" + message).lower()

    hits = [w for w in RESULT_WORDS if w in haystack]
    check("no result-bearing term appears in what the adviser is sent",
          not hits, f"found={hits}")

    # Everything the fixed scaffolding is allowed to contribute.
    corpus = " ".join((
        prompt.HEADER, prompt.FOOTER,
        " ".join(q.prompt for q in questions.QUESTIONS),
        " ".join(o for q in questions.QUESTIONS for o in q.options),
        " ".join(sectors.GICS_11),
        " ".join(str(v) for pair in questions.POSITION_RANGE.values() for v in pair),
        "Choose between and tickers honouring answer",
        "These are the tickers you may choose from grouped by sector",
    ))
    allowed = set(re.findall(r"[A-Za-z0-9.]+", corpus.lower()))
    allowed |= {t.lower() for t in universe}
    # The only numbers the template itself produces: the 1..9 enumeration and the size of
    # the universe. Deliberately NOT a blanket allowance for digits — a leaked figure has
    # to show up here as an unexplained token, which is the entire point of this check.
    allowed |= {f"{i}." for i in range(1, len(questions.QUESTIONS) + 1)}
    allowed |= {str(len(universe))}

    unexplained = sorted({t for t in re.findall(r"[A-Za-z0-9.]+", message.lower())
                          if t not in allowed})
    check("the message contains nothing but template, answers and symbols",
          not unexplained, f"unexplained={unexplained[:12]}")


def check_unknown_tickers():
    universe = data.tickers()
    doc = {"profile_summary": "p", "note": "n", "picks": [
        {"ticker": "AAPL", "sector": "Information Technology", "reason": "r"},
        {"ticker": "NOTREAL", "sector": "Financials", "reason": "r"},
        {"ticker": "aapl", "sector": "Information Technology", "reason": "duplicate"},
    ]}
    out = advisor.interpret(doc, ANSWERS, universe)
    check("unknown symbols never reach the basket",
          out["tickers"] == ["AAPL"] and out["dropped"] == ["NOTREAL"],
          f"tickers={out['tickers']} dropped={out['dropped']}")

    only_bad = {"picks": [{"ticker": "NOPE", "sector": "Financials", "reason": "r"}]}
    try:
        advisor.interpret(only_bad, ANSWERS, universe)
        check("a basket of nothing but unknowns is refused", False, "no error raised")
    except advisor.AdviserError as exc:
        check("a basket of nothing but unknowns is refused",
              exc.calm == advisor.MESSAGES["no_valid_tickers"].format(n=1))


def check_position_cap():
    universe = data.tickers()[:14]
    doc = {"picks": [{"ticker": t, "sector": sectors.sector_of(t), "reason": "r"}
                     for t in universe]}
    few = dict(ANSWERS, q8="Few (1-3)")
    out = advisor.interpret(doc, few, universe)
    check("the basket is trimmed to what the profile allows",
          len(out["tickers"]) == 3 and len(out["trimmed"]) == 11,
          f"kept={len(out['tickers'])} trimmed={len(out['trimmed'])}")


def check_malformed():
    for name, doc in (("not a dict", "{not json"), ("no picks", {"note": "n"}),
                      ("picks not a list", {"picks": "AAPL"})):
        try:
            advisor.interpret(doc, ANSWERS, data.tickers())
            check(f"malformed response refused ({name})", False, "no error raised")
        except advisor.AdviserError as exc:
            check(f"malformed response refused ({name})",
                  exc.calm == advisor.MESSAGES["bad_json"])


def check_message_hygiene():
    bad = [k for k, v in advisor.MESSAGES.items() if "sk-" in v or "%s" in v]
    check("no error message can carry a key or a raw format", not bad, f"bad={bad}")
    slots = {"code", "n", "u", "retry_after"}
    stray = [k for k, v in advisor.MESSAGES.items()
             if set(re.findall(r"\{(\w+)\}", v)) - slots]
    check("error messages only interpolate known integer slots", not stray, f"stray={stray}")


def check_env(tmp):
    fixture = tmp / ".env.fixture"
    fixture.write_text('# comment\n\nexport FOO="bar"\nA=B=C\nnot a line\n'
                       "SPACED = padded \n", encoding="utf-8")
    parsed = env.load(fixture)
    check("the .env parser handles quotes, export, comments and '=' in values",
          parsed == {"FOO": "bar", "A": "B=C", "SPACED": "padded"}, f"parsed={parsed}")
    check("a missing .env is empty, not an error", env.load(tmp / "nope.env") == {})


def check_no_key_state(monkey_env):
    ready, reason = advisor.available()
    if monkey_env:
        check("with no key the feature explains itself instead of failing",
              not ready and reason == advisor.MESSAGES["no_key"], f"reason={reason!r}")
    else:
        print("  SKIP  no-key state (a key is present in this environment)")


def main():
    import os
    import tempfile

    print("Formular self-check")
    check_sector_coverage()
    check_schema()
    check_prompt_clauses()
    check_no_result_leak()
    check_unknown_tickers()
    check_position_cap()
    check_malformed()
    check_message_hygiene()

    with tempfile.TemporaryDirectory() as tmp:
        check_env(Path(tmp))

    had_key = bool(env.api_key())
    os.environ.pop(env.VAR, None)
    saved, env.CANDIDATES = env.CANDIDATES, ()
    try:
        check_no_key_state(True)
    finally:
        env.CANDIDATES = saved
    if had_key:
        print("  note: a key exists in this environment; the no-key path was exercised "
              "with the lookup paths disabled.")

    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
