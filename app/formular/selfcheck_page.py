"""Drive the Formular modal through AppTest, with the API call replaced.

    .venv/bin/python3 app/formular/selfcheck_page.py

Exactly one AppTest lives in this process: a second one in the same interpreter segfaults
inside pyarrow, which is why the console's page checks are one-per-process throughout.

The whole reason this file can exist offline is that ui.py calls `advisor.call_adviser`
as an attribute lookup rather than binding the function at import. Rebinding the module
attribute below is therefore enough to take the network, the key and the SDK out of the
picture while still exercising the real dialog, the real form and the real state writes.

Two harness quirks decide the order of what follows, and neither is a defect in the page:

* Dialogs are drawn on the "event" delta path, which AppTest does NOT clear between runs.
  `at.get("dialog")` therefore keeps returning the block after the modal has closed, and
  the honest signal for "is it open" is the `fm_open` flag that governs whether the dialog
  function gets called at all. Worse, the stale dialog leaves orphaned radio nodes in the
  tree whose session-state entries Streamlit has already collected, so the NEXT `.run()`
  dies with a KeyError. So: every failure path is exercised first, while the modal is
  legitimately open, and the success that closes it comes last.
* The result metrics are `components.metric_row`, which is styled HTML rather than
  `st.metric`, so they are asserted on the rendered markup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest                              # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAGE = str(ROOT / "app" / "pages" / "simulator.py")

FAILURES = []

ANSWERS = {
    "fm_q1": "Some experience",
    "fm_q2": "1-12 months",
    "fm_q3": "20%",
    "fm_q4": "Hold",
    "fm_q5": "5,000-20,000",
    "fm_q6": ["Energy"],
    "fm_q7": ["Information Technology"],
    "fm_q8": "Moderate (4-7)",
    "fm_q9": "Balance steady moderate gains against moderate risk",
}

PICKS = ("AAPL", "JNJ", "MSFT", "PG", "XOM")


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{'' if condition else f' — {detail}'}")
    if not condition:
        FAILURES.append(name)


def state(at, key, default=None):
    try:
        return at.session_state[key]
    except (KeyError, AttributeError):
        return default


def fill(at):
    for key, value in ANSWERS.items():
        if isinstance(value, list):
            at.multiselect(key=key).set_value(value)
        else:
            at.radio(key=key).set_value(value)


def tree_text(at):
    return " ".join(str(getattr(e, "value", "")) for e in at.get("markdown")
                    + at.get("caption") + at.get("error"))


def _boom(formular):
    def raiser(answers, universe):
        raise formular.advisor.AdviserError(formular.advisor.MESSAGES["auth"])
    return raiser


def main():
    print("Formular page check (AppTest, adviser stubbed)")
    at = AppTest.from_file(PAGE, default_timeout=60)
    at.run()
    check("the simulator page opens with the add-on present", not at.exception,
          str(at.exception))

    formular = sys.modules.get("formular")
    check("the add-on was imported by the page", formular is not None)
    if formular is None:
        return 1

    calls = []

    def fake(answers, universe):
        calls.append(answers)
        return {"picks": [{"ticker": t, "sector": "—", "reason": "profile fit"}
                          for t in PICKS],
                "tickers": list(PICKS), "dropped": ["BRK.A"], "trimmed": [],
                "mismatched": [], "note": "A spread basket.",
                "profile_summary": "Moderate, medium horizon."}

    formular.advisor.call_adviser = fake

    at.button(key="fm_btn").click().run()
    check("clicking Formular opens the modal",
          len(at.get("dialog")) == 1 and state(at, "fm_open") is True)

    # --- failure paths first, while the modal is legitimately open ------------------

    # Nothing answered.
    at.button(key="fm_submit").click().run()
    check("an empty form is refused without closing the modal",
          state(at, "fm_open") is True and len(at.error) >= 1 and not calls,
          f"errors={len(at.error)} calls={len(calls)}")

    # The same sector on both the avoid and the prioritize list.
    fill(at)
    at.multiselect(key="fm_q7").set_value(["Energy"])
    at.button(key="fm_submit").click().run()
    check("a sector on both lists is refused without calling the adviser",
          state(at, "fm_open") is True and len(at.error) >= 1 and not calls,
          f"errors={len(at.error)} calls={len(calls)}")

    # The API refuses. The modal must stay open and say one fixed sentence.
    formular.advisor.call_adviser = _boom(formular)
    fill(at)
    at.button(key="fm_submit").click().run()
    check("an API failure keeps the modal open", state(at, "fm_open") is True)
    check("the failure is a fixed sentence, not an exception",
          any(e.value == formular.advisor.MESSAGES["auth"] for e in at.error),
          [e.value for e in at.error])
    check("a failed call leaves the basket alone",
          not state(at, "basket") and state(at, "stage") == "pick",
          f"basket={state(at, 'basket')} stage={state(at, 'stage')}")

    # --- the success closes the modal, so it goes last ------------------------------

    formular.advisor.call_adviser = fake
    fill(at)
    at.button(key="fm_submit").click().run()
    check("a complete form reaches the adviser exactly once", len(calls) == 1,
          f"calls={len(calls)}")
    check("the adviser received the nine answers, keyed by question",
          calls and set(calls[0]) == {f"q{i}" for i in range(1, 10)},
          f"keys={sorted(calls[0]) if calls else None}")
    check("success closes the modal", state(at, "fm_open") is False,
          state(at, "fm_open"))
    check("the picks became the basket", state(at, "basket") == set(PICKS),
          f"basket={state(at, 'basket')}")
    check("the basket is labelled as the adviser's",
          state(at, "basket_source") == "formular", state(at, "basket_source"))
    check("the page advanced to the result stage", state(at, "stage") == "result",
          state(at, "stage"))

    text = tree_text(at)
    check("the caption says the adviser answered before seeing any result",
          "chosen by the questionnaire adviser" in text
          and "before seeing any result" in text)
    check("the proposal names a symbol it could not use",
          "BRK.A" in text and "not in this study" in text)
    check("the disclaimer travels with the proposal",
          "NOT A RECOMMENDATION TO TRADE" in text and "measurement, not a forecast" in text)
    check("the sealed result was computed for the basket",
          "metric-value" in text and "Executed path" in text)
    check("no key material anywhere in the rendered page",
          "sk-" not in text and "ANTHROPIC_API_KEY=" not in text)
    check("the page never raised", not at.exception, str(at.exception))

    print(f"\n{'FAILED: ' + ', '.join(FAILURES) if FAILURES else 'all checks passed'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
