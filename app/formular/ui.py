"""The button, the modal and the proposal block.

The only file in this folder that touches Streamlit or session state, and the only one the
simulator page knows about. Three state keys, all prefixed `fm_` so they cannot collide
with the page's own six.

Two mechanics decide the shape of this file:

* A dialog body is a FRAGMENT. Widget interaction inside it reruns only the dialog, and
  `st.rerun()` at app scope closes it. The caption and the result gate on the simulator
  page are computed ABOVE the point where render() is called, so the success path MUST end
  in an app-scope rerun — otherwise the page would draw a caption describing the previous
  basket. Every failure path instead falls through and renders inline, staying in the same
  fragment run, which keeps the dialog open and the nine answers intact without touching
  state at all.
* The modal is opened by a FLAG, not by `if st.button(): dialog()`. A flag survives an
  unrelated rerun of the page and can be set by a test.
"""
import streamlit as st

import data

from . import advisor, questions

K_OPEN, K_ANSWERS, K_RESULT = "fm_open", "fm_answers", "fm_result"

INTRO = (
    "A teaching device, not advice. Answer nine questions and a language model — standing "
    "where an investor stood on the first trading day of January 2024, with no knowledge "
    "of what came next — proposes a basket. It is shown your answers, the ticker symbols "
    "and one sector label per symbol. Nothing else. This console then reveals what the "
    "sealed models actually did with that basket. **Not a live trading signal and not "
    "investment advice.**"
)

BLINDFOLD = (
    "The adviser never sees a return, a ranking, a trade count, a drawdown or any figure "
    "from the out-of-sample window. That blindfold is the point of the exercise: picking "
    "a basket by looking at the results first would be look-ahead wearing a friendly "
    "interface."
)

LABELS = ("QUESTIONNAIRE-DERIVED SELECTION · THE ADVISER SAW NO OUT-OF-SAMPLE RESULT · "
          "NOT A RECOMMENDATION TO TRADE")

VERDICT = (
    "These picks come from a language model reading your nine answers, the ticker list and "
    "a ticker-to-sector map. They are a plausible-sounding story, not a research output: "
    "the model had no access to this study's returns, rankings, trade counts or "
    "benchmarks. What the console computes below is what the sealed models **did** with "
    "this basket over a window that has already happened — a measurement, not a forecast, "
    "and the adviser never saw it."
)

SECTOR_NOTE = (
    "Sector labels come from this add-on's own file, not from the study: the sealed store "
    "carries no sector column at all. They are knowledge from outside the research, added "
    "so the two sector questions can mean something."
)


def _open():
    st.session_state[K_OPEN] = True


def _dismiss():
    # Without this the flag would survive the X button and the modal would spring back
    # open on the next rerun of the page.
    st.session_state[K_OPEN] = False


def _seed(q):
    """The previous answer, so reopening the form does not wipe it.

    Widget-keyed state is dropped while the dialog is closed (the widgets stop being
    rendered), so the saved answers are re-offered as defaults. Never written through the
    Session State API — only handed to the widget as a default, which keeps Streamlit from
    warning about a key that is set both ways.
    """
    saved = st.session_state.get(K_ANSWERS) or {}
    value = saved.get(q.id)
    if q.multi:
        return [v for v in (value or []) if v in q.options]
    return q.options.index(value) if value in q.options else None


@st.dialog("Formular — nine questions, answered blind", width="large", on_dismiss=_dismiss)
def _modal(universe):
    st.caption(INTRO)
    ready, reason = advisor.available()
    if not ready:
        st.info(reason, icon="ℹ️")

    with st.form("fm_form", border=False):
        answers = {}
        for q in questions.QUESTIONS:
            if q.multi:
                answers[q.id] = st.multiselect(q.prompt, q.options, default=_seed(q),
                                               key=f"fm_{q.id}", placeholder="none")
            else:
                # index=None: nothing is preselected, so "you must answer" is a real gate
                # rather than a default the user never looked at.
                answers[q.id] = st.radio(q.prompt, q.options, index=_seed(q),
                                         horizontal=q.horizontal, key=f"fm_{q.id}")
        st.caption(BLINDFOLD)
        submitted = st.form_submit_button("Ask the adviser", type="primary",
                                          key="fm_submit", disabled=not ready)

    if not submitted:
        return

    missing = questions.answered(answers)
    clash = sorted(set(answers["q6"]) & set(answers["q7"]))
    if missing or clash:
        # Rendered inline, in the same fragment run: no rerun, so the dialog stays open
        # and every answer above is still on screen.
        if missing:
            st.error(f"{len(missing)} question(s) still unanswered: "
                     + "; ".join(missing), icon="⚠️")
        if clash:
            st.error("These sectors are on both lists, so the instruction contradicts "
                     f"itself: {', '.join(clash)}.", icon="⚠️")
        return

    with st.spinner("Asking the adviser — one call. Your answers go out; no result comes "
                    "from this store."):
        try:
            result = advisor.call_adviser(answers, universe)
        except advisor.AdviserError as exc:
            st.error(exc.calm, icon="⚠️")
            return

    st.session_state.basket = set(result["tickers"])
    st.session_state.basket_source = "formular"
    st.session_state.basket_edited = False
    st.session_state.stage = "result"
    st.session_state[K_ANSWERS] = answers
    st.session_state[K_RESULT] = result
    st.session_state[K_OPEN] = False
    st.rerun()          # app scope: closes the dialog AND redraws the page around it


def _proposal(result):
    st.markdown("#### What the adviser proposed — before seeing any result")
    st.markdown(f'<div class="disclaimer-box">{LABELS}</div>', unsafe_allow_html=True)
    if result.get("profile_summary"):
        st.caption(f"It read your profile as: {result['profile_summary']}")

    for pick in result["picks"]:
        st.markdown(f"**{pick['ticker']}** · {pick['sector']} — {pick['reason']}")

    if result.get("note"):
        st.markdown(result["note"])

    if result["dropped"]:
        st.caption(f"It also named {', '.join(result['dropped'])} — not in this study's "
                   f"universe, so they were not added.")
    if result["trimmed"]:
        st.caption(f"It returned more names than the profile allows; "
                   f"{', '.join(result['trimmed'])} were left out to honour your answer "
                   f"about how many positions to hold.")
    if result["mismatched"]:
        pairs = ", ".join(f"{t} (said {said}, map says {ours})"
                          for t, said, ours in result["mismatched"])
        st.caption(f"Sector disagreements between the adviser and the map: {pairs}.")

    st.caption(VERDICT)
    st.caption(SECTOR_NOTE)


def render(slot):
    """Draw the button into `slot`; the modal and the proposal go on the page itself."""
    universe = data.tickers()
    slot.button("Formular", key="fm_btn", width="stretch",
                help="Nine questions; an adviser that has seen no result picks the basket",
                on_click=_open)

    if st.session_state.get(K_OPEN):
        _modal(universe)

    if st.session_state.get(K_RESULT):
        _proposal(st.session_state[K_RESULT])
