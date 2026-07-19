"""The nine questions, as data.

Kept apart from the widgets so the prompt builder, the validator and the offline checks
all read the same list. Nothing here imports Streamlit, so questions.py can be exercised
without a browser and without the app running.

The wording is the questionnaire as commissioned, not a paraphrase: these strings are what
the adviser is shown, so editing one here changes what the model reads.
"""
from collections import namedtuple

# horizontal=False when the options are long enough that a row of radio buttons wraps
# badly inside a dialog; required=True is every question — a half-answered profile would
# be filled in by the model's imagination, which is exactly what this page must not do.
Q = namedtuple("Q", "id prompt options multi horizontal")

# The seven sectors the questionnaire offers. The map in sectors.json carries eleven; the
# other four (Materials, Utilities, Real Estate, Consumer Staples) are neither avoided nor
# prioritized and the adviser is told it may still use them.
SECTORS_OFFERED = (
    "Information Technology",
    "Financials",
    "Health Care",
    "Energy",
    "Consumer Discretionary",
    "Communication Services",
    "Industrials",
)

QUESTIONS = (
    Q("q1", "Do you have prior experience investing in individual stocks?",
      ("Beginner", "Some experience", "Experienced"), False, True),
    Q("q2", "How long do you plan to hold your positions?",
      ("Less than 1 week", "1-4 weeks", "1-12 months", "1+ year"), False, True),
    Q("q3", "What is the largest drop in your capital (%) you would accept before selling "
            "everything?",
      ("5%", "10%", "20%", "30% or more"), False, True),
    Q("q4", "If your portfolio dropped 20% in a month, what would you do?",
      ("Sell everything", "Sell part of it", "Hold", "Buy more"), False, False),
    Q("q5", "How much capital do you want to allocate initially (USD)?",
      ("Under 1,000", "1,000-5,000", "5,000-20,000", "Over 20,000"), False, True),
    Q("q6", "Which sectors would you rather avoid?", SECTORS_OFFERED, True, False),
    Q("q7", "Which sectors would you like to prioritize?", SECTORS_OFFERED, True, False),
    Q("q8", "How many positions do you want to hold at the same time?",
      ("Few (1-3)", "Moderate (4-7)", "Fully diversified (8-10)"), False, True),
    Q("q9", "Which trade-off describes you best?",
      ("Minimize losses even if it means missing gains",
       "Maximize potential returns even with larger swings",
       "Balance steady moderate gains against moderate risk"), False, False),
)

# Question 8 is the only answer that becomes a hard numeric constraint. The prompt states
# the range and the response handler trims to the maximum, because the JSON schema cannot
# express it: minItems/maxItems are not supported by structured outputs.
POSITION_RANGE = {
    "Few (1-3)": (1, 3),
    "Moderate (4-7)": (4, 7),
    "Fully diversified (8-10)": (8, 10),
}

BY_ID = {q.id: q for q in QUESTIONS}


def answered(answers):
    """The prompts of every required question the user left blank."""
    return [q.prompt for q in QUESTIONS if not answers.get(q.id)]
