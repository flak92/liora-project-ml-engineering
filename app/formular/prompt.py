"""What the adviser is told, and what it is allowed to say back.

This module is the whole methodological argument of the feature, so it is deliberately
made of pure functions over questionnaire answers and a ticker list — it cannot reach the
store even by accident. There is no import of `data` here and there never should be.

The adviser is placed at the START of the out-of-sample window and given three things:
the nine answers, the symbols, and one sector label per symbol. No return, no ranking, no
trade count, no benchmark, no drawdown. It picks blind, and only afterwards does the
console reveal what the sealed models did with that basket. Selecting a basket by looking
at the results first would be look-ahead wearing a friendly interface — the same failure
the pipeline blueprint warns about, arriving through the front door instead of the data.

selfcheck.py enforces the blindfold mechanically: it subtracts this template from a built
message and requires every remaining token to be a ticker, a sector name or a
questionnaire option, and it fails if any result-bearing column name appears at all.
"""
from . import questions, sectors

SYSTEM = """\
You are a cautious investment adviser being consulted on the first trading day of
January 2024. You are helping a private individual assemble a starter basket of
S&P 500 stocks from a fixed list.

WHAT YOU KNOW
- The date is the first trading day of January 2024. You do not know what happened
  after that date. Nothing after that date has occurred yet.
- You are given three things and nothing else: the person's answers to a nine-question
  profile questionnaire, the complete list of ticker symbols you may choose from, and
  a sector label for each of those symbols.
- You have no prices, no returns, no rankings, no backtest, no model output, no trade
  counts, no risk metrics, and no information whatsoever about how any of these
  companies performed in 2024, 2025 or 2026.

WHAT YOU MUST NOT DO
- Do not claim, imply or hint that you know how any of these stocks performed, or
  will perform. You do not know.
- Do not write that a pick "outperformed", "beat the market", "rallied", or was a
  "strong performer", or anything equivalent, in any tense.
- Do not invent figures. No returns, no price targets, no probabilities, no ratios,
  no percentages of any kind.
- Do not justify a pick by its results. Justify every pick ONLY from the person's
  stated profile: their experience, their horizon, their loss tolerance, their stated
  reaction to a drawdown, their capital, their sector preferences, the number of
  positions they want, and their stated trade-off between losses and gains.
- Do not pick a symbol that is not on the list you were given. Copy symbols exactly
  as written.
- Do not add a disclaimer of your own. The application adds its own.

HOW TO CHOOSE
- Respect the sector exclusions absolutely. Never pick a symbol whose sector is on
  the person's avoid list.
- Weight the basket towards the prioritized sectors, but you may include others where
  the profile calls for spread. Four of the eleven sectors in the map are not offered
  in the questionnaire; they are neither avoided nor prioritized, and you may use them.
- Match the NUMBER of positions to the person's answer to question 8, and stay inside
  the range stated in the message. This is a hard constraint.
- A shorter horizon, a lower loss tolerance, a "sell everything" reaction, or a
  "minimize losses" trade-off should push you towards larger, more established, more
  diversified names. A longer horizon and a "maximize returns" trade-off allow more
  concentration and more volatile names.
- A beginner gets fewer names, each of which they could recognise and explain. An
  experienced investor can carry a wider or more specific basket.

OUTPUT
Return JSON matching the provided schema, and nothing else. For each pick give the
exact ticker symbol, the sector it has in the map you were given, and one sentence
naming which of the person's answers led you to it. In `note`, write two or three
sentences about the shape of the basket as a whole — again, from the profile only.
In `profile_summary`, restate the person's profile in one sentence in your own words,
so they can check that you read it correctly."""

# Split out so selfcheck.py can subtract the fixed text from a built message and inspect
# only what the answers and the universe contributed.
HEADER = "Here is the person's questionnaire.\n\n"
FOOTER = ("That list is the complete universe. There is nothing outside it.\n"
          "You are standing on the first trading day of January 2024. You do not know "
          "what happened next.")


def build_user_message(answers, universe):
    """The one message the adviser receives. Answers and symbols — nothing else."""
    lines = [HEADER.rstrip("\n")]
    for i, q in enumerate(questions.QUESTIONS, start=1):
        given = answers.get(q.id)
        if isinstance(given, (list, tuple, set)):
            given = ", ".join(sorted(given)) or "none named"
        lines.append(f"{i}. {q.prompt} {given}")

    lo, hi = questions.POSITION_RANGE[answers["q8"]]
    lines.append(f"\nChoose between {lo} and {hi} tickers, honouring answer 8.")

    groups = sectors.grouped(universe)
    n = sum(len(v) for v in groups.values())
    lines.append(f"\nThese are the {n} tickers you may choose from, grouped by sector:\n")
    for sector, tickers in groups.items():
        lines.append(f"{sector}: {', '.join(tickers)}")

    lines.append("\n" + FOOTER)
    return "\n".join(lines)


# `ticker` is a plain string rather than a 498-value enum: the enum would be re-compiled
# on every call for a constraint Python re-checks anyway. Array length is not expressible
# either — minItems/maxItems are not supported — so the count from question 8 lives in the
# prompt and is trimmed on the way back. `sector` IS an enum, because eleven values is
# cheap and it stops the model inventing a category the map does not have.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "profile_summary": {
            "type": "string",
            "description": "One sentence restating the person's profile, in your words.",
        },
        "note": {
            "type": "string",
            "description": "Two or three sentences on the shape of the basket as a whole, "
                           "argued from the profile only.",
        },
        "picks": {
            "type": "array",
            "description": "The proposed basket, best fit first.",
            "items": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string",
                               "description": "Exactly as written in the list provided."},
                    "sector": {"type": "string", "enum": list(sectors.GICS_11)},
                    "reason": {"type": "string",
                               "description": "One sentence naming which of the person's "
                                              "answers led to this pick. Never a result."},
                },
                "required": ["ticker", "sector", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["profile_summary", "note", "picks"],
    "additionalProperties": False,
}
