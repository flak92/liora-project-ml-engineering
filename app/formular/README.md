# Formular — an adviser that picks before it is allowed to look

An **optional add-on** to the Basket Simulator, built from the first line to be removed
again. A button opens a modal with nine questions about the investor's profile; the
answers become one call to a language model; the symbols it returns are ticked into the
basket and computed by the page's existing pipeline.

## The blindfold, which is the whole point

The adviser is told it is standing on **the first trading day of January 2024** — the
start of the out-of-sample window — and it is shown exactly three things:

1. the nine questionnaire answers,
2. the ticker symbols it may choose from,
3. one sector label per symbol.

It never sees a return, a ranking, a trade count, a benchmark or a drawdown, and it is
told in as many words that it does not know what happened next. Only after it has
committed does the console reveal what the sealed models *did* with that basket.

This is not decoration. Ranking the assets and offering "the ten best" would be selection
on the test set arriving through the user interface instead of through the data — the
exact failure the pipeline blueprint warns about. An adviser that has to speak first, and
is then measured, demonstrates the problem rather than committing it.

`selfcheck.py` proves the blindfold mechanically: it subtracts the fixed template from a
built message and fails if a single token remains that is not a ticker, a sector name or a
questionnaire option, and it fails if any result-bearing column name appears at all. Three
negative controls confirm the gate fires when violated.

## Where the sector labels come from

`sectors.json`, typed by hand into this folder. They are **not** a research output: the
sealed store carries no sector, industry or GICS column in any table, and no artifact
contains one. They are knowledge from outside the study, present only so the two sector
questions can mean something, and they are labelled as such on screen. They leave with
this folder. A few assignments at the sector boundary are genuinely arguable; nothing
downstream depends on them being authoritative.

To swap in a different classification, replace `sectors.json` alone — every consumer goes
through `sectors.py`, and `selfcheck.py` will tell you at once if the new file does not
cover the universe exactly.

## The key

Copy `.env.example` (in the repository root) to `.env` and fill in `ANTHROPIC_API_KEY`.
A real environment variable wins over the file, so `ANTHROPIC_API_KEY=… make on` also
works. `.env` is git-ignored at any depth, including next to this file.

Without a key the button still appears and the modal still opens — it explains what is
missing and disables the submit. Without the `anthropic` package, same. Every other page,
and `make verify`, never need either.

**The app never displays, logs or stores the key.** No error message is built from an
exception's text, because SDK errors can carry request headers.

## Checks

```sh
.venv/bin/python3 app/formular/selfcheck.py        # offline: map, schema, blindfold, parsing
.venv/bin/python3 app/formular/selfcheck_page.py   # AppTest, adviser stubbed, no network
```

Neither needs a key or a network. `selfcheck_page.py` insists on one AppTest per process
(a second one in the same interpreter segfaults inside pyarrow).

## Removing it after the presentation

```sh
git revert -n <commit-sha>           # drops app/formular/, .env.example and the seam
git checkout HEAD -- .gitignore      # but KEEP the secret-ignoring rules
rm -f .env app/formular/.env         # untracked, so git will not do it for you
git commit -m "revert(formular): the questionnaire adviser comes out after the presentation"
```

By hand, if the commit has been rebased away: `git rm -r app/formular .env.example`, then
revert three hunks in `app/pages/simulator.py` (the soft import, the `formular` caption
branch, the `b3` column and its two lines), one line in `requirements.txt`, one sentence
in `README.md`. Leave the `.gitignore` rules — they are right whether or not this feature
exists.

Deleting the folder alone is also safe: the seam imports it inside a `try`, so the button
simply stops appearing and the page is exactly what it was before.

**Then rotate the key**, whether or not you believe it ever leaked. Removing a commit does
not remove it from clones that already fetched it.
