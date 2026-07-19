"""Formular — an optional questionnaire that asks a language model to pick a basket.

An ADD-ON, built to be removed. Everything it needs lives in this folder plus one
`.env.example` at the repository root; the rest of its footprint is four lines of seam in
app/pages/simulator.py, one line in requirements.txt, one sentence in README.md and the
secret-ignoring rules in .gitignore. See README.md in this folder for the removal recipe.

The one thing it must never become is a way to choose a basket by looking at the answers
first. The adviser is placed at the start of the out-of-sample window and shown only the
questionnaire, the ticker symbols and a sector label — never a return, a ranking or a
trade count. prompt.py states that, and selfcheck.py proves it mechanically.

Public surface: render(slot). Nothing else is imported from outside this folder.
"""
from .ui import render

__all__ = ["render"]
