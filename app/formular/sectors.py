"""Ticker to sector, from a map that belongs to this folder and not to the study.

The sealed store has no sector data of any kind — no column in any of its tables, no key
in any artifact. This map is knowledge from outside the research, typed into sectors.json
so the two sector questions can mean something. It is labelled as such on screen, and it
leaves with the folder.

Reading it grouped by sector rather than ticker-by-ticker is deliberate: that is the shape
the prompt sends (one line per sector beats 498 lines per ticker), and a grouped file is
the one a human can actually check by eye.
"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parent / "sectors.json"

_doc = json.loads(PATH.read_text(encoding="utf-8"))
BY_SECTOR = {s: tuple(ts) for s, ts in _doc["sectors"].items()}
PROVENANCE = _doc["_provenance"]

# The eleven GICS sectors, in the file's own order. Used as the response schema's enum, so
# the model cannot invent a twelfth.
GICS_11 = tuple(BY_SECTOR)

SECTOR_OF = {t: s for s, ts in BY_SECTOR.items() for t in ts}


def sector_of(ticker, default="—"):
    return SECTOR_OF.get(ticker, default)


def coverage(universe):
    """(mapped, unmapped, extra) against the store's ticker list.

    Fed by selfcheck.py, which requires unmapped and extra to be empty: a ticker with no
    sector would be silently invisible to a sector-filtered adviser, and a mapped ticker
    that is not in the store would be offered and then rejected on the way back.
    """
    uni, mapped = set(universe), set(SECTOR_OF)
    return sorted(uni & mapped), sorted(uni - mapped), sorted(mapped - uni)


def grouped(universe):
    """The universe as {sector: [tickers]}, sectors with no ticker dropped."""
    keep = set(universe)
    out = {}
    for sector, tickers in BY_SECTOR.items():
        present = [t for t in tickers if t in keep]
        if present:
            out[sector] = present
    return out
