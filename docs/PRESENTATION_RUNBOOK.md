# Presentation Runbook — XGB ∥ LSTM methodology consoles

For whoever presents this, knowing the project or not. Two separate read-only Streamlit apps, native
Plotly, every number read from a frozen artifact — **nothing trains at runtime, nothing can be broken
by clicking.** Lines marked **say:** can be read aloud.

**Time:** ~12 min (XGB 8, LSTM 4). Short version — §7.

---

## §0 The one idea

> We do not claim "the best indicator". We ask, honestly and reproducibly: **from raw OHLCV, what is
> the smallest set of features that survives being chosen and judged by data that played no part in
> choosing it — and which OHLCV mechanism families carry across assets rather than being one asset's
> quirk?** The proof standard is frozen; the loop can never loosen its own bar; the OOS window is read
> **zero** times before certification.

Two model classes, same discipline, two apps: **XGB** (`make on`, :8503) and **LSTM** (`make on-lstm`, :8502).

---

## §1 Before you start (3 min)

```bash
cd /opt/to_liora_school/liora-project-ml-engineering
make family-transfer            # ensure family_transfer.json exists (Rung 8 artifact)
make family-transfer-selftest   # should print 33/33 checks passed
make on                         # XGB console → http://localhost:8503
make on-lstm                    # LSTM console → http://localhost:8502  (second terminal)
```

Check: XGB opens on **three pages** (Data Journey · Smart Methodology · Family Transfer); LSTM on
**five** (Methodology · Universal Backbone · Asset Overrides · Operating Point · OOS Results). Stop
after: `make off` and `make off-lstm`. Status any time: `make status`, `make status-lstm`.

---

## §2 XGB — Data Journey (1 min)

**say:** "This branch is the *method*, not a product. The first page shows where features come from
and the wall the data never crosses. Raw hourly bars, a frozen 17-feature core that is never searched,
45 searchable candidates grouped into 12 mechanism families, and a hard Train/OOS boundary."

Point at the facts row: **oos reads = 0**. **say:** "Everything is decided on Train; the out-of-sample
window is scored once, later, and never used to choose anything. That is the whole integrity claim."

---

## §3 XGB — Smart Methodology (2 min)

Point at the funnel `26 → 11 → 9 → 2 → 1`.

**say:** "Five numbers. 26 candidate arms are accepted in cross-fit; 11 beat the procedure-level
max-null; 9 survive all three nulls — marginal, regime, conditional; 2 are kept after the model tunes
around them; and both resolve to **one** feature. These are a snapshot of one run, not a target — a
fresh panel could give 30 → 7 → 2 → 0 and be just as correct."

Scroll to the **live guard** at the bottom.

**say:** "This is the only thing that runs live. It tries to weaken the null from 50 permutations to 5
— and the guard rejects it, in Python, on every visit. The loop cannot loosen its own proof standard."

---

## §4 XGB — Family Transfer · Rung 8 (3 min) *(the new work)*

**say:** "Rungs 1–6 answer per asset. Rung 8 asks the panel question: **which families travel across
assets?** It is a pure read of the same frozen artifacts — it never trains, never touches the bar
store, never reads OOS."

Point at the transfer funnel + the status bar.

**say:** "Each of the 12 families gets exactly one status, by the furthest stage its arms reached:
cross-fit-unstable, search-inflated, regime-dependent, conditional-failure, tuning-dependent,
asset-conditional, or panel-stable."

Point at the fact line under the funnel.

**say:** "And the honest headline: on this snapshot **no family is panel-stable**. The one retained
feature transfers on a single asset. That is the methodology's *asset-specificity does not travel*,
made mechanical — not a disappointment, a finding."

Show the **minimal panel family set** and the **taxonomy** table.

**say:** "The minimal set is a set-cover over confirmed asset-folds, not a ranking — and an empty set
would be a valid result. The taxonomy panel flags whether the 12-family split is right — a suggestion
only; changing it is a new search epoch, never automatic."

Toggle the panel radio to **Study panel (6 assets)**.

**say:** "The study scope is six assets, chosen for different behaviours. This is a development
experiment, not a certification — and where the snapshot lacks data, it says `INSUFFICIENT_EVIDENCE`
rather than inventing a number."

---

## §5 LSTM — Methodology + Backbone (2 min) — switch to :8502

**say:** "Same discipline, different model class. Per asset, an LSTM predicts whether a Triple-Barrier
trade wins, from a window of causal daily features. Feature search is Train-only, purged walk-forward
CV, behind an overfit gate."

Go to **Universal Backbone**. Point at the ablation ranking.

**say:** "One LSTM is trained across all assets on a shared feature set. This bar chart is honest
importance: how much validation AUC-PR falls when each feature is removed — measured on Train, never
on OOS."

---

## §6 LSTM — Overrides · Operating Point · OOS (2 min)

- **Asset Overrides:** "On top of the shared backbone, each asset selects its own extra features by the
  same Train-only search. A bigger set is not better — it is more search."
- **Operating Point:** "The entry threshold θ, the trade direction and Kelly fraction are fit on Train
  out-of-fold log-growth — never on OOS. A trade floor requires at least two trades to count as a
  strategy."
- **OOS Results:** "And the payoff: the 2024→2026 window, scored **exactly once per asset**. These
  distributions are out-of-sample — honest unseen-data performance, not a backtest tuned on the test."

**Closing:** "Two model classes, one frozen proof standard, zero OOS reads before certification. The
new piece — Rung 8 — turns *which mechanisms actually transfer between assets* from an opinion into a
deterministic, testable artifact."

---

## §7 Short version (3 min)

XGB **Smart Methodology** (funnel + live guard) → XGB **Family Transfer** (status bar + "no family is
panel-stable" + minimal set) → LSTM **OOS Results** (one-shot distributions). Closing line as above.

---

## §8 If something breaks

| symptom | fix |
|---|---|
| Family Transfer page says "not found" | `make family-transfer` (and `make family-transfer-panel6` for the 6-asset view) |
| a page shows a snapshot/missing warning | `make family-transfer`; the frozen snapshot must be present |
| port busy | `make on PORT=8601` / `make on-lstm LSTM_PORT=8602` |
| "is this certification?" | **No.** Say: development experiment on a known panel; certification is Rung 9, not started |
| "what are the results / how much does it earn?" | The XGB branch produces **no** trading claim (it is the method); LSTM OOS numbers are honest one-shot, not advice |

---

## §9 Questions that will come

- **Why no single best feature?** Because the honest answer on this panel is that almost nothing
  transfers — one feature, on one asset. Reporting a "best indicator" would be exactly the search
  inflation the nulls exist to catch.
- **Is 6 assets enough?** No — it is a development panel to exercise the report. Rung 9 (fresh panel)
  is the transfer test, and it is deliberately not done here.
- **Is this AI-generated?** The aggregator is a pure function of frozen artifacts with a self-test
  suite (registry, determinism, fail-closed, dedup, leakage, snapshot parity). Nothing is accepted
  that cannot be recomputed byte-for-byte.
- **Data leakage / look-ahead?** Features read closed bars only; every choice is on Train out-of-fold;
  OOS is read once. The Family Transfer aggregator reads **zero** OOS and imports no training code.

---

*This runbook describes the presentation; it is not its source. Every screen renders from the frozen
snapshot and the family_transfer artifact — if those change, the apps change with them.*
