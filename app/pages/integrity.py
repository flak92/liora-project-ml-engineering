"""Integrity — the research contract, shown rather than claimed.

Every number here is read from the sealed store; nothing on this page is typed in. It
answers, in order: which epoch is this, what produced it, under which frozen parameters,
was the OOS read discipline kept,
does the interpretation layer describe THESE artifacts, when does the model stay idle,
and what the study does not cover.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import components as C
import data

C.page_header("Integrity", "The dataset's own record: identity, frozen parameters, OOS read "
                           "discipline, coverage and the limits — straight from the sealed store.")
C.guard(stop=False)

health = data.health()
run = data.research_run()
checks = data.integrity()
n_pass = sum(1 for c in checks if c.get("status") == "PASS")

# ---------------------------------------------------------------- 1. status
st.subheader("Dataset status")
c1, c2, c3 = st.columns(3)
c1.metric("Health", health["status"])
c2.metric("Integrity checks", f"{n_pass}/{len(checks)} PASS")
c3.metric("Research status", "FROZEN" if run.get("research_status", "").startswith("FROZEN")
          else run.get("research_status", "—"))
st.caption("Colour never carries the status on its own — the text above is the status. "
           "A single failing check turns the whole dataset unusable by design (fail-closed).")

# ---------------------------------------------------------------- 2. identity
st.subheader("Identity and provenance")
st.dataframe(pd.DataFrame([
    {"field": "epoch", "value": run.get("epoch", "—")},
    {"field": "release label", "value": run.get("presentation_freeze", "—")},
    {"field": "built at", "value": run.get("created_at", "—")},
    {"field": "XGB recipe hash", "value": run.get("xgb_recipe_hash", "—")},
    {"field": "LSTM recipe hash", "value": run.get("lstm_recipe_hash", "—")},
    {"field": "XGB Train / OOS", "value": f"{run.get('xgb_train','—')}  |  {run.get('xgb_oos','—')}"},
    {"field": "LSTM Train / OOS", "value": f"{run.get('lstm_train','—')}  |  {run.get('lstm_oos','—')}"},
]), hide_index=True, width="stretch")

reads = data.oos_reads()
if reads:
    reason = next((r.get("reason") for r in reads if r.get("reason")), "")
    if reason:
        st.caption(f"Epoch opened because: {reason}")
st.caption(
    "The epoch and the producing commit are deliberately anonymized in this public "
    "release; the recipe hashes above are the identity that survives, and they are the "
    "ones the research audits quote. The LSTM OOS window is declared on calendar "
    "boundaries (from 2024-01-01), so each sealed LSTM row reports its first realized "
    "session, 2024-01-02 — 1 January is a market holiday. XGB declares its splits on "
    "session dates already, so both surfaces read the same there.")

# ---------------------------------------------------------------- 2b. frozen parameters
st.subheader("Frozen parameters")
_xgb, _lstm = data.frozen_parameters()


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, (tuple, list)):
        return " + ".join(f"{x} bp" for x in v)
    return str(v)


st.dataframe(pd.DataFrame([
    {"parameter": label, "XGB (1h)": _fmt(_xgb.get(key)), "LSTM (1d)": _fmt(_lstm.get(key))}
    for label, key in [
        ("label horizon H", "H"),
        ("purge", "purge"),
        ("embargo", "embargo"),
        ("sequence length", "seq_len"),
        ("warmup window", "warmup"),
        ("Train window", "train"),
        ("OOS window (declared)", "oos"),
        ("take-profit (xATR)", "tp_atr"),
        ("stop-loss (xATR)", "sl_atr"),
        ("barrier evaluated on", "barrier_mode"),
        ("costs per side", "costs_bps"),
        ("entry fill", "entry_fill"),
        ("exit fill", "exit_fill"),
        ("time-exit fill", "scheduled_exit_fill"),
        ("capital mode", "capital_mode"),
        ("theta grid", "theta_grid"),
        ("Train-OOF trade floor", "min_oof_trades"),
        ("random seed", "seed"),
    ]]), hide_index=True, width="stretch")
st.caption(
    "Read from config/xgb.json and config/lstm.json — the same files the pipelines read, "
    "not prose. The fill conventions are declared once, in the XGB config; the LSTM "
    "engine is a port of the same trade mechanics (METHODOLOGY §3). Model "
    "initialization: XGBoost trains per asset from scratch after per-asset Optuna; the "
    "LSTM warm-starts every asset from one committed universal backbone — the "
    "per-asset cold-start study exists behind LSTM_COLD_START=1 and did not run in this "
    "epoch.")

# ---------------------------------------------------------------- 3. OOS read ledger
st.subheader("OOS read discipline")
st.markdown("The out-of-sample window is **never used to choose anything** — it only "
            "reports. Every read is counted in an append-only ledger (an interrupted, "
            "resumed pass counts every read), so the discipline is auditable rather than "
            "asserted. The table below is that ledger **summarised per pipeline**; the "
            "cumulative counter spans all epochs the asset has ever been sealed in, and "
            "its per-asset rows live in the research branch's ledger files.")
if reads:
    st.dataframe(pd.DataFrame([{
        "pipeline": r["pipe"],
        "assets sealed": r["tickers"],
        "reads this epoch": r["reads_this_epoch"],
        "cumulative read_count (min / mean / max)":
            f"{r['cum_read_min']} / {r['cum_read_mean']:.1f} / {r['cum_read_max']}"
            if r["cum_read_mean"] is not None else "—",
        "epoch closed": "yes" if r["closed"] else "no",
    } for r in reads]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- 4. coverage
st.subheader("What the interpretation layer covers")
cov = data.model_hash_coverage()
if cov:
    st.dataframe(pd.DataFrame([{
        "model": r["model"], "assets described": r["tickers"],
        "distinct sealed models": r["models"],
        "interpretation recipes": r["recipes"],
    } for r in cov]), hide_index=True, width="stretch")
    st.caption("One interpretation recipe across the whole epoch means every asset was "
               "described by the same frozen method; the model count matching the asset "
               "count means the layer describes THESE artifacts, not a stale set.")

# ---------------------------------------------------------------- 5. idle
st.subheader("When the model stays idle")
mm = data.result_mode_matrix()
if mm["models"]:
    rows = []
    for mode in mm["modes"]:
        row = {"result_mode": mode}
        for model in mm["models"]:
            row[model] = mm["counts"].get((model, mode), 0)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("`TRAIN_OOF_FLOOR_NOT_MET` is the honest not-promoted state: no operating "
                "point on the Train-OOF grid cleared the trade floor, so the asset was **not "
                "promoted to a strategy** (such an asset usually never trades OOS, but it "
                "can trade and simply not be promoted). `HODL_FALLBACK_NO_MODEL_TRADES` means the model never "
                "traded out-of-sample, so the executed path is buy-and-hold — its single "
                "trade is the benchmark's, never counted as a model trade.")

# ---------------------------------------------------------------- 6. checks
st.subheader("Integrity checks")
if checks:
    st.dataframe(pd.DataFrame([{"check": c["check_name"], "status": c["status"],
                                "detail": c["message"]} for c in checks]),
                 hide_index=True, width="stretch")

# ---------------------------------------------------------------- 7. limits
st.subheader("Known limits")
st.markdown(
    "- **Survivorship** — the universe is today's S&P 500 constituents applied backward.\n"
    "- **Single feed** — one equities source; no cross-vendor reconciliation of bars.\n"
    "- **Corporate actions** — prices are split-adjusted from a reviewed event table; "
    "spin-offs and special dividends are deliberately NOT adjusted (they are real price "
    "drops for a price benchmark), and splits below 3:2 are not detectable from bars.\n"
    "- **Bull OOS window** — buy-and-hold is a hard benchmark over this period.\n"
    "- **Barrier timing** — barriers are evaluated on closes, which is conservative on "
    "win-rate: the tighter stop is pierced intra-bar more often than the wider target.\n"
    "- **In-sample interpretation** — the ENTRY ranges and contributions are "
    "**Train-derived descriptions of a sealed model**, not an out-of-sample result and not "
    "a live trading signal.")
st.caption("Full written audits (Polish) live in docs-facts-infos/ at the repository root: "
           "the OHLCV data audit, the methodological-integrity audit (its §6 is the list "
           "of MINOR findings), and the research-consistency report. Artifact integrity "
           "is not only asserted here — `make verify` recomputes every hash offline.")

C.integrity_footer()
