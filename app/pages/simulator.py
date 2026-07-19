"""Basket Simulator — pick assets, see what the sealed models did with them.

Nothing is simulated at runtime: every figure is read from sealed per-asset rows and
summed. There is no per-trade data in this release, so a basket is a sum of ENDPOINTS —
no equity curve, no drawdown path, no timing. The arithmetic and the two predicates live
in app/basket.py; the diagram lives in app/venn.py; this file is the interface.

Two ways to build a basket, and they never fight: a preset writes the basket, the tile
grid writes the basket, and nothing ever writes back into the preset widget. When the two
diverge the caption says so in words rather than silently resetting a control.
"""
import html
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

import basket as B
import components as C
import data
import theme
import venn

# Optional add-on, built to be deleted in one `git rm -r app/formular`. Caught broadly on
# purpose: a missing folder, a syntax error or a malformed data file inside it must all
# degrade to "the button is not there", never to a page that will not open.
try:
    import formular
except Exception:
    formular = None

GRID_COLS = 10

# Above this basket size the detail table falls back to st.dataframe (no tooltips).
# Measured on this store the manifests are small — 1.3 KB per ticker worst case, ~0.3 MB
# for the whole universe — so the cap is about DOM size and the scroll ergonomics of one
# st.markdown block, not payload weight.
MAX_TIP_ROWS = 120

# Ticker tiles, scoped by the container class Streamlit derives from the widget key
# (frontend: "st-key-" + key.replace(/[^a-zA-Z0-9_-]/g,'-'), verified on streamlit 1.58.0).
# Undocumented DOM coupling: if a future version changes that expression the tiles stop
# being square and nothing else breaks. AppTest cannot see CSS, so this is eyes-only.
GRID_CSS = f"""<style>
div[class*="st-key-tk_"] button {{
    aspect-ratio: 1 / 1;
    width: 100%;
    padding: 0;
    font-size: 0.8rem;
    border-radius: 6px;
}}
div[class*="st-key-tk_"] button[data-testid="stBaseButton-secondary"] {{
    background: {theme.SURFACE};
    color: {theme.TEXT};
    border: 1px solid {theme.BORDER};
}}
</style>"""

DETAIL_TABLE_COLS = [
    ("ticker", "ticker", "", None, None),
    ("strategy", "strategy", "", None, None),
    ("end_capital", "End capital (USD)", "num", None, None),
    ("return_pct", "Return %", "num", None, None),
    ("trades", "trades", "num", None, None),
    ("wins", "Won", "num", None, "Winning trades — feeds the overlap diagram"),
    ("losses", "Lost", "num", None, "wins + losses = trades"),
    ("max_drawdown_pct", "Max DD %", "num", None, None),
    ("win_rate_pct", "Win rate %", "num", None, None),
    ("profit_factor", "Profit factor", "num", None, None),
]


def _num2(v):
    # profit_factor is NULL where it is not rankable: rows with <2 model trades, plus a few
    # where gross loss is zero so the ratio is undefined (PF_ZERO_GROSS_LOSS_POLICY=not_rankable
    # — LSTM CAH/EL/MTB won every trade). Sealed counts: 170/498 XGB, 50/495 LSTM. Read back as
    # NaN; render an em-dash — an empty-looking cell beats the string "nan".
    return "—" if pd.isna(v) else f"{float(v):.2f}"


def _int(v):
    return "—" if pd.isna(v) else f"{int(v):,}"


_FMT = {"end_capital": _num2, "return_pct": _num2, "trades": _int, "wins": _int,
        "losses": _int, "max_drawdown_pct": _num2, "win_rate_pct": _num2,
        "profit_factor": _num2}


# ---------------------------------------------------------------- tooltip

def _tip_html(ticker, model):
    """The ⓘ body: which optional features the Train-only search selected for this asset.

    Reads asset_features, which holds the SELECTED features only — never the frozen core.
    218 of 498 XGB assets (NVDA among them) selected nothing at all, and saying so is the
    point: it is a finding about the search, not a gap in the data."""
    rows = data.features(ticker, model)
    row = data.asset(ticker, model)
    if row is None:
        return f'<div class="pad-tip-cap">No {model.upper()} run for this ticker.</div>'
    if not rows:
        return ('<div class="pad-tip-cap">No optional features selected — the frozen core '
                'manifest runs alone.</div>')
    declared = row.get("selected_feature_count")
    drift = "" if declared in (None, len(rows)) else (
        f' <span title="the store disagrees with itself">⚠ store says {declared}</span>')
    groups = {}
    for r in rows:                                    # already ordered by feature_id
        groups.setdefault(r.get("feature_family") or "other", []).append(r)
    body = []
    for family, feats in groups.items():
        body.append(f'<tr class="pad-group"><th colspan="2">'
                    f'{html.escape(str(family))} · {len(feats)}</th></tr>')
        for f in feats:
            formula = f.get("formula")
            extra = (f'<div class="pad-formula">{html.escape(str(formula))}</div>'
                     if formula else "")
            body.append(f'<tr><td class="pad-id">{html.escape(str(f.get("feature_id", "")))}</td>'
                        f'<td>{html.escape(str(f.get("feature_name", "")))}{extra}</td></tr>')
    return (f'<div class="pad-tip-cap">{len(rows)} feature(s) selected by the Train-only '
            f'per-asset search, on top of the frozen core manifest.{drift}</div>'
            f'<table class="pad-feat">{"".join(body)}</table>')


def _detail_css():
    """CSS for the detail table and its ⓘ tooltip. Two constraints worth keeping:
    - pure CSS (st.markdown strips <script>), so the tooltip opens on :hover AND
      :focus-within — the latter makes it reachable by keyboard Tab and by tap;
    - the table must NOT sit in an overflow-x wrapper: a scroll container clips the
      absolutely-positioned tooltip.
    Colors come from theme.py because this console ships one fixed dark theme."""
    ink, line = theme.TEXT, theme.BORDER
    return f"""<style>
.pad-table {{ width:100%; border-collapse:collapse; font-size:.9rem; margin:.25rem 0 .75rem; }}
.pad-table th, .pad-table td {{ padding:.4rem .6rem; border-bottom:1px solid {line};
    text-align:left; white-space:nowrap; }}
.pad-table thead th {{ background:{theme.SURFACE}; color:{ink}; font-weight:600; }}
.pad-table th.num, .pad-table td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.pad-table tbody tr:hover {{ background:rgba(230,237,243,.05); }}
.pad-info {{ position:relative; display:inline-flex; align-items:center; gap:.45rem; }}
.pad-i {{ display:inline-flex; align-items:center; justify-content:center;
    width:1.05rem; height:1.05rem; border:1px solid currentColor; border-radius:50%;
    font-size:.68rem; font-style:italic; font-weight:600; opacity:.55; cursor:help; }}
.pad-info:hover .pad-i, .pad-info:focus-within .pad-i {{ opacity:1; }}
.pad-tip {{ display:none; position:absolute; left:100%; top:-.4rem; z-index:1000; width:20rem;
    max-height:21rem; overflow-y:auto; background:{theme.SURFACE}; color:{ink};
    border:1px solid {line}; border-radius:.5rem; padding:.6rem .75rem;
    box-shadow:0 8px 24px rgba(0,0,0,.45); white-space:normal; }}
.pad-info:hover .pad-tip, .pad-info:focus-within .pad-tip {{ display:block; }}
.pad-tip-cap {{ font-size:.75rem; opacity:.8; margin-bottom:.4rem; }}
.pad-feat {{ width:100%; border-collapse:collapse; font-size:.78rem; }}
.pad-feat td {{ padding:.14rem .35rem; border-bottom:1px dashed {line}; white-space:normal; }}
.pad-feat .pad-group th {{ text-align:left; padding:.45rem .35rem .15rem; font-size:.7rem;
    text-transform:uppercase; letter-spacing:.05em; opacity:.65; white-space:normal; }}
.pad-id {{ opacity:.55; width:2.4rem; text-align:right; font-variant-numeric:tabular-nums; }}
.pad-formula {{ font-size:.68rem; opacity:.6; font-family:{theme.MONO}; word-break:break-all; }}
</style>"""


def _detail_table_html(det, model):
    head = "".join(
        f'<th class="{cls}" title="{html.escape(tip)}">{html.escape(hdr)}</th>' if tip
        else f'<th class="{cls}">{html.escape(hdr)}</th>'
        for _, hdr, cls, _, tip in DETAIL_TABLE_COLS)
    body = []
    for row in det.itertuples(index=False):
        cells = []
        for col, _, cls, _, _ in DETAIL_TABLE_COLS:
            v = getattr(row, col)
            if col == "ticker":
                tk = html.escape(str(v))
                cells.append(f'<td><span class="pad-info" tabindex="0" role="button" '
                             f'aria-label="Feature set for {tk}">{tk}'
                             '<span class="pad-i" aria-hidden="true">i</span>'
                             f'<span class="pad-tip">{_tip_html(str(v), model)}</span></span></td>')
            else:
                fmt = _FMT.get(col)
                cells.append(f'<td class="{cls}">{html.escape(fmt(v) if fmt else str(v))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return ('<table class="pad-table"><thead><tr>' + head + "</tr></thead><tbody>"
            + "".join(body) + "</tbody></table>")


# ---------------------------------------------------------------- state

def _select_method():
    """Commit the switch's choice to a plain key that survives pages where the widget is not
    drawn (Streamlit clears widget-keyed state once its widget stops being rendered). The
    switch is required=True, so None cannot arrive — but if it ever did, keep the old model
    rather than crashing every lookup downstream."""
    if st.session_state.method_sel:
        st.session_state.method = st.session_state.method_sel
        # A ranking preset means "top ten of the model you are looking at", so switching the
        # model has to re-derive it. Without this the basket keeps the previous model's
        # ranking while the caption still calls it a preset — the caption would be lying.
        source = st.session_state.get("basket_source")
        if data.PRESET_PER_MODEL.get(source):
            st.session_state.basket = set(
                data.preset_tickers(source, st.session_state.method))


def _apply_preset():
    """Preset -> basket. The only other writer is the grid; nothing writes back into the
    preset widget, so the two can never fight over a value.

    st.pills round-trips its value as the FORMATTED label and only converts back if that
    label is still in the table built during the current render. So accept either form,
    and ignore anything that is neither rather than raising in a callback."""
    raw = st.session_state.get("preset_sel")
    key = data.PRESET_KEY_BY_LABEL.get(raw, raw)
    if key not in data.PRESET_LABELS:
        return
    st.session_state.basket = set(data.preset_tickers(key, st.session_state.method))
    st.session_state.basket_source = key
    st.session_state.basket_edited = False


def _toggle(ticker):
    """A tile edits the basket without erasing where it came from: the caption can then say
    "preset X, edited by hand", which is the story this page advertises. Only a basket built
    from nothing is called hand-picked."""
    b = st.session_state.basket
    b.discard(ticker) if ticker in b else b.add(ticker)
    st.session_state.basket_edited = True
    if not st.session_state.get("basket_source"):
        st.session_state.basket_source = "manual"


RANDOM_MIN, RANDOM_MAX = 5, 15


def _random_basket(tickers):
    """A fresh draw every click — unseeded on purpose, unlike the seeded preset, so the
    demo can keep pulling new baskets instead of replaying one."""
    n = min(random.randint(RANDOM_MIN, RANDOM_MAX), len(tickers))
    st.session_state.basket = set(random.sample(list(tickers), n))
    st.session_state.basket_source = "random"
    st.session_state.basket_edited = False


def _clear_basket():
    st.session_state.basket.clear()
    st.session_state.basket_source = ""
    st.session_state.basket_edited = False


def _go(stage):
    st.session_state.stage = stage


# ---------------------------------------------------------------- page

C.page_header("Basket Simulator",
              "Pick a basket of S&P 500 assets and read what the sealed models did with it — "
              "against the same basket simply held.")
# stop=True like every other store-reading page: stop=False belongs to the three static
# pages, which render fine without the database. This one queries it immediately.
C.guard()

st.session_state.setdefault("stage", "pick")
st.session_state.setdefault("basket", set())
st.session_state.setdefault("basket_source", "")
st.session_state.setdefault("basket_edited", False)
st.session_state.setdefault("method", "XGBoost")

method = st.session_state.method
model = data.MODEL_KEY[method]
df = data.simulator_rows(model)
tickers = list(df["ticker"])

# The two pipelines sealed 498 and 495 assets, so three tickers exist for XGB only. Narrow
# them out of THIS model's view, but never out of the stored basket: deleting them would
# make a round trip through the switch quietly shrink the basket for good.
covered = st.session_state.basket & set(tickers)
dropped = st.session_state.basket - covered

st.sidebar.markdown(
    f"**The game**\n\n"
    f"- Every ticker you pick is a ${B.ENTRY_USD:,.0f} entry, held over the sealed "
    f"out-of-sample window.\n"
    f"- The model only decides ENTRY; take-profit and stop-loss are a mechanical ATR "
    f"contract.\n"
    f"- Buy & hold over the same window is the benchmark, and it is a hard one.")

col_m, col_n = st.columns([1, 2])
with col_m:
    # A switch, not a dropdown: the choice is strictly one of two, so it should look like
    # two positions rather than a text field with a caret in it. required=True means an
    # option can never be deselected, so `method` always names a real model.
    st.segmented_control("Model", list(data.MODEL_KEY), default=method, required=True,
                         key="method_sel", on_change=_select_method)
with col_n:
    st.markdown("**Basket preset** — every membership is derived from the sealed store")
    # Re-light the chip after a page switch: preset_sel is widget-keyed, so Streamlit drops
    # it when the widget stops rendering, and the caption below would then still name a
    # preset that no longer looks selected. basket_source is the surviving copy.
    if (st.session_state.basket_source in data.PRESET_LABELS
            and not st.session_state.get("preset_sel")):
        st.session_state.preset_sel = st.session_state.basket_source
    # The labels are CONSTANT: st.pills matches its value by formatted string, so a label
    # that changed with the model (".. (XGBoost)") stopped matching after a switch and the
    # raw label landed in session_state. Which model a ranking preset uses is said below.
    st.pills("preset", list(data.PRESET_LABELS),
             format_func=lambda k: data.PRESET_LABELS[k],
             key="preset_sel", on_change=_apply_preset, label_visibility="collapsed")
    st.caption(f"Top / bottom / busiest rank the **{method}** rows; the other presets are "
               "the same set whichever model is selected.")

if dropped:
    st.caption(f"{len(dropped)} of your picks have no {method} row and sit out this "
               f"calculation: {', '.join(sorted(dropped))}. They stay in the basket and "
               f"come back when the model does.")

n_sel = len(covered)
source = st.session_state.basket_source
edited = (", edited by hand" if st.session_state.basket_edited else "")
if source in data.PRESET_LABELS:
    st.caption(f"Basket: **{n_sel}** ticker(s) — preset “{data.PRESET_LABELS[source]}”"
               f"{edited}. ${B.ENTRY_USD * n_sel:,.0f} to invest.")
elif source == "random" and n_sel:
    st.caption(f"Basket: **{n_sel}** ticker(s), drawn at random{edited}. "
               f"${B.ENTRY_USD * n_sel:,.0f} to invest.")
elif source == "formular" and n_sel:
    st.caption(f"Basket: **{n_sel}** ticker(s), chosen by the questionnaire adviser"
               f"{edited} — it answered before seeing any result. "
               f"${B.ENTRY_USD * n_sel:,.0f} to invest.")
elif n_sel:
    st.caption(f"Basket: **{n_sel}** ticker(s), picked by hand. "
               f"${B.ENTRY_USD * n_sel:,.0f} to invest.")
else:
    st.caption("Basket is empty — choose a preset above, or pick from the grid below.")

st.button("Calculate basket", type="primary", disabled=(n_sel == 0),
          on_click=_go, args=("result",))

# The grid is always on screen: it is the page, not an option. Every tile is rebuilt on
# every rerun (~0.6 s for the whole universe), which is the price of one source of truth —
# a fragment kept the counter here in step while the caption and button above it lagged.
st.subheader(f"Pick by hand — the {len(tickers)}-tile grid")
b1, b2, b3, _ = st.columns([2, 2, 2, 4])
b1.button("Random", width="stretch",
          help=f"Draw a fresh basket of {RANDOM_MIN} to {RANDOM_MAX} tickers",
          on_click=_random_basket, args=(tickers,))
b2.button("Clear", width="stretch", on_click=_clear_basket)
if formular:
    formular.render(b3)

st.markdown(GRID_CSS, unsafe_allow_html=True)
picked = st.session_state.basket
st.caption(f"selected: **{len(picked)}** · to invest: ${B.ENTRY_USD * len(picked):,.0f}"
           + (f" · {len(dropped)} not scored under {method}" if dropped else ""))
with st.container(height=560):
    for start in range(0, len(tickers), GRID_COLS):
        for col, tk in zip(st.columns(GRID_COLS), tickers[start:start + GRID_COLS]):
            col.button(tk, key=f"tk_{tk.replace('.', '_')}",
                       type="primary" if tk in picked else "secondary",
                       on_click=_toggle, args=(tk,), width="stretch")

if st.session_state.stage != "result" or not n_sel:
    st.stop()

# ---------------------------------------------------------------- result

r = B.compute_basket(sorted(covered), df, data.hodl_returns(model))
window = str(df["oos_window"].iat[0]) if len(df) else "—"

st.divider()
st.subheader(f"Basket of {r['n']} — sealed out-of-sample window {window}")
if window != "—":
    # The header shows the first REALIZED session (from the sealed rows); the page footer
    # shows the DECLARED calendar window (from research_run). For LSTM those differ by one
    # day because 1 January is a market holiday — say so rather than let a reader find it.
    st.caption("Dates here are the first and last realized sessions of the frozen window; "
               "the footer below quotes the declared calendar boundaries, which for the "
               "daily model start one day earlier (1 January is a market holiday).")

# Three numbers, never one. The executed path is what happened to the capital; the model
# result is what floor-met configurations actually traded; the benchmark is what holding
# would have produced. A row whose model never traded carries the buy-and-hold path —
# folding it into a strategy figure would credit the model with the benchmark.
ml = (f"{r['ml_return_pct']:+.2f}%" if r["ml_return_pct"] is not None else "—")
hodl = (f"{r['hodl_return_pct']:+.2f}%" if r["hodl_return_pct"] is not None else "—")
C.metric_row([
    ("Executed path (all assets)", f"{r['return_pct']:+.2f}%   ({r['final']:,.0f} USD)"),
    (f"{method} model result ({r['ml_n']} of {r['n']})",
     f"{ml}   ({r['ml_final']:,.0f} USD)" if r["ml_return_pct"] is not None else "—"),
    ("Price-only buy & hold", f"{hodl}   ({r['hodl_final']:,.0f} USD)"),
])
# The three USD figures do NOT share a base — the middle one is a subset of the basket, so
# comparing the dollars side by side is meaningless while the percentages remain comparable.
st.caption(f"Bases differ: the executed path and buy & hold are the endpoint of "
           f"${r['invested']:,.0f} over {r['n']} assets, the model result of "
           f"${r['ml_invested']:,.0f} over {r['ml_n']}. Compare the percentages; the dollars "
           f"only within one column.")

if r["no_model_trade_n"]:
    st.caption(f"**{r['no_model_trade_n']} of {r['n']} assets carry no model result** — "
               "either the model never traded them, or its configuration never cleared the "
               "Train-OOF trade floor and the OOS run is a diagnostic replay. Where the model "
               "never traded, the capital path IS the benchmark's; a diagnostic replay has its "
               "own path and carries no model result. Both sit inside the executed path and "
               f"outside the {method} model-result figure.")
if r["hodl_return_pct"] is not None:
    if r["ml_n"] == 0:
        # Every asset fell back to the benchmark, so the two paths are the same path and a
        # verdict between them would be theatre — the gap is rounding, not skill.
        st.caption("No asset in this basket produced a model result, so the executed path IS "
                   "the benchmark path: the tiny difference between the two figures is "
                   "rounding, not performance. This is what a basket of assets the models "
                   "declined to trade looks like — an explicit verdict, not a gap.")
    else:
        verdict = "**beat**" if r["beats_hodl"] else "did **not** beat"
        st.caption(f"Over this window the executed path {verdict} price-only buy & hold (splits "
                   "adjusted, dividends excluded — the same price plane the strategy trades). "
                   "The OOS window is never optimized against, and every read of it is counted "
                   "in the ledger (Integrity page).")

st.subheader("Per-asset detail")
det = r["rows"].copy()
modes = det.pop("result_mode")
det.insert(1, "strategy", [C.status_label(m) for m in modes])
st.caption(
    "Every column counts MODEL trades, so a row can read 0 while the money still moved — "
    "the benchmark's own trade is sealed separately and is not shown here. Strategy status "
    "is the sealed `result_mode`, in the same words the rest of the console uses: ACTIVE "
    "means at least two model trades; a one-trade row is still an ML result, too thin to "
    "compare; IDLE / HODL means the model never traded, so the capital simply followed the "
    "benchmark; NO-TRADE (floor) names the CONFIGURATION, not the row — it never cleared the "
    "Train-OOF floor, so the run is a diagnostic replay and is not promoted, even in the one "
    "case in the whole universe where such a model did trade out of sample (LSTM CEG, 40 "
    "trades).")

if len(det) > MAX_TIP_ROWS:
    st.dataframe(det, hide_index=True, width="stretch")
    st.caption(f"Over {MAX_TIP_ROWS} assets the per-ticker feature tooltips are switched off — "
               "open a single asset on the Asset Indicator page to read its manifest.")
else:
    st.markdown(_detail_css(), unsafe_allow_html=True)
    st.markdown(_detail_table_html(det, model), unsafe_allow_html=True)
    st.caption("Hover or Tab onto a ticker's ⓘ to see which optional features the Train-only "
               "search selected for that asset.")

st.subheader("XGB vs LSTM — asset-level result agreement (this basket)")
counts = B.venn_counts(covered, data.simulator_rows("xgb"),
                       data.simulator_rows("lstm"))
if counts is None:
    st.info("No diagram: no asset in this basket is a promoted strategy in BOTH stores. "
            "That is a result, not a gap — a basket of idle assets has no trades to compare.")
else:
    note = venn.winrate_note(counts["xw"], counts["lw"], data.payoff_ratios())
    if note:
        st.error(note, icon="⚠️")
    run = data.research_run()
    st.iframe(venn.venn_html(counts["xw"], counts["lw"], counts["bw"], counts["bl"],
                             counts["meta"],
                             {"xgb": run.get("xgb_oos", "—"), "lstm": run.get("lstm_oos", "—")}),
              height=1060)
    skipped = []
    if counts["nonstrategy"]:
        skipped.append(f"{counts['nonstrategy']} not a promoted strategy in both stores")
    if counts["excluded"]:
        skipped.append(f"{counts['excluded']} missing from one store")
    if skipped:
        st.caption("Left out of the diagram: " + " · ".join(skipped) + ".")

C.integrity_footer()
