"""app/basket.py — the basket arithmetic behind the simulator page. No Streamlit here.

Two functions, both pure: `compute_basket()` turns a set of tickers into the three
numbers that must never be conflated, and `venn_counts()` turns it into the shares the
pixel diagram draws. Keeping them free of Streamlit means the predicates — which are the
honest part of this feature — can be read, tested and reused without a UI runtime.

There is no per-trade data in this release, so a basket is a sum of per-asset ENDPOINTS.
No equity curve, no drawdown path, no timing, no correlation between holdings.
"""
ENTRY_USD = 1000.0

# The columns the detail table shows, in order. `strategy` (the sealed result_mode, rendered
# through components.status_label) is inserted by the page after `ticker`.
DETAIL_COLS = ["ticker", "end_capital", "return_pct", "trades", "wins", "losses",
               "max_drawdown_pct", "win_rate_pct", "profit_factor"]


def compute_basket(selected, df, hodl):
    """Basket outcome from the sealed per-asset rows, reported as THREE separate numbers.

    The distinction matters and used to be invisible. When a model never traded an asset, the
    sealed row still carries a capital path — the buy-and-hold one, because that is what actually
    happened to the money. Summing every row therefore answers "what happened to the basket",
    NOT "what did the model do". Both are legitimate questions, so both are returned, named:

      executed  every selected row — what the capital actually did
      ml        only rows that are a MODEL RESULT (see the predicate below)
      hodl      the same basket held over the identical window (price-only benchmark)

    A MODEL RESULT means the configuration cleared its own Train-OOF trade floor AND the model
    actually traded out of sample: `trade_floor_met == 1 AND model_trades > 0`. Both halves are
    needed and neither is redundant:
      - `model_trades > 0` alone would count a DIAGNOSTIC replay. LSTM CEG is sealed
        TRAIN_OOF_FLOOR_NOT_MET yet made 40 OOS trades — the model traded, but its configuration
        was never accepted as a strategy, so its capital is not a strategy result.
      - `trade_floor_met == 1` alone would count assets that cleared the floor on Train and then
        never traded out of sample; their capital path is the benchmark's.
    CEG is the only row in the whole universe where the two halves disagree (all 136 XGB
    floor-not-met rows have model_trades = 0), so this predicate moves exactly one asset. It is
    a correction of MEANING, not of magnitude — and CEG stays visible in the per-asset table,
    labelled by its sealed result_mode.

    NOTE the deliberate difference from the comparison diagram, which qualifies on
    `result_mode == 'ML_MULTI_TRADE'` (floor met AND >= 2 trades). That is a STRICTER set — 774
    rows against 786 — because win/loss shares need at least two trades to mean anything. The 12
    rows between them are ML_ONE_TRADE_LOW_EVIDENCE: still a model result, too thin to compare.
    Two questions, two predicates, two different words. (Counts re-derived here.)
    """
    rows = df[df["ticker"].isin(selected)]
    invested = ENTRY_USD * len(rows)
    final = float(rows["end_capital"].sum())
    pnl = final - invested

    traded = rows[(rows["model_trades"].fillna(0) > 0) & (rows["trade_floor_met"].fillna(0) == 1)]
    ml_invested = ENTRY_USD * len(traded)
    ml_final = float(traded["end_capital"].sum())

    held = [t for t in rows["ticker"] if t in hodl]
    hodl_final = sum(ENTRY_USD * (1.0 + hodl[t] / 100.0) for t in held)
    hodl_invested = ENTRY_USD * len(held)

    cols = DETAIL_COLS + (["result_mode"] if "result_mode" in rows.columns else [])
    return {"n": len(rows), "invested": invested, "final": final, "pnl": pnl,
            "return_pct": (pnl / invested * 100.0) if invested else 0.0,
            "ml_n": len(traded), "ml_invested": ml_invested, "ml_final": ml_final,
            "ml_return_pct": ((ml_final - ml_invested) / ml_invested * 100.0) if ml_invested else None,
            "no_model_trade_n": len(rows) - len(traded),
            "hodl_final": hodl_final, "hodl_n": len(held),
            "hodl_return_pct": ((hodl_final - hodl_invested) / hodl_invested * 100.0)
            if hodl_invested else None,
            "beats_hodl": final > hodl_final if held else None,
            "rows": rows[cols].reset_index(drop=True)}


def venn_counts(basket, dfx, dfl):
    """The basket's TRADES under both sealed stores, as percentage points of each model's own
    trade count (that is what the pixel-Venn draws: 1 square = 1 pp, 100 squares per model).

    xw / lw   winning trades of that model as a share of its basket trades (sealed `wins` column)
    bw / bl   the models AGREE on the ticker: trades on tickers where both ended net-positive
              (bw) / both net-negative (bl). Each share has its own denominator, so the drawn
              block is the smaller of the two and both raw shares travel in `meta`.
    A buy-and-hold fallback carries ZERO model trades (the benchmark's single trade is sealed
    separately as `benchmark_trades`), so it cannot clear the min-2 rule and never enters the
    comparison. Only tickers present in BOTH stores are compared; None when nothing is comparable.
    """
    cols = ["ticker", "model_trades", "wins", "losses", "return_pct", "result_mode"]
    if dfx.empty or dfl.empty:
        return None
    x = dfx[dfx["ticker"].isin(basket)][cols].set_index("ticker").rename(
        columns={"model_trades": "trades"})
    ll = dfl[dfl["ticker"].isin(basket)][cols].set_index("ticker").rename(
        columns={"model_trades": "trades"})
    both = sorted(set(x.index) & set(ll.index))
    # A ticker enters the comparison only where BOTH models produced a PROMOTED strategy, read
    # from the sealed taxonomy: result_mode == 'ML_MULTI_TRADE'. That label is exactly
    # (trade_floor_met == 1 AND model_trades >= 2) — verified equal over all 993 sealed rows —
    # so reading the label rather than re-deriving the rule keeps the app and the store from
    # ever drifting apart. It also stops a diagnostic replay (floor not met, but the model did
    # trade — LSTM CEG) from being drawn as a strategy.
    common = [t for t in both
              if x.loc[t, "result_mode"] == "ML_MULTI_TRADE"
              and ll.loc[t, "result_mode"] == "ML_MULTI_TRADE"]
    nonstrategy = len(both) - len(common)
    if not common:
        return None
    x, ll = x.loc[common], ll.loc[common]
    tx, tl = int(x["trades"].sum()), int(ll["trades"].sum())
    if tx == 0 or tl == 0:
        return None
    agree_win = (x["return_pct"] > 0) & (ll["return_pct"] > 0)
    agree_lose = (x["return_pct"] <= 0) & (ll["return_pct"] <= 0)
    bw_x = 100 * x.loc[agree_win, "wins"].sum() / tx
    bw_l = 100 * ll.loc[agree_win, "wins"].sum() / tl
    bl_x = 100 * x.loc[agree_lose, "losses"].sum() / tx
    bl_l = 100 * ll.loc[agree_lose, "losses"].sum() / tl
    return {"xw": round(100 * x["wins"].sum() / tx), "lw": round(100 * ll["wins"].sum() / tl),
            "bw": round(min(bw_x, bw_l)), "bl": round(min(bl_x, bl_l)),
            "meta": {"n": len(common), "tx": tx, "tl": tl,
                     "wx": int(x["wins"].sum()), "wl": int(ll["wins"].sum()),
                     "bw_x": bw_x, "bw_l": bw_l, "bl_x": bl_x, "bl_l": bl_l},
            "nonstrategy": nonstrategy,
            "excluded": len(basket) - len(both)}
