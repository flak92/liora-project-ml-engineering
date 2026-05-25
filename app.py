"""Stocks Recommender Based on User Profile — Streamlit app (course pattern from module 117).

Run:
    source .venv/bin/activate
    streamlit run app.py
"""

import streamlit as st

from src.data_loader import load_prices, load_tickers
from src.plots import (
    get_price_line_summary,
    plot_price_line,
    plot_return_hist,
    plot_sector_count,
    plot_volume_box,
)

st.set_page_config(page_title="Stocks Recommender Based on User Profile", layout="wide")
st.title("Stocks Recommender Based on User Profile")


@st.cache_data
def get_data():
    tickers = load_tickers()
    prices = load_prices()
    return tickers, prices


tickers, prices = get_data()

page = st.sidebar.radio("Page", ["Exploration", "DataViz"])

if page == "Exploration":
    st.subheader("1. Exploration")
    st.write("First rows of the metadata table:")
    st.dataframe(tickers.head(10))
    st.write(f"Prices table shape: {prices.shape[0]} rows, {prices.shape[1]} columns")
    st.write("Summary statistics on price columns:")
    st.dataframe(prices[["open", "high", "low", "close", "adj_close", "volume"]].describe())

    if st.checkbox("Show missing values"):
        st.dataframe(prices.isna().sum())

elif page == "DataViz":
    st.subheader("2. DataViz — 4 plots for Step 1")

    st.markdown("**Plot 1 — countplot** (sector, categorical variable)")
    st.caption("Shows how many stocks sit in each sector — the universe is imbalanced, so a naïve picker would overweight Industrials and Financials.")
    st.pyplot(plot_sector_count(tickers), clear_figure=True)

    st.markdown("**Plot 2 — boxplot** (mean daily volume per stock)")
    st.caption(
        "Y-axis = average shares traded per day on IEX (k = thousands). "
        "Red dashed line = median (~104k shares/day)."
    )
    st.markdown(
        """
- **503 stocks, one average per name** — each point is the mean daily volume over all days in the dataset (not a single day).
- **Typical stock ~104k shares/day** — half of S&P names trade less than the median; half trade more.
- **Middle 50% sit between ~60k and ~200k/day** — the blue box; most names are in this band, not extremely quiet or hyper-liquid.
- **A few names dominate liquidity** — e.g. NVDA ~1.6M vs NVR ~1.7k shares/day on average; the dots are those extreme mega-liquid names.
        """
    )
    st.pyplot(plot_volume_box(prices), clear_figure=True)

    st.markdown("**Plot 3 — histplot** (daily returns)")
    st.caption(
        "X-axis = % price change vs the previous day. Y-axis = how many days fall in each bin (k = thousands)."
    )
    st.markdown(
        """
- **What ~725k means** — We have **503 stocks**, each with **~1,460 trading days** in the CSV. That is **503 × ~1,460 ≈ 726k rows** (one row = one stock on one day). We drop **503 days** (the first day of each stock, no “yesterday” to compare) → **~725k daily returns**. Each bar asks: “On how many of those days did the price move by this %?”
- **Typical day ≈ +0.07%** — red dashed line (median); prices usually move very little in one session.
- **Most mass sits near 0%** — the histogram is highest around “no big move”; that is normal market behaviour.
- **Long tails left and right** — rare but real crashes (e.g. −53% in one day) and spikes (e.g. +127%); risk is in those tails.
- **Why it matters for the recommender** — average return is tiny (~0.07%/day), but a typical daily move is ~1% (up or down) and the spread across all days (std) is ~2%/day — so clustering must use **risk**, not return alone.
        """
    )
    st.pyplot(plot_return_hist(prices), clear_figure=True)

    st.markdown("**Plot 4 — lineplot** (price over time)")
    st.caption(
        "Y-axis = split/dividend-adjusted close (USD). Use the dropdown to compare how different names evolved over the same calendar."
    )
    ticker = st.selectbox("Choose a ticker", sorted(tickers["ticker"]))
    summary = get_price_line_summary(prices, ticker)
    st.markdown(
        f"""
- **What the line shows** — **{summary['n_days']:,} trading days** for **{ticker}**, from **{summary['start_date'].strftime('%Y-%m-%d')}** to **{summary['end_date'].strftime('%Y-%m-%d')}**. Each point is the **adjusted close** (USD): corporate actions are already baked in, so the path is comparable over time.
- **This ticker in our window** — starts at **{summary['start_price']:.2f} USD**, ends at **{summary['end_price']:.2f} USD** → **total return {summary['total_return_pct']:+.0f}%** over the full series (not per day).
- **Same market, different stories** — compare dropdown names: e.g. **NVDA ~+1973%**, **AAPL ~+236%**, **KO ~+100%**, **NVR ~+53%** in this CSV. Steep lines are past winners, not a promise of future performance.
- **Price path ≠ daily risk** — a smooth upward line can still have **~2% daily volatility** (Plot 3). The Y-axis here is **level**, not day-to-day swings.
- **Why it matters for the recommender** — Step 1 clusters by **risk profile**, not by “who drew the prettiest line”. This plot is a **sanity check** for demos: when we recommend a stock from a cluster, you can open it here and see **why that name’s history matches** (or differs from) the user’s horizon and loss tolerance.
        """
    )
    st.pyplot(plot_price_line(prices, ticker), clear_figure=True)
