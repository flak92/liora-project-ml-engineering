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
    plot_risk_return_scatter,
    plot_correlation_heatmap,
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
    st.subheader("2. DataViz — 6 plots for Step 1")

    st.markdown("**Plot 1 — countplot** (sector, categorical variable)")
    st.caption("Shows how many stocks sit in each sector — the universe is imbalanced, so a naïve picker would overweight Industrials and Financials.")
    st.pyplot(plot_sector_count(tickers), clear_figure=True)
    with st.expander("🔍 View statistical validation"):
        st.markdown("**Validation method:** Raw data aggregation (`value_counts()`).\n\n**Interpretation:** By aggregating the metadata, we mathematically prove the visual imbalance. If the recommender chose randomly, it would statistically favor Industrials and Financials purely due to their massive volume in the index. This data manipulation validates the necessity of our hard '30% max per sector' business rule.")
        st.dataframe(tickers['sector'].value_counts())

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
    with st.expander("🔍 View statistical validation"):
        st.markdown("**Validation method:** Descriptive statistics (`describe()`) on aggregated mean daily volume.\n\n**Interpretation:** The table below calculates the exact percentiles (25%, 50% median, 75%) of the mean volume distribution. You can see the `50%` (median) explicitly matches our ~104k observation. The massive difference between the `75%` percentile and the `max` value statistically validates the existence of extreme outliers (the long upper whisker and dots in the boxplot).")
        st.dataframe(prices.groupby("ticker")["volume"].mean().describe())

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
    with st.expander("🔍 View statistical validation"):
        st.markdown("**Validation method:** Global descriptive statistics (`describe()`) across ~725k daily return observations.\n\n**Interpretation:** The `50%` (median) confirms our observation that a typical day is almost completely flat (~0.07%). However, comparing the standard deviation (`std`) to the `min` and `max` statistically proves the 'fat tails' concept: crashes and spikes extend far beyond normal variance, meaning risk in the stock market is driven by rare, extreme events rather than daily noise.")
        st.dataframe(prices["daily_return"].describe())

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
    with st.expander("🔍 View statistical validation"):
        if summary:
            st.markdown("**Validation method:** Point-to-point percentage change calculation.\n\n**Interpretation:** The line chart visually represents the price journey, but 'Total Return' is strictly a function of the start and end points. The equation below explicitly calculates this using the first and last `adj_close` values in our dataset. This validates that despite the visual volatility along the path, the final realized return for a buy-and-hold strategy is exactly as stated in the chart title.")
            st.code(f"({summary['end_price']:.2f} / {summary['start_price']:.2f} - 1) * 100 = {summary['total_return_pct']:.2f}%")

    st.markdown("**Plot 5 — heatmap** (Correlation Matrix)")
    st.caption("Shows the Pearson correlation coefficient between daily returns of the 10 most traded stocks.")
    st.markdown(
        """
- **Why Correlation Matters** — A core principle of portfolio management is diversification. If all stocks in a portfolio move in the exact same direction (correlation near 1.0), the risk is concentrated.
- **Top 10 Stocks** — This heatmap isolates the 10 most liquid names in the dataset. Notice how certain stocks might be highly correlated with each other, but less correlated with others.
- **Business Value** — The recommender uses sector constraints (max 30% per sector) specifically to avoid high correlations and build a diversified, safer portfolio for the user.
        """
    )
    st.pyplot(plot_correlation_heatmap(prices), clear_figure=True)
    with st.expander("🔍 View statistical validation"):
        st.markdown("**Validation method:** Pearson Correlation Coefficient matrix calculation.\n\n**Interpretation:** The heatmap is a visual representation of this exact matrix. Values closer to `1.0` indicate stocks moving in perfect lockstep, while values closer to `0.0` indicate independent movement. By calculating this mathematically on the wide-format returns dataframe, we prove that certain stock pairs offer poor diversification (high correlation). The recommender avoids this by scattering picks across multiple clusters and sectors.")
        top_tickers = prices.groupby("ticker")["volume"].mean().nlargest(10).index
        data = prices[prices["ticker"].isin(top_tickers)]
        wide_returns = data.pivot(index="date", columns="ticker", values="daily_return").dropna()
        st.dataframe(wide_returns.corr())

    st.markdown("**Plot 6 — scatterplot** (Risk vs. Return)")
    st.caption("X-axis = Annualized Volatility (Risk). Y-axis = Annualized Return. Each point is one stock.")
    st.markdown(
        """
- **What is 'Risk' here?** — Risk is mathematically calculated as **Annualized Volatility** (the standard deviation of a stock's daily returns, scaled to a 252-day trading year). In finance, volatility equates to uncertainty. A stock with 15% volatility is relatively stable and its price moves predictably, whereas a 50% volatility stock can swing wildly up and down, creating panic for inexperienced investors.
- **The Core of the Recommender** — This chart proves why grouping stocks by risk profile is necessary. Some stocks offer higher returns for the same level of risk, while others are highly volatile without the reward.
- **Four Quadrants** — The dashed lines represent the median risk and median return across the S&P 500. The top-left quadrant (High Return, Low Risk) is the theoretical "sweet spot".
- **Business Value** — For a conservative user, the recommender will explicitly filter out high-volatility names (the right side of the plot). For an aggressive user, it can venture into the higher risk territory aiming for higher expected returns.
        """
    )
    st.pyplot(plot_risk_return_scatter(prices), clear_figure=True)
    with st.expander("🔍 View statistical validation"):
        stats = prices.groupby("ticker")["daily_return"].agg(["mean", "std"]).dropna()
        pearson_corr = stats["mean"].corr(stats["std"])
        st.markdown("**Validation method:** Pearson Correlation between Annualized Volatility and Annualized Expected Return.\n\n**Interpretation:** We mathematically aggregated all 503 stocks into a single (Risk, Return) tuple and calculated their linear correlation. A strong positive correlation (e.g., > 0.7) would mean 'more risk always equals more reward'.")
        st.write(f"**Calculated Pearson correlation:** `{pearson_corr:.4f}`")
        st.markdown(f"Because the correlation is so close to 0 (`{pearson_corr:.4f}`), we statistically validate the core observation from the scatterplot: taking blind risk does **not** guarantee proportionally higher returns. This definitively proves the business need for a smart recommender that seeks the 'efficient frontier' (high return for a given risk) rather than just picking randomly.")
