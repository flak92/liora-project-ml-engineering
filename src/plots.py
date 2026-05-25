"""Plots from the Liora course (Seaborn 112 + ML2 methodology EDA)."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_plot():
    sns.set()
    return plt.figure()


def plot_sector_count(tickers: pd.DataFrame):
    """Categorical variable — countplot (methodology Step 2)."""
    fig = setup_plot()
    sns.countplot(x="sector", data=tickers)
    plt.title("Number of stocks per sector")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_volume_box(prices: pd.DataFrame):
    """Quantitative variable — boxplot on mean daily volume per stock (503 points)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.set()
    mean_volume = prices.groupby("ticker")["volume"].mean()
    median = mean_volume.median()

    # Vertical box: volume on Y (easier to read than horizontal).
    sns.boxplot(y=mean_volume, ax=ax, color="skyblue", width=0.4)
    ax.set_xticks([])
    ax.set_xlabel("Each box = 503 stocks (one mean volume per stock)")
    ax.set_ylabel("Mean daily volume (thousands of shares / day, IEX)")
    ax.set_title("Mean daily volume per stock")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _pos: f"{x / 1_000:.0f}k"))

    # Mark the median (~104k) so it is visible on the chart.
    ax.axhline(median, color="darkred", linewidth=1.2, linestyle="--", label=f"Median ≈ {median / 1_000:.0f}k")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_return_hist(prices: pd.DataFrame):
    """Quantitative variable — histogram (methodology Step 2)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.set()
    returns = prices["daily_return"].dropna()
    median = returns.median()

    sns.histplot(returns, bins=50, kde=True, ax=ax, color="steelblue")

    ax.set_xlabel("Daily price change (% per day, split/dividend-adjusted close)")
    ax.set_ylabel("Number of days (thousands)")
    ax.set_title("Daily return distribution — all S&P 500 stocks, all trading days")

    def _pct_label(x, _pos):
        pct = x * 100
        if abs(pct) < 10:
            return f"{pct:.1f}%"
        return f"{pct:.0f}%"

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_pct_label))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: f"{y / 1_000:.0f}k"))

    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.axvline(median, color="darkred", linestyle="--", linewidth=1.2, label=f"Median ≈ {median * 100:.2f}%/day")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def get_price_line_summary(prices: pd.DataFrame, ticker: str) -> dict:
    """Summary stats for Plot 4 bullets and title."""
    data = prices.loc[prices["ticker"] == ticker]
    if data.empty:
        return {}
    start, end = data.iloc[0], data.iloc[-1]
    total_return_pct = (end["adj_close"] / start["adj_close"] - 1) * 100
    return {
        "ticker": ticker,
        "start_date": start["date"],
        "end_date": end["date"],
        "start_price": float(start["adj_close"]),
        "end_price": float(end["adj_close"]),
        "n_days": len(data),
        "total_return_pct": float(total_return_pct),
    }


def plot_price_line(prices: pd.DataFrame, ticker: str):
    """Time series — lineplot (Matplotlib 111 / Seaborn 112)."""
    summary = get_price_line_summary(prices, ticker)
    if not summary:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.text(0.5, 0.5, f"No data for {ticker}", ha="center", va="center")
        ax.axis("off")
        return fig

    data = prices.loc[prices["ticker"] == ticker]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.set()
    sns.lineplot(data=data, x="date", y="adj_close", ax=ax, linewidth=1.4, color="steelblue")

    ax.set_xlabel("Trading date")
    ax.set_ylabel("Adjusted close (USD)")
    ax.set_title(
        f"{ticker} — price path (split/dividend-adjusted)\n"
        f"{summary['start_date'].strftime('%Y-%m-%d')} to {summary['end_date'].strftime('%Y-%m-%d')} · "
        f"total return {summary['total_return_pct']:+.0f}%",
        fontsize=11,
        pad=10,
    )
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    fig.subplots_adjust(left=0.08, bottom=0.16, right=0.98, top=0.82)
    return fig
