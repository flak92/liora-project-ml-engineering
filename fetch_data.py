"""Fetch historical daily OHLCV data for S&P 500 constituents from Alpaca Markets.

Primary data source for the Stocks Recommender Based on User Profile (mentor-approved
2026-05-22). Uses Alpaca's free IEX feed via alpaca-py and the REST bars API.

Outputs (written under ./data/):
    - tickers.csv                    : metadata for every ticker
    - prices_long.csv                : one row per (date, ticker)
    - prices_close_wide.csv          : wide-format adjusted close matrix
    - by_ticker/SP500/{TKR}.csv      : individual OHLCV file per ticker
    - failed_tickers.csv             : tickers that could not be downloaded

Credentials are read from .env.local in this directory:
    ALPACA_API_KEY
    ALPACA_API_SECRET

Usage:
    python fetch_data.py                  # 10-year window (default)
    python fetch_data.py --years 5        # shorter history
    python fetch_data.py --limit 20       # smoke test: first 20 tickers only
    python fetch_data.py --batch-size 5   # symbols per Alpaca request
"""

from __future__ import annotations

import argparse
import io
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


DATA_DIR = Path(__file__).parent / "data"
ENV_FILE = Path(__file__).parent / ".env.local"

SP500_WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
)

DEFAULT_FEED = "iex"
REQUEST_PAUSE_SEC = 0.35


def _load_credentials() -> tuple[str, str]:
    load_dotenv(ENV_FILE)
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError(
            f"Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET in {ENV_FILE}."
        )
    return api_key, api_secret


def _read_wiki_tables(url: str) -> list[pd.DataFrame]:
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_sp500_tickers() -> pd.DataFrame:
    """Return S&P 500 constituents with ticker, name, sector, industry metadata."""
    tables = _read_wiki_tables(SP500_WIKI)
    df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
    df.columns = ["ticker", "name", "sector", "industry"]

    # Alpaca accepts standard symbols with dots (e.g. BRK.B), unlike Yahoo's BRK-B.
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["index"] = "SP500"
    df["country"] = "US"
    return df


def _bars_to_long(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.reset_index()
    out = out.rename(columns={"timestamp": "date"})
    if "symbol" in out.columns:
        out = out.drop(columns="symbol")
    if "date" not in out.columns:
        return pd.DataFrame()

    out["date"] = pd.to_datetime(out["date"], utc=True).dt.tz_convert(None).dt.normalize()
    out["ticker"] = ticker
    return out


def _fetch_bars_rest(
    api_key: str,
    api_secret: str,
    tickers: list[str],
    start: datetime,
    end: datetime,
    *,
    adjustment: str,
    feed: str,
) -> pd.DataFrame:
    """Fetch daily bars via REST, following next_page_token pagination."""
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    base_params = {
        "symbols": ",".join(tickers),
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "adjustment": adjustment,
        "feed": feed,
        "limit": 10_000,
        "sort": "asc",
    }

    collected: dict[str, list[dict]] = {t: [] for t in tickers}
    page_token: str | None = None

    while True:
        params = dict(base_params)
        if page_token:
            params["page_token"] = page_token

        resp = requests.get(ALPACA_DATA_URL, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        for symbol, bars in payload.get("bars", {}).items():
            collected.setdefault(symbol, []).extend(bars)

        page_token = payload.get("next_page_token")
        if not page_token:
            break

    rows: list[dict] = []
    for ticker, bars in collected.items():
        for bar in bars:
            rows.append(
                {
                    "date": pd.to_datetime(bar["t"], utc=True).tz_convert(None).normalize(),
                    "ticker": ticker,
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                }
            )

    return pd.DataFrame(rows)


def _fetch_bars(
    client: StockHistoricalDataClient,
    api_key: str,
    api_secret: str,
    tickers: list[str],
    start: datetime,
    end: datetime,
    *,
    adjustment: str,
    feed: str,
) -> pd.DataFrame:
    """Fetch daily bars; multi-symbol requests use REST pagination."""
    if len(tickers) > 1:
        return _fetch_bars_rest(
            api_key,
            api_secret,
            tickers,
            start,
            end,
            adjustment=adjustment,
            feed=feed,
        )

    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=feed,
        adjustment=adjustment,
        limit=10_000,
    )
    response = client.get_stock_bars(request)
    if response.df is None or response.df.empty:
        return pd.DataFrame()
    return _bars_to_long(response.df, tickers[0])


def _merge_raw_and_adjusted(raw: pd.DataFrame, adjusted: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if adjusted.empty:
        out = raw.copy()
        out["adj_close"] = out["close"]
        return out

    merged = raw.merge(
        adjusted[["date", "ticker", "close"]].rename(columns={"close": "adj_close"}),
        on=["date", "ticker"],
        how="left",
    )
    merged["adj_close"] = merged["adj_close"].fillna(merged["close"])
    return merged


def _download_one(
    client: StockHistoricalDataClient,
    api_key: str,
    api_secret: str,
    ticker: str,
    start: datetime,
    end: datetime,
    *,
    feed: str,
) -> pd.DataFrame | None:
    try:
        raw = _fetch_bars(
            client, api_key, api_secret, [ticker], start, end, adjustment="raw", feed=feed
        )
        if raw.empty:
            return None
        adjusted = _fetch_bars(
            client, api_key, api_secret, [ticker], start, end, adjustment="all", feed=feed
        )
        return _merge_raw_and_adjusted(raw, adjusted)
    except Exception:
        return None


def download_prices(
    client: StockHistoricalDataClient,
    api_key: str,
    api_secret: str,
    tickers: list[str],
    years: int,
    *,
    feed: str,
    batch_size: int,
) -> tuple[pd.DataFrame, list[str]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365 * years)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    n_batches = -(-len(tickers) // batch_size)

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  batch {i // batch_size + 1}/{n_batches} ({len(batch)} tickers)…")

        try:
            raw = _fetch_bars(
                client, api_key, api_secret, batch, start, end, adjustment="raw", feed=feed
            )
            adjusted = _fetch_bars(
                client, api_key, api_secret, batch, start, end, adjustment="all", feed=feed
            )
        except Exception as exc:
            print(f"    batch failed: {exc!s}; retrying tickers individually")
            raw = pd.DataFrame()
            adjusted = pd.DataFrame()

        if raw.empty:
            retry_list = list(batch)
        else:
            got = set(raw["ticker"].unique())
            retry_list = [t for t in batch if t not in got]
            frames.append(_merge_raw_and_adjusted(raw, adjusted))

        for ticker in retry_list:
            time.sleep(REQUEST_PAUSE_SEC)
            df = _download_one(client, api_key, api_secret, ticker, start, end, feed=feed)
            if df is None:
                failed.append(ticker)
                continue
            frames.append(df)

        time.sleep(REQUEST_PAUSE_SEC)

    if not frames:
        return pd.DataFrame(), failed

    prices = pd.concat(frames, ignore_index=True)
    keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    prices = prices[[c for c in keep if c in prices.columns]]
    return prices.drop_duplicates(subset=["date", "ticker"]).sort_values(["ticker", "date"]), failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, default=10, help="history window in years (default: 10)")
    parser.add_argument("--limit", type=int, default=None, help="cap number of tickers (smoke test)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="tickers per Alpaca batch request (default: 1; use REST pagination for larger batches)",
    )
    parser.add_argument("--feed", type=str, default=DEFAULT_FEED, help="Alpaca data feed (default: iex)")
    args = parser.parse_args()

    api_key, api_secret = _load_credentials()
    client = StockHistoricalDataClient(api_key, api_secret)

    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching S&P 500 ticker list from Wikipedia…")
    tickers_df = get_sp500_tickers()
    print(f"  S&P 500: {len(tickers_df)} tickers")

    if args.limit:
        tickers_df = tickers_df.head(args.limit)
        print(f"  --limit set: using first {len(tickers_df)} tickers")

    tickers_df.to_csv(DATA_DIR / "tickers.csv", index=False)
    print(f"  wrote {DATA_DIR / 'tickers.csv'}")

    print(f"\nDownloading {args.years}y of daily history from Alpaca ({args.feed} feed)…")
    prices, failed = download_prices(
        client,
        api_key,
        api_secret,
        tickers_df["ticker"].tolist(),
        args.years,
        feed=args.feed,
        batch_size=args.batch_size,
    )

    if prices.empty:
        print("No prices downloaded — check credentials, network, or Alpaca status.")
        return

    prices.to_csv(DATA_DIR / "prices_long.csv", index=False)
    print(f"  wrote {DATA_DIR / 'prices_long.csv'} ({len(prices):,} rows)")

    by_ticker_dir = DATA_DIR / "by_ticker"
    count = 0
    for ticker, df in prices.groupby("ticker"):
        sub = by_ticker_dir / "SP500"
        sub.mkdir(parents=True, exist_ok=True)
        df.drop(columns="ticker").to_csv(sub / f"{ticker}.csv", index=False)
        count += 1
    print(f"  wrote per-ticker files → {by_ticker_dir}/SP500/ ({count} tickers)")

    wide = prices.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    wide.to_csv(DATA_DIR / "prices_close_wide.csv")
    print(f"  wrote {DATA_DIR / 'prices_close_wide.csv'} ({wide.shape[0]} dates x {wide.shape[1]} tickers)")

    if failed:
        pd.DataFrame({"ticker": failed}).to_csv(DATA_DIR / "failed_tickers.csv", index=False)
        print(f"  {len(failed)} tickers failed → {DATA_DIR / 'failed_tickers.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
