"""Load CSV files used in the Streamlit app."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_tickers() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "tickers.csv")


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "prices_long.csv", parse_dates=["date"])
    df = df.sort_values(["ticker", "date"])
    df["daily_return"] = df.groupby("ticker")["adj_close"].pct_change()
    return df
