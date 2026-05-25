# Data Audit — Alpaca S&P 500 daily OHLCV dataset

**Project:** Stocks Recommender Based on User Profile  
**Source:** Alpaca Markets, free IEX feed (`fetch_data.py`, 2026-05-24)  
**Files audited:** `data/tickers.csv`, `data/prices_long.csv` (rebuilt from `data/by_ticker/SP500/*.csv`)  
**Template reference:** [Liora — TEMPLATE Data Audit](https://docs.google.com/spreadsheets/d/1JI7_DBcSXJl5UxB8VY-Ybyr3T1NdK0PCDkO1t-ZRjj8/edit?gid=0#gid=0)

---

## Dataset overview

| Item | Value |
|------|-------|
| Universe | S&P 500 constituents (US equities) |
| Metadata rows (`tickers.csv`) | 503 |
| Price rows (`prices_long.csv`) | 726,018 |
| Symbols with price data | 503 / 503 |
| Calendar span (global min → max date) | 2017-11-15 → 2026-05-22 |
| Distinct trading dates | 1,723 |
| History per symbol (days) | min 139 · median 1,461 · mean 1,443 · max 1,716 |
| Duplicate (date, ticker) pairs | 0 |
| OHLC consistency violations | 0 (no `high < low`, no open/close outside range) |
| Weekend rows | 0 |

**Panel structure:** this is an **unbalanced panel**. Most symbols (~484) start on or after 2020-07-01 (IEX feed coverage); a small subset has longer history back to 2017; three recent listings have fewer than 500 trading days (Q, PSKY, SNDK). Feature engineering must account for listing-date heterogeneity and missing pre-listing history — not random missingness.

---

## Table 1 — `tickers.csv` (metadata)

**Category:** S&P 500 constituent reference table  
**Number of lines in the table:** 503 (+ 1 header)

| # | Name of the column | Description | Variable's type | % missing | Categorical / Quantitative | Distribution | Comments |
|---|-------------------|-------------|-----------------|-----------|---------------------------|--------------|----------|
| 1 | `ticker` | Exchange trading symbol (primary key for joining to price data) | `object` (string) | 0.0% | Categorical — more than 10 categories (503 unique) | 503 unique symbols; examples: `AAPL`, `MSFT`, `BRK.B`, `BF.B` | Scraped from Wikipedia. Alpaca uses dot notation for share classes (`BRK.B`), not Yahoo's dash form. No duplicates. |
| 2 | `name` | Company legal / common name | `object` (string) | 0.0% | Categorical — more than 10 categories (503 unique) | One name per ticker; e.g. *Apple Inc.*, *Berkshire Hathaway* | Free text; useful for display in Streamlit, not for modelling. Encoding not needed. |
| 3 | `sector` | GICS sector classification | `object` (string) | 0.0% | Categorical — more than 10 categories (11 unique) | Industrials (79), Financials (76), Information Technology (73), Health Care (59), Consumer Discretionary (48), Consumer Staples (36), Utilities (31), Real Estate (31), Materials (26), Communication Services (23), Energy (17) | Sector constraint in recommender (≤ 30% per sector) maps directly to this column. Imbalanced but expected for S&P 500. |
| 4 | `industry` | GICS sub-industry classification | `object` (string) | 0.0% | Categorical — more than 10 categories (127 unique) | Top industries: Health Care Equipment (16), Application Software (15), Electric Utilities (15), Semiconductors (14), Industrial Machinery (14) | Finer granularity than sector. High cardinality — consider grouping or target encoding if used as a feature. |
| 5 | `index` | Benchmark index membership | `object` (string) | 0.0% | Categorical — less than 10 categories | **SP500** (503) | Constant in current extract. Column kept for schema compatibility if DAX 40 is added later. |
| 6 | `country` | Country of listing / primary market | `object` (string) | 0.0% | Categorical — less than 10 categories | **US** (503) | Constant in current extract. All constituents are US-listed S&P 500 names. |

---

## Table 2 — `prices_long.csv` (daily OHLCV)

**Category:** Daily market prices (long format: one row per date × ticker)  
**Number of lines in the table:** 726,018 (+ 1 header)

| # | Name of the column | Description | Variable's type | % missing | Categorical / Quantitative | Distribution | Comments |
|---|-------------------|-------------|-----------------|-----------|---------------------------|--------------|----------|
| 1 | `date` | Trading session date (US market calendar) | `datetime64` (stored as date in CSV) | 0.0% | Quantitative (temporal ordinal) | Min: 2017-11-15 · Max: 2026-05-22 · 1,723 distinct dates | No weekend dates observed. Panel is sparse: not every ticker is present on every date (IPO/delisting/IEX start dates). Use inner/left joins carefully when building wide matrices. |
| 2 | `ticker` | Stock symbol (foreign key → `tickers.csv`) | `object` (string) | 0.0% | Categorical — more than 10 categories (503 unique) | 503 symbols; median 1,461 obs/symbol | Full coverage of S&P 500 universe. Three symbols with short history: Q (139 d), PSKY (200 d), SNDK (320 d). |
| 3 | `open` | Opening price (raw, unadjusted) | `float64` | 0.0% | Quantitative | Mean 202.40 · Std 398.49 · Min 3.67 · Q1 63.63 · Median 116.76 · Q3 219.89 · Max 9,857.51 | IEX feed, `adjustment=raw`. Right-skewed (high-price stocks like BRK.A pull the tail). All values > 0. |
| 4 | `high` | Highest traded price during the session (raw) | `float64` | 0.0% | Quantitative | Mean 204.59 · Std 402.24 · Min 3.95 · Q1 64.36 · Median 118.06 · Q3 222.20 · Max 9,948.28 | Always ≥ `low` (verified: 0 violations). Slightly ≥ `open`/`close` on average. |
| 5 | `low` | Lowest traded price during the session (raw) | `float64` | 0.0% | Quantitative | Mean 200.19 · Std 394.77 · Min 0.00 · Q1 62.90 · Median 115.46 · Q3 217.50 · Max 9,838.78 | **One row with `low = 0`:** AT&T (`T`), 2021-06-11 (open/high/close ≈ 29.35, volume 814k — likely bad IEX tick). Flag for cleaning. Otherwise all prices > 0. |
| 6 | `close` | Closing price (raw, unadjusted) | `float64` | 0.0% | Quantitative | Mean 202.42 · Std 398.49 · Min 3.73 · Q1 63.63 · Median 116.77 · Q3 219.93 · Max 9,933.51 | Primary field for raw return calculations. 60 daily moves > ±50% detected (splits, special situations, or thin IEX prints — review before using raw returns). |
| 7 | `adj_close` | Split/dividend-adjusted closing price | `float64` | 0.0% | Quantitative | Mean 169.52 · Std 339.15 · Min 0.75 · Q1 54.57 · Median 102.74 · Q3 195.50 · Max 9,933.51 | Fetched with `adjustment=all` and merged on date. Differs from `close` on **82.1%** of rows (mean abs diff ≈ 33.1 USD) — expected after corporate actions. **Use this column for return / Sharpe / correlation work**, not raw `close`. |
| 8 | `volume` | Number of shares traded during the session (IEX) | `float64` (integer-valued) | 0.0% | Quantitative | Mean 174,251 · Std 306,373 · Min 0 · Q1 44,156 · Median 86,797 · Q3 180,681 · Max 17,073,478 | IEX volume ≠ consolidated market volume (free-tier limitation). 93 zero-volume days (0.013%). Heavy right skew — consider log-transform for modelling. |

---

## Derived file — `prices_close_wide.csv` (not column-audited row-by-row)

| Property | Value |
|----------|-------|
| Shape | 1,723 dates × 503 tickers |
| Content | Pivot of `adj_close` |
| Missing cells | Present where a ticker was not listed yet or has no IEX bar for that date |
| Purpose | Correlation matrices, portfolio-level return series, heatmaps |

NaN pattern is **structural** (listing-date gaps), not random missing data.

---

## Global quality summary & biases

| Check | Result |
|-------|--------|
| Missing values (all columns) | 0.0% |
| Duplicate keys | None |
| Negative prices | None |
| OHLC logic | Pass |
| Metadata ↔ prices join | 100% match on `ticker` |

### Known limitations (document for Step 1 report)

1. **IEX feed, not SIP** — free Alpaca tier uses the IEX exchange feed. Prices align closely with Yahoo/Alpaca audits (~99.5% on overlaps) but volume is exchange-specific and may understate total market activity.
2. **Truncated history** — although the script requests 10 years, most symbols only have daily bars from ~mid-2020 onward on the free IEX feed. Do not assume a full 10-year balanced panel.
3. **Survivorship bias** — universe is current S&P 500 constituents as of download date (Wikipedia snapshot). Former index members that were removed are not included.
4. **Look-ahead in metadata** — sector/industry labels reflect current GICS classification, not historical reclassifications.
5. **Corporate actions** — `adj_close` handles splits/dividends, but extreme raw daily moves (> 50%, n = 60) should be inspected before volatility features.
6. **Single outlier** — one `low = 0` row should be filtered or winsorized in preprocessing.

---

## Suggested preprocessing actions (for Step 2)

- [ ] Drop or impute the single `low = 0` observation (`T`, 2021-06-11).
- [ ] Build returns from `adj_close`, not `close`.
- [ ] Align feature windows per ticker using each symbol's first valid date (handle unbalanced panel).
- [ ] Flag or winsorize the 60 raw daily moves exceeding ±50%.
- [ ] Join `tickers.csv` on `ticker` for sector/industry features and enforce the 30% sector cap at recommendation time.

---

*Generated from automated profiling of local CSV files on 2026-05-24. Statistics computed on the full 503-symbol extract (726,018 rows).*
