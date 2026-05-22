# Liora Stock — Stock Portfolio Recommender

A beginner-friendly stock portfolio recommender that turns a short questionnaire about the investor (experience, time horizon, loss tolerance, monthly budget, sector preferences) into a small portfolio of 5–10 stocks picked from a known universe (S&P 500 + DAX 40), along with the risk metrics that justify each pick.

> **Disclaimer:** This project is decision support for a beginner investor, **not** financial advice.

**Training course:** Data Scientist · **Difficulty:** 8/10

## Problem

Retail investors who want to pick individual stocks usually have to choose between paying for a broker's recommendation or scrolling through hundreds of tickers in an online screener with no idea where to begin. Existing tools either lump everyone into a generic "aggressive / moderate / conservative" bucket, or assume the user already speaks the language of finance.

## Approach

The pipeline is split into three steps:

1. **Clustering** — group the stock universe by historical risk/return profile (volatility, Sharpe ratio, beta vs. index, max drawdown, sector) using K-Means or hierarchical clustering. This yields natural groups: stable dividend payers, growth tech, cyclicals, defensive low-beta names, etc.
2. **Return ranking** — train a regression model (linear regression as baseline, then Random Forest / Gradient Boosting) to estimate the expected 12-month return from fundamentals and a few technical features. The goal is not to beat the market — only to rank stocks _within_ each cluster.
3. **Recommendation** — map the user's questionnaire to a target risk profile, pick the relevant clusters, and return the top-N stocks per cluster, with the constraint that **no single sector exceeds 30%** of the portfolio.

The whole thing is wrapped in a **Streamlit** web app so the questionnaire can be played with interactively and the recommended portfolio is shown together with the metrics that support each pick.

## Data sources

- **[Alpaca Markets](https://alpaca.markets/)** (free IEX feed) — **primary** source for daily OHLCV. Selected on 2026-05-22 with mentor approval (see [`meeting_notes/2026-05-22.md`](meeting_notes/2026-05-22.md)). Rationale: broker-grade validated data pipeline, richer programmatic API, and compatibility with backtesting tools like QuantConnect. A side-by-side audit against yfinance (`data provider choose.html`) showed ~99.5% similarity on overlapping (ticker, date) pairs.
- **[yfinance](https://pypi.org/project/yfinance/)** — kept as a fallback / cross-check source only. Observed limitations: unvalidated data quality and occasional incorrect average-close values on some days.
- **S&P 500 constituents** — scraped from Wikipedia; ~503 tickers. _Note:_ Alpaca's free tier does **not** cover DAX 40, so the initial universe is S&P 500 only. The clustering / ranking methodology is market-agnostic, so DAX 40 can be re-added later via a different provider if needed.
- **[Alpha Vantage](https://www.alphavantage.co/)** (free tier) — backup source for P/E, EPS, dividend yield.
- **[Kaggle Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)** — ~7,000 US tickers with daily OHLCV; offline backup.
- **[FRED](https://fred.stlouisfed.org/)** — risk-free rate (10-year US Treasury) for the Sharpe ratio.

## Tech stack

`pandas`, `numpy`, `scipy` for data wrangling and statistics · `matplotlib`, `seaborn` for visualisations · `scikit-learn` for clustering, regression and pipelines · `streamlit` for the demo.

## Setup

Requires Python 3.9+ (tested on 3.14). On macOS the system Python is "externally managed" (PEP 668), so install everything inside a virtual environment.

### 1. Create and activate a virtual environment

```bash
cd apr26_bds_int_stock_portfolio
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows PowerShell
```

You'll know the venv is active when your shell prompt shows `(.venv)`. To leave it later: `deactivate`.

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs the data-acquisition dependencies (`yfinance`, `pandas`, `lxml`, `html5lib`, `requests`). The ML/visualisation libraries listed in **Tech stack** above will be added to `requirements.txt` as later pipeline steps are built.

### 3. Download the price data

```bash
python fetch_data.py                  # 5 years of history (default)
python fetch_data.py --years 10       # 10-year window
python fetch_data.py --limit 20       # smoke test: first 20 tickers only
python fetch_data.py --batch-size 25  # smaller batches if your network is flaky
```

Outputs land in `./data/`:

| File                              | Format       | Purpose                                              |
| --------------------------------- | ------------ | ---------------------------------------------------- |
| `tickers.csv`                     | metadata     | ticker, name, sector, industry, index, country       |
| `prices_long.csv`                 | long         | one row per (date, ticker) — best for feature eng.   |
| `prices_close_wide.csv`           | wide         | adj. close matrix — best for returns / correlations  |
| `by_ticker/SP500/{TKR}.csv`       | per-ticker   | one OHLCV file per S&P 500 stock                     |
| `by_ticker/DAX40/{TKR}.csv`       | per-ticker   | one OHLCV file per DAX 40 stock                      |
| `failed_tickers.csv`              | retry list   | only written if some downloads failed                |

A full 10-year run covers ~543 tickers and takes roughly 10–15 minutes, producing ~150 MB of data. Yahoo Finance occasionally throttles or drops connections; the script automatically retries failed tickers individually before giving up.

### Troubleshooting

- **`HTTP 403` from Wikipedia** — already worked around with a browser User-Agent.
- **`OperationalError: unable to open database file`** — yfinance's local SQLite cache can't be written. Usually means the root volume is full. Free disk space and clear the cache: `rm -rf ~/Library/Caches/py-yfinance`.
- **`getaddrinfo() thread failed to start`** / **`cert verify locations`** — also disk-pressure symptoms; same fix as above.
- **`zsh: permission denied: .venv/bin/activate`** — `activate` must be *sourced*, not executed: `source .venv/bin/activate`.

## Deliverables

- Exploration, data visualization and pre-processing **report**.
- Modeling **report**.
- Final **report** + associated **code**.
- **Streamlit** application + oral **defense**.

## Project timeline & deadlines

Mentor: **Paul Grolier**. Framing meeting tentatively scheduled for **Wednesday 2026-05-13** afternoon (Zoom). Original kickoff message: [Slack thread](https://dstinternatio-d5c8877.slack.com/archives/C0B2TU0HJM9/p1778517743660219).

| Step | Deliverable                                                                                                  | Deadline                    |
| ---- | ------------------------------------------------------------------------------------------------------------ | --------------------------- |
| 0    | Framing meeting                                                                                              | Week of 2026-05-11          |
| 1    | Data mining + DataViz                                                                                        | **2026-05-27**              |
| 2    | Pre-processing + feature engineering → **Rendering 1**: data exploration, data viz and pre-processing report | **2026-06-03**              |
| 3.1  | Modeling — baseline models, first iterations                                                                 | **2026-06-10**              |
| 3.2  | Modeling — ML metrics, optimization, model comparison                                                        | **2026-06-24**              |
| 3.3  | Modeling — bagging/boosting, Deep Learning, interpretability → **Rendering 2**: modeling report              | **2026-07-01**              |
| 4    | Final report + clean commented code on GitHub                                                                | **2026-07-08**              |
| 5    | Streamlit application + oral defense                                                                         | **2026-07-23 – 2026-07-21** |

### Step requirements

- **Step 1 — Data mining + DataViz** _(due 2026-05-27)_: define context and scope, near-exhaustive dataset analysis to highlight structure, difficulties and biases. Use the _TEMPLATE - Data Audit_. Deliver **at least 5 relevant visualizations**, each with a precise commentary providing a _business_ opinion **and** validated by data manipulation or a statistical test.
- **Step 2 — Pre-processing + feature engineering** _(due 2026-06-03)_: cleaning, transformations, feature engineering, dataset enrichment. End state: dataset ready for in-depth analysis / ML / DL modeling. After Rendering 1, the mentor will instantiate the official GitHub repo for the group following the provided template.
- **Step 3 — Modeling** _(2026-06-10 → 2026-07-01)_: baseline → optimization → advanced (bagging/boosting + Deep Learning) → interpretability + scientific & business conclusions.
- **Step 4 — Final report** _(due 2026-07-08)_: merges Renderings 1 & 2, adds conclusion and opening, plus clean commented code on GitHub.
- **Step 5 — Defense** _(2026-07-23 → 2026-07-21)_: 20 min presentation + 10 min jury Q&A. Either Powerpoint + Streamlit demo, or the entire presentation through the Streamlit app. The app must be aesthetically pleasing with several tabs, carefully coded (no re-training of the model at runtime) and bug-free.

> Intermediate and final reports must include illustrations, a proper layout and no spelling mistakes. **Reports not up to standard or delivered late will not validate the project.**

## Reference documents (provided by Liora)

- [Stock Portfolio Recommender — project brief](https://drive.google.com/file/d/1RNjZQYzrXEXiVZIOicOqD5EmOEz2BYfd/view?usp=drive_link).
- [Projects_methodology_reports](https://docs.google.com/document/d/1sbgOhiBA4hIYgkO-wrEDZrAejmoz9Ezr5EEwDqsdGMw/edit?usp=sharing) — guide for writing the different reports.
- [TEMPLATE - Data Audit](https://docs.google.com/spreadsheets/d/1JI7_DBcSXJl5UxB8VY-Ybyr3T1NdK0PCDkO1t-ZRjj8/edit?usp=sharing) — template for the Step 1 dataset analysis.
- [Defense_Methodology](https://docs.google.com/document/d/1bF9K4yBjaeWvBRdnNCIpwHDLqdZUHX1VRiEpQOQPY0A/edit?usp=sharing) — defense organization document.
- [Teaching Assistants — booking page](https://calendly.com/d/cmd6-vh4-6cd/teaching-assistant-cursus-ds?month=2025-05) — Calendly to book individual support sessions.

> Source: original [Slack kickoff message](https://dstinternatio-d5c8877.slack.com/archives/C0B2TU0HJM9/p1778517743660219) from Paul Grolier.

## Bibliography

- Aroussi, R. — _[yfinance documentation](https://ranaroussi.github.io/yfinance/)_.
- López de Prado, M. (2018). _Advances in Financial Machine Learning_. Wiley.

## Actions

- [x] **Decide the project's data source** — _Decided 2026-05-22 (mentor-approved): **Alpaca free IEX feed** for S&P 500 daily OHLCV; yfinance kept as fallback. DAX 40 deferred (not covered by Alpaca's free tier)._
- [ ] Migrate `fetch_data.py` from yfinance to the Alpaca API — by next meeting (**2026-05-28**).
- [ ] Set up a **Streamlit** project skeleton for presentation plots.
- [ ] Produce **5 initial visualizations** + fill the **Data Audit** Excel sheet — Deadline **2026-05-27**.
- [ ] Full data-exploration / DataViz / pre-processing **report (Rendering 1)** — Deadline **2026-06-03**.

> Mentor unavailable the first week of June 2026. Next meeting: **Thursday 2026-05-28, 09:00**.
