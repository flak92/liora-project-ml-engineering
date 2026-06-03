"""Regenerate the six EDA figures for the report from the local Alpaca CSVs,
and print the verified statistics used in the report captions/commentary.

Run:  ../.venv/bin/python make_figures.py
Figures are written to reports/figures/*.png ; stats are printed as JSON.
Every number quoted in REPORT.md §3-§4 comes from this script's output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import plots as P  # noqa: E402
from src.data_loader import load_prices, load_tickers  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
DPI = 150
OUT: dict = {}


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return name


tickers = load_tickers()
prices = load_prices()  # adds daily_return from adj_close
ret = prices["daily_return"].dropna()

# ---- Fig 1: sector count + chi-square goodness-of-fit vs uniform -------------
save(P.plot_sector_count(tickers), "01_sector_count.png")
sector_counts = tickers["sector"].value_counts()
chi2, chi2_p = sps.chisquare(sector_counts.values)
OUT["sector"] = {
    "counts": sector_counts.to_dict(),
    "n_sectors": int(sector_counts.size),
    "chi2": round(float(chi2), 1),
    "chi2_p": float(chi2_p),
    "expected_uniform": round(len(tickers) / sector_counts.size, 1),
}

# ---- Fig 2: mean daily volume per stock + log-normality ----------------------
save(P.plot_volume_box(prices), "02_volume_box.png")
mean_vol = prices.groupby("ticker")["volume"].mean()
log_vol = np.log(mean_vol[mean_vol > 0])
jb_v, jb_v_p = sps.jarque_bera(log_vol)
OUT["volume"] = {
    "median": round(float(mean_vol.median()), 0),
    "mean": round(float(mean_vol.mean()), 0),
    "min": round(float(mean_vol.min()), 0),
    "max": round(float(mean_vol.max()), 0),
    "logvol_jarque_bera": round(float(jb_v), 1),
    "logvol_jb_p": float(jb_v_p),
    "logvol_skew": round(float(sps.skew(log_vol)), 3),
}

# ---- Fig 3: daily return distribution + Jarque-Bera --------------------------
save(P.plot_return_hist(prices), "03_return_hist.png")
jb, jb_p = sps.jarque_bera(ret)
OUT["returns"] = {
    "n": int(ret.size),
    "mean_pct": round(float(ret.mean() * 100), 4),
    "median_pct": round(float(ret.median() * 100), 4),
    "std_pct": round(float(ret.std() * 100), 3),
    "skew": round(float(sps.skew(ret)), 3),
    "excess_kurtosis": round(float(sps.kurtosis(ret)), 1),
    "jarque_bera": round(float(jb), 1),
    "jb_p": float(jb_p),
}

# ---- Fig 4: anchor tickers, rebased to 100 (ragged-history visual) -----------
anchors = ["AAPL", "MSFT", "JPM", "XOM", "SNDK"]
fig, ax = plt.subplots(figsize=(10, 5))
anchor_info = {}
for tk in anchors:
    d = prices.loc[prices["ticker"] == tk, ["date", "adj_close"]].dropna().sort_values("date")
    if d.empty:
        continue
    rebased = d["adj_close"] / d["adj_close"].iloc[0] * 100
    ax.plot(d["date"], rebased, linewidth=1.4, label=tk)
    anchor_info[tk] = {
        "first": d["date"].iloc[0].strftime("%Y-%m-%d"),
        "last": d["date"].iloc[-1].strftime("%Y-%m-%d"),
        "n_days": int(len(d)),
        "total_return_pct": round(float(d["adj_close"].iloc[-1] / d["adj_close"].iloc[0] * 100 - 100), 0),
    }
ax.set_yscale("log")
ax.set_title("Anchor tickers — adjusted close rebased to 100 at each ticker's first bar (log scale)")
ax.set_xlabel("Trading date")
ax.set_ylabel("Indexed price (first bar = 100, log scale)")
ax.grid(True, which="both", alpha=0.25, linestyle="--")
ax.legend(loc="upper left", ncol=5)
fig.tight_layout()
save(fig, "04_anchor_lines.png")
OUT["anchors"] = anchor_info

# ---- Fig 5: correlation heatmap (top-10 by volume) + off-diagonal means ------
save(P.plot_correlation_heatmap(prices), "05_corr_heatmap.png")
top10 = prices.groupby("ticker")["volume"].mean().nlargest(10).index
wide = prices[prices["ticker"].isin(top10)].pivot(index="date", columns="ticker", values="daily_return").dropna()
pear = wide.corr().values
spear = wide.corr(method="spearman").values
off = ~np.eye(pear.shape[0], dtype=bool)
OUT["corr"] = {
    "top10_tickers": list(top10),
    "pearson_offdiag_mean": round(float(pear[off].mean()), 3),
    "pearson_offdiag_min": round(float(pear[off].min()), 3),
    "pearson_offdiag_max": round(float(pear[off].max()), 3),
    "spearman_offdiag_mean": round(float(spear[off].mean()), 3),
    "n_obs": int(wide.shape[0]),
}

# ---- Fig 6: risk/return scatter with SNDK annotated + Pearson(vol,ret) -------
st = prices.groupby("ticker")["daily_return"].agg(["mean", "std"]).dropna()
st["annual_return"] = st["mean"] * 252 * 100
st["annual_volatility"] = st["std"] * np.sqrt(252) * 100
fig, ax = plt.subplots(figsize=(9, 6))
import seaborn as sns  # noqa: E402

sns.set()
sns.scatterplot(data=st, x="annual_volatility", y="annual_return", alpha=0.55, edgecolor=None,
                color="steelblue", ax=ax)
mv, mr = st["annual_volatility"].median(), st["annual_return"].median()
ax.axvline(mv, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(mr, color="gray", linestyle="--", linewidth=1, alpha=0.7)
if "SNDK" in st.index:
    s = st.loc["SNDK"]
    ax.scatter([s["annual_volatility"]], [s["annual_return"]], color="darkred", s=60, zorder=5)
    ax.annotate("SNDK", (s["annual_volatility"], s["annual_return"]),
                textcoords="offset points", xytext=(-38, -4), color="darkred", fontweight="bold")
ax.set_xlabel("Annualized volatility (risk) %")
ax.set_ylabel("Annualized expected return %")
ax.set_title("Risk vs. return — all S&P 500 stocks (SNDK highlighted)")
fig.tight_layout()
save(fig, "06_risk_return.png")
pear_vr = sps.pearsonr(st["annual_volatility"], st["annual_return"])
OUT["risk_return"] = {
    "n_stocks": int(len(st)),
    "pearson_vol_ret_r": round(float(pear_vr[0]), 3),
    "pearson_vol_ret_p": float(pear_vr[1]),
    "median_vol": round(float(mv), 1),
    "median_ret": round(float(mr), 1),
    "sndk": {
        "annual_return": round(float(st.loc["SNDK", "annual_return"]), 0) if "SNDK" in st.index else None,
        "annual_volatility": round(float(st.loc["SNDK", "annual_volatility"]), 0) if "SNDK" in st.index else None,
    },
}

print(json.dumps(OUT, indent=2))
print("\nFIGURES:", sorted(p.name for p in FIG.glob("*.png")))
