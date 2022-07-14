#%%
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import set_style, Colors
from pyfin_sentiment.model import SentimentModel
import datetime

set_style()

#%%
df = pd.read_parquet("data/gme_casestudy/gme_tweets.parquet").convert_dtypes()
model = SentimentModel()

#%%
fig, ax = plt.subplots(figsize=(12, 5))
df.groupby(df["created_at"].dt.to_period("D"))["id"].count().sort_index().plot(ax=ax)


#%%
preds = model.predict(df["text"].to_list())

#%%
df = df.assign(sentiment=pd.Series(preds).map({"1": 1, "2": 0, "3": -1}))

#%%
fig, ax = plt.subplots(figsize=(12, 5))
df.groupby(df["created_at"].dt.to_period("H"))["sentiment"].mean().rolling(
    24
).mean().sort_index().plot(ax=ax)

#%%
senti_pct = (
    pd.pivot_table(
        data=df,
        index=df["created_at"].dt.to_period("H"),
        columns="sentiment",
        aggfunc="count",
        values="id",
    )
    .apply(lambda r: r / r.sum(), axis=1)
    .fillna(0)
    .rename({-1: "neg", 0: "neu", 1: "pos"}, axis=1)
    .assign(delta=lambda d: d.eval("pos - neg"))
)

#%%
fig, axes = plt.subplots(
    2, 1, figsize=(16, 10), sharey=True, gridspec_kw={"height_ratios": [2, 1]}
)
smoothed_data = senti_pct.rolling(24).mean().iloc[25:]


axes[0].plot(
    smoothed_data.index.to_timestamp(), smoothed_data["pos"], color="green", zorder=10
)
axes[0].plot(
    smoothed_data.index.to_timestamp(), smoothed_data["neg"], color="red", zorder=10
)
axes[1].plot(
    smoothed_data.index.to_timestamp(),
    smoothed_data["delta"],
    color=Colors.DARKBLUE.value,
    zorder=10,
)

axes[0].fill_between(
    x=smoothed_data.index.to_timestamp(),
    y1=smoothed_data["pos"],
    y2=smoothed_data["neg"],
    hatch="///",
    edgecolor="0.5",
    facecolor="none",
    lw=0,
    zorder=5,
)


for ax_idx in (0, 1):
    axes[ax_idx].xaxis.set_major_locator(mdates.WeekdayLocator(0))
    axes[ax_idx].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[ax_idx].tick_params("x", length=5, which="major")
    date_fmt = mdates.DateFormatter("%b-%d")
    axes[ax_idx].xaxis.set_major_formatter(date_fmt)
    axes[ax_idx].yaxis.set_major_formatter("{x:.0%}")


to_highlight = [datetime.datetime(2021, 1, 14), datetime.datetime(2021, 2, 3)]
for ts in to_highlight:
    axes[0].axvline(x=ts, color="k", alpha=0.5, zorder=-1, ls="--", ymax=0.95)
    axes[1].axvline(x=ts, color="k", alpha=0.5, zorder=-1, ls="--", ymax=0.95)

    axes[0].text(s=f"{ts:%b-%d}", x=ts + datetime.timedelta(hours=10), y=0.52)

# pos + neg labels
axes[0].text(
    s="positive",
    x=smoothed_data.index.to_timestamp().max() + datetime.timedelta(hours=10),
    y=smoothed_data["pos"].iloc[-1],
    color="green",
    va="center",
    weight="bold",
)
axes[0].text(
    s="negative",
    x=smoothed_data.index.to_timestamp().max() + datetime.timedelta(hours=10),
    y=smoothed_data["neg"].iloc[-1],
    color="red",
    va="center",
    weight="bold",
)

# titles
axes[0].set_title("Positive and Negative Tweets (%)", weight="bold")
axes[1].set_title("Positive Tweets (%) - Negative Tweets (%)", weight="bold")


plt.tight_layout()
sns.despine()
fig.savefig("outputs/plots/gme_sentiment.pdf", bbox_inches="tight")

#%%
import yfinance as yf

stock_price = yf.Ticker("GME").history(
    start="2021-01-06", end="2021-02-28", interval="1h"
)["Close"]

#%%
fig, ax = plt.subplots(figsize=(16, 5))
stock_price.rolling(24).mean().plot()

#%%
