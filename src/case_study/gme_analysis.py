#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import set_style, Colors
from pyfin_sentiment.model import SentimentModel

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
import matplotlib.dates as mdates
import matplotlib.ticker as ticker


fig, axes = plt.subplots(
    2, 1, figsize=(16, 9), sharey=True, gridspec_kw={"height_ratios": [2, 1]}
)
smoothed_data = senti_pct.rolling(24).mean()


axes[0].plot(smoothed_data.index.to_timestamp(), smoothed_data["pos"], color="green")
axes[0].plot(smoothed_data.index.to_timestamp(), smoothed_data["neg"], color="red")
axes[1].plot(smoothed_data.index.to_timestamp(), smoothed_data["delta"], color=Colors.DARKBLUE.value)

axes[0].fill_between(
    x=smoothed_data.index.to_timestamp(),
    y1=smoothed_data["pos"],
    y2=smoothed_data["neg"],
    hatch="///",
    alpha=0.25,
    facecolor="none",
    lw=0
)



for ax_idx in (0, 1):
    axes[ax_idx].xaxis.set_major_locator(mdates.WeekdayLocator(0))
    axes[ax_idx].xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes[ax_idx].tick_params("x", length=5, which="major")
    date_fmt = mdates.DateFormatter("%b-%d")
    axes[ax_idx].xaxis.set_major_formatter(date_fmt)
    axes[ax_idx].yaxis.set_major_formatter("{x:.0%}")

plt.tight_layout()
sns.despine()
