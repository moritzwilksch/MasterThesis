#%%
from cProfile import label

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.plotting import Colors, set_style

set_style()

#%%
df = pd.read_parquet("data/labeled/labeled_tweets.parquet")
df = df.assign(label=lambda d: d["label"].replace("0", "2"))
finsome = pd.read_csv("data/finSoMe/finsome.csv")
#%%
dist = df["label"].value_counts(normalize=True).sort_index()
dist_finsome = finsome["market_sentiment"].value_counts(normalize=True).sort_index()
dist_finsome

#%%
fig, ax = plt.subplots(figsize=(12, 3))

plot_df = df["label"].value_counts(normalize=True).sort_index()
plot_df_finsome = (
    finsome["market_sentiment"]
    .to_frame()
    .rename({"market_sentiment": "label"}, axis=1)
    .assign(
        label=lambda d: d["label"].map({"Bullish": "1", "Bearish": "3", "Unsure": "2"})
    )["label"]
    .value_counts(normalize=True)
).sort_index()
height = 0.35
ax.barh(
    y=np.arange(3) + 0.19,
    width=plot_df,
    height=height,
    color=Colors.DARKBLUE.value,
    label="pyFin-Sentiment",
    # hatch="\\\\",
    # ec=(1, 1, 1, 0.3),
)
ax.barh(
    y=np.arange(3) - 0.19,
    width=plot_df_finsome,
    color="0.75",
    height=height,
    label="FinSoMe",
    # hatch="//",
    # ec=(0, 0, 0, 0.3),
)
sns.despine(left=True)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.set(
    yticks=np.arange(3),
)
ax.set_yticklabels(["Bullish", "Neutral/\nUnsure", "Bearish"], weight="bold")
ax.yaxis.set_tick_params(length=0, pad=20)
ax.set_xlabel("Proportion of Tweets per Category", weight="bold")
ax.legend()
fig.savefig(
    "outputs/plots/class_distributions.pdf", bbox_inches="tight",
)
