#%%
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.preprocessing import Preprocessor
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


def plot_class_distribution():
    fig, ax = plt.subplots(figsize=(12, 3))

    plot_df = df["label"].value_counts(normalize=True).sort_index()
    plot_df_finsome = (
        finsome["market_sentiment"]
        .to_frame()
        .rename({"market_sentiment": "label"}, axis=1)
        .assign(
            label=lambda d: d["label"].map(
                {"Bullish": "1", "Bearish": "3", "Unsure": "2"}
            )
        )["label"]
        .value_counts(normalize=True)
    ).sort_index()
    height = 0.35
    ax.barh(
        y=np.arange(3) + 0.19,
        width=plot_df,
        height=height,
        color=Colors.DARKBLUE.value,
        label="pyFin",
        # hatch="\\\\",
        # ec=(1, 1, 1, 0.3),
    )
    ax.barh(
        y=np.arange(3) - 0.19,
        width=plot_df_finsome,
        color="0.75",
        height=height,
        label="Fin-SoMe",
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
        "outputs/plots/class_distributions.pdf",
        bbox_inches="tight",
    )


plot_class_distribution()

#%%
def plot_doc_length(df, finsome):
    pyfin = pl.Series("text", df["text"]).to_frame()
    finsome = pl.Series("text", finsome["tweet"]).to_frame()

    pyfin_nwords = pyfin.select(pl.col("text").str.split(" ").arr.lengths())
    finsome_nwords = finsome.select(pl.col("text").str.split(" ").arr.lengths())

    plot_df = pd.DataFrame(
        {
            "words": pl.concat([pyfin_nwords, finsome_nwords]).to_numpy().ravel(),
            "Dataset": ["pyFin"] * pyfin.height + ["Fin-SoMe"] * finsome.height,
        }
    )

    plot_df["Dataset"] = plot_df["Dataset"].astype("category")

    fig, ax = plt.subplots(figsize=(17, 6))

    sns.histplot(
        data=plot_df.query("Dataset == 'Fin-SoMe'"),
        x="words",
        hue="Dataset",
        ax=ax,
        binwidth=1,
        palette=[Colors.GREEN.value, Colors.DARKBLUE.value],
        alpha=0.6,
        edgecolor=Colors.GREEN.value,
    )

    sns.histplot(
        data=plot_df.query("Dataset == 'pyFin'"),
        x="words",
        hue="Dataset",
        ax=ax,
        binwidth=1,
        palette=[Colors.GREEN.value, Colors.DARKBLUE.value],
        alpha=0.6,
        edgecolor=Colors.DARKBLUE.value,
    )

    sns.despine()

    ax.set_xlim(-1, 70)
    ax.set_xlabel("Number of Words in a Post", weight="bold")
    ax.set_ylabel("Count", weight="bold")

    fig.savefig(
        "outputs/plots/word_count_distribution.pdf",
        bbox_inches="tight",
    )


plot_doc_length(df, finsome)

#%%



