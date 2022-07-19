#%%
import time
from abc import ABC

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from rich.progress import track

from src.utils.db import get_client

DB = get_client()
from rich import print

from src.utils.log import log
from src.utils.plotting import (Colors, scale_lightness, set_style,
                                when_then_else)
from src.utils.twitter_api import TwitterAPI

coll = DB["thesis"]["tweet_counts"]
set_style()


def get_spy_ticker_list() -> list:
    """Fetches list of all S&P500 tickers from wikipedia. Saves to disk."""
    spy = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = spy["Symbol"].to_list()

    with open("data/raw/spy_tickes.txt", "w") as f:
        f.writelines("\n".join(tickers))
    return tickers


def scrape_to_db():
    spy = get_spy_ticker_list()

    api = TwitterAPI()
    for ticker in track(spy):
        try:
            counts = api.get_tweet_count(
                f"${ticker}",
                start_time="2022-04-01T00:00:00Z",
                end_time="2022-05-01T00:00:00Z",
            )

            doc = {
                "ticker": ticker,
                "total_tweet_count": counts.get("meta").get("total_tweet_count"),
            }

            log.debug(f"Got ${ticker}")
        except:
            doc = {"ticker": ticker, "total_tweet_count": None}
            log.warning(f"Failed getting ${ticker}")

        finally:
            coll.insert_one(doc)
            time.sleep(4)

    log.info("Done!")


#%%
class Plots(ABC):
    @classmethod
    def volume_by_top_n(cls, n: int):
        top_pct = (
            (
                counts.sort("total_tweet_count")
                .tail(n)
                .select(pl.col("total_tweet_count").cast(pl.Float32))
                .sum()
                / counts.select("total_tweet_count").sum()
            )
            .to_numpy()
            .ravel()[0]
        )

        print(f"Top {n} tickers created {top_pct:.1%} of tweet volume.")

    @classmethod
    def top_tickers_tweet_count(cls, counts: pl.DataFrame, save: bool = False):
        excluded = ["AME", "OGN", "TEL", "AMP", "KEY", "STX"]

        cls.volume_by_top_n(20)
        cls.volume_by_top_n(56)

        plotdf = (
            counts.with_column((pl.col("total_tweet_count") / 30).alias("avg_daily"))
            .filter(pl.col("avg_daily") >= 100)
            .sort("avg_daily", reverse=False)
            # .tail(20)
            .with_columns(
                [
                    when_then_else(
                        pl.col("ticker").is_in(excluded), pl.lit(1), pl.lit(0)
                    ).alias("linewidth"),
                    when_then_else(
                        pl.col("ticker").is_in(excluded),
                        pl.lit(
                            scale_lightness(
                                sns.desaturate(Colors.DARKBLUE.value, 0.1), 4.1
                            )
                        ),
                        pl.lit(Colors.DARKBLUE.value),
                    ).alias("color"),
                ]
            )
        )

        print(f"Matching {plotdf.height} tickers")

        fig, ax = plt.subplots(figsize=(15, 7))
        plotdf_top20 = plotdf.tail(20)

        ax.barh(
            y=plotdf_top20["ticker"],
            width=plotdf_top20["avg_daily"],
            height=0.7,
            color=plotdf_top20["color"],
            linestyle="--",
            linewidth=plotdf_top20["linewidth"],
            ec=Colors.DARKBLUE.value,
            zorder=10,
        )

        for ticker, p in zip(plotdf_top20["ticker"], ax.patches):
            if ticker in excluded:
                p.set_label("Excluded: cryptocurrency")
                break

        ax.grid(True, "major", axis="x", zorder=-10, ls="--")
        ax.set_xlabel("Average number of tweets per day", weight="bold")
        ax.legend(framealpha=1)
        ax.margins(y=0.01)
        sns.despine(left=True)

        if save:
            plt.savefig(
                "outputs/plots/tweet_counts.pdf",
                # dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()


counts = pl.DataFrame(list(coll.find({}, {"_id": 0})))
Plots.top_tickers_tweet_count(counts, save=True)
