#%%
import time
from turtle import width

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import seaborn as sns
from rich.progress import track

from src.utils.db import client
from src.utils.log import log
from src.utils.plotting import set_style
from src.utils.twitter_api import TwitterAPI

coll = client["thesis"]["tweet_counts"]
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
res = list(coll.find({}, {"_id": 0}))
# counts = pd.DataFrame(res)
counts = pl.DataFrame(res)

#%%
plotdf = (
    counts.with_column((pl.col("total_tweet_count") / 30).alias("avg_daily"))
    .filter(pl.col("avg_daily") >= 100)
    .sort("avg_daily", reverse=True)
)
print(f"Matching {plotdf.height} tickers")
fig, ax = plt.subplots(figsize=(9, 11))
sns.barplot(
    data=plotdf.to_pandas(),
    orient="h",
    x="avg_daily",
    y="ticker",
    ax=ax,
    color="k",
    dodge=False,
)

ax.axvline(x=100, ls="--", color="white")

ax.set_xlabel("Average number of tweets per day", weight="bold")
ax.set_ylabel("Ticker", weight="bold")

sns.despine()
plt.savefig(
    "outputs/plots/tweet_counts.png", dpi=300, bbox_inches="tight", facecolor="white"
)


#%%
query = " OR ".join(f"${t}" for t in plotdf["ticker"])
query = f"({query})"
#%%
TwitterAPI().get_tweet_count(query)
