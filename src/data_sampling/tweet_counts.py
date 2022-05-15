import time
from xml.etree.ElementPath import get_parent_map

import pandas as pd
from rich import print
from rich.progress import track

from src.utils.db import client
from src.utils.logging import log
from src.utils.twitter_api import TwitterAPI

coll = client["thesis"]["tweet_counts"]


def get_spy_ticker_list() -> list:
    """Fetches list of all S&P500 tickers from wikipedia. Saves to disk."""
    spy = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    tickers = spy["Symbol"].to_list()

    with open("data/raw/spy_tickes.txt", "w") as f:
        f.writelines("\n".join(tickers))
    return tickers


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
