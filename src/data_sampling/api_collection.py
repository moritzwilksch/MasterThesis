import os
import time

import requests
from dotenv import load_dotenv

from src.utils.db import client as DB
from src.utils.log import log
from src.utils.twitter_api import TwitterAPI

load_dotenv(".env")


class TwitterAPICollector:
    def __init__(self, params: dict) -> None:
        self.api = TwitterAPI()
        self.params = params

    def collect(self):
        next_token = None
        while True:
            response = self.api.search_request(
                params=self.params, next_token=next_token
            )
            data_to_write = response.json().get("data")
            next_token = response.json().get("meta").get("next_token")

            # rate limited
            if response.status_code == 429:
                log.warn(f"Got a 429. Sleeping for 2 min...")
                time.sleep(120)
                continue

            # error
            if not response.ok:
                log.warn("Response not ok!")
                DB.thesis.prod_log.insert_one(response.json())

            # no more data = finish
            if next_token is None:
                log.info(f"Did not receive next_token. Stopping.")
                break

            if data_to_write is None:
                log.warn("No data to write!")
            else:
                # persist to DB
                DB.thesis.prod_tweet.insert_many(data_to_write)
                log.info(f"Saved {len(data_to_write)} tweets.")

            time.sleep(3.5)


params = {
    "start_time": "2021-04-01T00:00:00Z",
    "end_time": "2021-05-01T00:00:00Z",
    "max_results": 10,
}

collector = TwitterAPICollector(params=params)
collector.collect()
