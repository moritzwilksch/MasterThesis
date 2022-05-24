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

            # rate limited
            if response.status_code == 429:
                log.warn(f"Got a 429. Sleeping for 2 min...")
                time.sleep(120)
                continue

            # error
            if not response.ok:
                log.warn("Response not ok!")
                DB.thesis.prod_log.insert_one(response.json())

            # cast to json and extract important parameters
            response_as_json = response.json()
            try:
                data_to_write = response_as_json.get("data")
                next_token = response_as_json.get("meta").get("next_token")
                newest_id = response_as_json.get("meta").get("newest_id")
            except:
                log.warn(
                    "Error extracting data, next_token, or newest_id from response."
                )
                DB.thesis.prod_log.insert_one(response_as_json)
                log.info("Logged problematic response to db")
                continue

            # no more data = finish
            if next_token is None:
                log.info(f"Did not receive next_token. Stopping.")
                break

            if (
                data_to_write is None
            ):  # just save guards the DB write which fails on empty data
                log.warn("No data to write!")
            else:
                # persist to DB
                DB.thesis.prod_tweet.insert_many(data_to_write)
                log.info(f"Saved {len(data_to_write)} tweets. newest_id = {newest_id}")

            time.sleep(3.5)


params = {
    "start_time": "2022-02-01T00:00:00Z",  # oldest time
    "end_time": "2022-03-01T00:00:00Z",  # newest, most recent time
    "max_results": 500,
}

collector = TwitterAPICollector(params=params)
collector.collect()
