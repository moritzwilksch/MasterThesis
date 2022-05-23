import os
import time
from urllib import response

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import toml
from dotenv import load_dotenv

from src.utils.log import log

load_dotenv()


class TwitterAPI:
    def __init__(self, config_file: str = "config/api_collection.toml") -> None:
        self.oauth_url = "https://api.twitter.com/oauth2/token"
        self.count_url = "https://api.twitter.com/2/tweets/counts/all"
        self.search_url = "https://api.twitter.com/2/tweets/search/all"
        self.bearer = f"Bearer {self.get_bearer()}"

        self.config = toml.load(config_file)
        self.query = f"{self.config['raw_query']} {self.config['query_options']}"

        log.debug(f"{self.query =}")
        log.info(f"Query contains {self.query.count('$')} tickers.")

    def get_bearer(self):
        querystring = {"grant_type": "client_credentials"}
        response = requests.post(
            self.oauth_url,
            auth=(os.getenv("TWITTER_API_KEY"), os.getenv("TWITTER_API_SECRET")),
            params=querystring,
        )
        return response.json().get("access_token")

    def get_tweet_count(
        self,
        query: str,
        granularity: str = "day",
        query_modifiers: str = "lang:en -is:retweet has:cashtags -is:nullcast -has:images -has:videos",  # -is:nullcast removes ads
        **kwargs,
    ) -> dict:
        querystring = {
            "query": f"{query} {query_modifiers}",
            "granularity": granularity,
        }
        headers = {"Authorization": self.bearer}

        response = requests.get(
            self.count_url, headers=headers, params=(querystring | kwargs)
        )
        return response.json()

    def search_request(
        self, params: dict = None, next_token: str = None
    ) -> requests.Response:
        params["query"] = self.query
        params["tweet.fields"] = "created_at,text,id,author_id,public_metrics,entities"
        if next_token is not None:
            params["next_token"] = next_token

        # response = joblib.load("outputs/stub_response.joblib")
        response = requests.get(
            self.search_url, params=params, headers={"Authorization": self.bearer}
        )
        # joblib.dump(response, "outputs/stub_response.joblib")
        # log.info("saved")

        return response
