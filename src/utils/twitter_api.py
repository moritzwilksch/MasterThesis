import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()


class TwitterAPI:
    def __init__(self) -> None:
        self.oauth_url = "https://api.twitter.com/oauth2/token"
        self.count_url = "https://api.twitter.com/2/tweets/counts/all"
        self.bearer = self.get_bearer()

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
        headers = {"Authorization": f"Bearer {self.bearer}"}

        response = requests.get(
            self.count_url, headers=headers, params=(querystring | kwargs)
        )
        return response.json()
