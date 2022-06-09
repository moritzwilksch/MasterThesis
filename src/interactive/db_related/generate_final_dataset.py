""" Generates final dataset as .parquet from DB """

#%%
import polars as pl

from src.utils.db import get_client

DB = get_client()

df = pl.from_dicts(
    list(
        DB.thesis.labeled_tweets.find(
            {"label": {"$ne": ""}},
            projection={"text": True, "label": True, "_id": False, "id": True},
        )
    )
)

# df.to_parquet("data/labeled/labeled_tweets.parquet")
