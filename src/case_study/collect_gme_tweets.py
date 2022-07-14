#%%
from src.data_sampling.api_collection import TwitterAPI, TwitterAPICollector

#%%
params = {
    "start_time": "2021-01-01T00:00:00Z",  # oldest time
    "end_time": "2021-03-01T00:00:00Z",  # newest, most recent time
    "max_results": 500,
}
collector = TwitterAPICollector(params, config_file="config/gme_collection.toml")
# collector.collect()

#%%
from src.utils.db import get_client

DB = get_client()

#%%
import polars as pl

results = DB.thesis.gme_tweets.find(
    {}, {"id": True, "created_at": True, "text": True, "_id": False}
)
df = pl.from_dicts(list(results)).with_column(
    pl.col("created_at").str.strptime(pl.Datetime)
)
#%%
df.to_parquet("data/raw/gme_tweets.parquet")
