#%%
import polars as pl
from src.utils.db import get_client

DB = get_client()["thesis"]["labeled_tweets"]

#%%
df = pl.from_dicts(list(DB.find({"label": {"$ne": ""}}, projection={"_id": False})))
#%%
diffs = (
    df.sort(by="labeled_at").select(pl.col("labeled_at").diff().dt.seconds()).drop_nulls()
)

#%%
diffs.select(pl.col("labeled_at").median())

#%%
from datetime import datetime

df = pl.DataFrame(
    {
        "foo": [
            datetime(2022, 5, 5, 12, 31, 34),
            datetime(2022, 5, 5, 12, 47, 1),
            datetime(2022, 5, 6, 8, 59, 11),
        ]
    }
)
diffs = df.select(pl.col("foo").diff().dt.seconds().cast(pl.Float32).drop_nulls())

# diffs.median()  # fails: called `Result::unwrap()` on an `Err` value: OutOfSpec("validity mask length must match the number of values")
# diffs.select(pl.col("foo").cast(pl.Float32).median())  # works


