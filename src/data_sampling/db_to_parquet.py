#%%
import polars as pl
import pyarrow
from pymongoarrow.api import Schema

from pymongoarrow.api import find_arrow_all

from src.utils.db import client as DB



schema = Schema({"id": pyarrow.string(), "created_at": pyarrow.string(), "text": pyarrow.string()})
data = find_arrow_all(
    DB.thesis.prod_tweet,
    query={},
    projection={"_id": False, "created_at": True, "text": True, "id": True},
    schema=schema,
)

#%%
df = pl.from_arrow(data)

#%%
df.to_parquet("data/raw/db_export_small.parquet")