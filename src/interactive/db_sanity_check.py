#%%
import datetime

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import pyarrow
import seaborn as sns
from pymongoarrow.api import Schema

# from pymongoarrow.monkey import patch_all,
from pymongoarrow.api import find_arrow_all

from src.utils.db import client as DB



schema = Schema({"created_at": pyarrow.string(), "text": pyarrow.string()})
data = find_arrow_all(
    DB.thesis.prod_tweet,
    query={},
    projection={"_id": False, "created_at": True, "text": True},
    schema=schema,
)

df = pl.from_arrow(data)


#%%
n_per_day = (
    df.with_column(pl.col("created_at").str.strptime(pl.Datetime))
    .groupby(pl.col("created_at").dt.strftime("%Y-%m-%d"))
    .agg(pl.col("text").count())
    .sort("created_at")
    .with_column(pl.col("created_at").str.strptime(pl.Date))
    .filter(pl.col("created_at") > datetime.date(2021, 11, 30))
)


fig, ax = plt.subplots(figsize=(15, 8))
sns.lineplot(data=n_per_day.to_pandas(), x="created_at", y="text", ax=ax)
