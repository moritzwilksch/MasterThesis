import pandas as pd
import polars as pl

from src.utils.db import client as DB
from src.utils.log import log
from src.utils.storage import bucket

data = list(DB.thesis.prod_tweet.find(filter={}, projection={"_id": False}))
log.info(f"Found {len(data)} DB entries.")

# read data into pandas df, normalize and cast objs to str
df = pd.json_normalize(data)
obj_cols = df.select_dtypes("object").columns
df[obj_cols] = df[obj_cols].astype("string")
df = pl.from_pandas(df)

# save locally and upload
df.to_parquet("data/raw/db.bak.parquet")
log.info("Saved to parquet.")
bucket.upload_file("data/raw/db.bak.parquet", "data/raw/db.bak.parquet")
log.info(f"Uploaded to S3.")
