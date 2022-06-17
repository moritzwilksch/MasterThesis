#%%
import joblib
import pandas as pd
from src.utils.preprocessing import Preprocessor
import polars as pl

#%%
model = joblib.load("outputs/models/final_LogisticRegressionModel.gz")

#%%
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
df = prepper.process(df)


# TODO: should we merge labels? drop 2?
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)


df = df.to_pandas()

#%%
df["prediction"] = model.predict(df["text"])

#%%
wrong = df.query("(label=='3' & prediction=='1') | (label=='1' & prediction=='3')")
wrong
