#%%
import polars as pl
import spacy

from src.utils.preprocessing import Preprocessor

#%%
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")

prepper = Preprocessor()
df = prepper.process(df)

df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)
#%%
from textaugment import EDA

t = EDA()

#%%
augmented = df.select(
    [pl.col("text").apply(lambda x: t.synonym_replacement(x)), pl.col("label")]
)

#%%
