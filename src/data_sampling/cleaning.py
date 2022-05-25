#%%
from abc import ABC
import polars as pl
from rich import print
import pandas as pd

#%%
df = pl.read_parquet("data/raw/db_export_small.parquet")  # small only has date, text, id

#%%
def track_size(f: callable, *args, **kwargs) -> callable:
    def wrapper(df: pl.DataFrame, *args, **kwargs):
        print(f"Before {f.__name__}(...): n = {df.height:,}")
        result = f(df)
        return result

    return wrapper


class Cleaning(ABC):
    @staticmethod
    # @track_size
    def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
        return df.distinct()

    @staticmethod
    # @track_size
    def add_num_cashtags_col(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_cashtags = df.select("text").to_pandas()["text"].str.count(r"\$[a-zA-Z]+")
        return df.with_column(pl.from_pandas(num_cashtags).alias("n_cashtags"))

    @staticmethod
    # @track_size
    def add_num_hashtags_col(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_hashtags = df.select("text").to_pandas()["text"].str.count(r"\#\w+")
        return df.with_column(pl.from_pandas(num_hashtags).alias("n_hashtags"))

    @staticmethod
    # @track_size
    def add_num_mentions_col(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_hashtags = df.select("text").to_pandas()["text"].str.count(r"@[A-Za-z0-9_]+")
        return df.with_column(pl.from_pandas(num_hashtags).alias("n_mentions"))

    @staticmethod
    # @track_size
    def add_num_words_col(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col("text").str.split(" ").arr.lengths().alias("n_words")
        )

    @staticmethod
    # @track_size
    def filter_num_cashtags_col(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("n_cashtags") <= 4)

    @staticmethod
    # @track_size
    def remove_spam(df: pl.DataFrame) -> pl.DataFrame:
        # more tags than words
        df = df.filter(
            (pl.col("n_cashtags") / pl.col("n_words") <= 0.5)
            | (pl.col("n_hashtags") / pl.col("n_words") <= 0.5)
        )

        return df


clean: pl.DataFrame = (
    df.pipe(Cleaning.drop_duplicates)
    .pipe(Cleaning.add_num_cashtags_col)
    .pipe(Cleaning.add_num_hashtags_col)
    .pipe(Cleaning.add_num_mentions_col)
    .pipe(Cleaning.add_num_words_col)
    .pipe(Cleaning.filter_num_cashtags_col)
    .pipe(Cleaning.remove_spam)
)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import set_style

set_style()
fig, ax = plt.subplots(figsize=(15, 6))
sns.histplot(clean.select("n_hashtags").to_numpy().ravel(), binwidth=1)

#%%
# clean.filter((pl.col("n_hashtags") > 5) & (pl.col("n_hashtags") < 10))
clean
