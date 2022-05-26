#%%
from abc import ABC

import pandas as pd
import polars as pl
from rich import print

#%%
df = pl.read_parquet(
    "data/raw/db_export_small.parquet"
)  # small only has date, text, id

#%%
def track_size(f: callable, *args, **kwargs) -> callable:
    def wrapper(df: pl.DataFrame, *args, **kwargs):
        print(f"Before {f.__name__}(...): n = {df.height:,}")
        result = f(df)
        return result

    return wrapper


class Cleaning(ABC):
    URL_REGEX = (
        r"[A-Za-z0-9]+://[A-Za-z0-9%-_]+(/[A-Za-z0-9%-_])*(#|\\?)[A-Za-z0-9%-_&=]*"
    )
    CASHTAG_REGEX = r"\$[a-zA-Z]+"
    HASHTAG_REGEX = r"\#\w+"
    MENTION_REGEX = r"@[A-Za-z0-9_]+"

    @staticmethod
    # @track_size
    def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
        return df.distinct()

    @staticmethod
    # @track_size
    def add_num_cashtags_col(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_cashtags = (
            df.select("text").to_pandas()["text"].str.count(Cleaning.CASHTAG_REGEX)
        )
        return df.with_column(pl.from_pandas(num_cashtags).alias("n_cashtags"))

    @staticmethod
    # @track_size
    def add_num_hashtags_col(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_hashtags = (
            df.select("text").to_pandas()["text"].str.count(Cleaning.HASHTAG_REGEX)
        )
        return df.with_column(pl.from_pandas(num_hashtags).alias("n_hashtags"))

    @staticmethod
    # @track_size
    def add_num_mentions_col(df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_hashtags = (
            df.select("text").to_pandas()["text"].str.count(Cleaning.MENTION_REGEX)
        )
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

    def remove_hyperlinks(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(pl.col("text").str.replace_all(Cleaning.URL_REGEX, ""))


#%%
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

#%%
sample = clean.sample(1000).transpose()

#%%
sample = Cleaning.remove_hyperlinks(clean.sample(1000)).transpose()

#%%
with open("outputs/dump/clean_sample.md", "w") as f:
    f.writelines(sample.transpose().to_pandas().to_markdown())
