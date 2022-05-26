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
    def wrapper(cls, df: pl.DataFrame, *args, **kwargs):
        print(f"Before {f.__name__}(...): n = {df.height:,}")
        result = f(cls, df)
        return result

    return wrapper


class Cleaning(ABC):
    URL_REGEX = (
        r"[A-Za-z0-9]+://[A-Za-z0-9%-_]+(/[A-Za-z0-9%-_])*(#|\\?)[A-Za-z0-9%-_&=]*"
    )
    CASHTAG_REGEX = r"\$[a-zA-Z]+"
    HASHTAG_REGEX = r"\#\w+"
    MENTION_REGEX = r"@[A-Za-z0-9_]+"
    CRYPTO_TERMS = [
        "bitcoin",
        "etherium",
        "btc",
        "eth",
        "nft",
        "token",
        "wallet",
        "web3",
        "airdrop",
        "wagmi",
        "solana",
        "opensea",
        "cryptopunks",
        "uniswap",
        "lunar",
        "hodl",
        "binance",
        "coinbase",
        "cryptocom",
        "doge",
    ]

    @classmethod
    @track_size
    def drop_duplicates(cls, df: pl.DataFrame) -> pl.DataFrame:
        df = df.distinct()
        return df.filter(~(pl.col("text").is_duplicated() & (pl.col("n_words") > 5)))

    @classmethod
    @track_size
    def add_num_cashtags_col(cls, df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_cashtags = (
            df.select("text").to_pandas()["text"].str.count(cls.CASHTAG_REGEX)
        )
        return df.with_column(pl.from_pandas(num_cashtags).alias("n_cashtags"))

    @classmethod
    @track_size
    def add_num_hashtags_col(cls, df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_hashtags = (
            df.select("text").to_pandas()["text"].str.count(cls.HASHTAG_REGEX)
        )
        return df.with_column(pl.from_pandas(num_hashtags).alias("n_hashtags"))

    @classmethod
    @track_size
    def add_num_mentions_col(cls, df: pl.DataFrame) -> pl.DataFrame:
        # TODO: fix this as soon as polars implements .str.extract_all()
        num_hashtags = (
            df.select("text").to_pandas()["text"].str.count(cls.MENTION_REGEX)
        )
        return df.with_column(pl.from_pandas(num_hashtags).alias("n_mentions"))

    @classmethod
    @track_size
    def add_num_words_col(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col("text").str.split(" ").arr.lengths().alias("n_words")
        )

    @classmethod
    @track_size
    def filter_num_hashtags_and_cashtags(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("n_cashtags") <= 4).filter(
            pl.col("hashtags") <= 7
        )  # lots of spam with >= 8 hashtags

    @classmethod
    @track_size
    def remove_spam(cls, df: pl.DataFrame) -> pl.DataFrame:
        # more tags than words
        df = df.filter(
            ~(
                (pl.col("n_cashtags") / pl.col("n_words") > 0.5)
                | (pl.col("n_hashtags") / pl.col("n_words") > 0.5)
                | (pl.col("n_mentions") / pl.col("n_words") > 0.5)
            )
        )

        return df

    @classmethod
    @track_size
    def remove_hyperlinks(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(pl.col("text").str.replace_all(cls.URL_REGEX, ""))

    @classmethod
    @track_size
    def remove_crypto_posts(cls, df: pl.DataFrame) -> pl.DataFrame:
        n_cryptoterms_col = df.select(
            [
                pl.col("text")
                .str.to_lowercase()
                .str.contains(term)
                .cast(pl.UInt8)
                .alias(f"contains_{term}")
                for term in cls.CRYPTO_TERMS
            ]
        ).sum(axis=1)

        return df.with_column(n_cryptoterms_col.alias("n_cryptoterms")).filter(
            pl.col("n_cryptoterms") < 2
        )


#%%
clean: pl.DataFrame = (
    df.pipe(Cleaning.remove_hyperlinks)
    .pipe(Cleaning.add_num_words_col)
    .pipe(Cleaning.drop_duplicates)
    .pipe(Cleaning.add_num_cashtags_col)
    .pipe(Cleaning.add_num_hashtags_col)
    .pipe(Cleaning.add_num_mentions_col)
    .pipe(Cleaning.filter_num_hashtags_and_cashtags)
    .pipe(Cleaning.remove_spam)
    .pipe(Cleaning.remove_crypto_posts)
)

#%%
clean.pipe(Cleaning.remove_crypto_posts)

#%%
# sample = clean.sample(1000)
sample = clean.filter(pl.col("n_hashtags") == 7).sample(1000)
with open("outputs/dump/clean_sample.txt", "w") as f:
    f.writelines(f"\n {'-'*256} \n".join(sample.select("text").to_series().to_numpy()))


#%%
# import matplotlib.pyplot as plt
# import seaborn as sns

# from src.utils.plotting import set_style

# set_style()
# fig, ax = plt.subplots(figsize=(15, 6))
# sns.histplot(clean.select("n_hashtags").to_numpy().ravel(), binwidth=1)


#%%
sample = clean.sample(100_000)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()
mtx = tfidf_vec.fit_transform(sample.select("text").to_numpy().ravel())

#%%
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

sims = pairwise_distances(mtx, n_jobs=-1)
