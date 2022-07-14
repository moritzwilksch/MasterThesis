#%%
from abc import ABC

import polars as pl
from rich import print

from src.utils.db import get_client
from src.utils.storage import bucket

DB = get_client()

#%%
df = pl.read_parquet(
    # "data/raw/db_export_small.parquet"
    "data/raw/gme_tweets.parquet"
)  # small only has date, text, id

#%%
def track_size(f: callable, *args, **kwargs) -> callable:
    def wrapper(cls, df: pl.DataFrame, *args, **kwargs):
        result = f(cls, df)
        print(f"After {f.__name__}(...): n = {result.height:,}")
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
        return df.with_column(
            pl.col("text").str.count_match(cls.CASHTAG_REGEX).alias("n_cashtags")
        )

    @classmethod
    @track_size
    def add_num_hashtags_col(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col("text").str.count_match(cls.HASHTAG_REGEX).alias("n_hashtags")
        )

    @classmethod
    @track_size
    def add_num_mentions_col(cls, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.col("text").str.count_match(cls.MENTION_REGEX).alias("n_mentions")
        )

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
            pl.col("n_hashtags") <= 7
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
            pl.col("n_cryptoterms") <= 2
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
if False:  # DANGER ZONE
    prompt = input(
        "DO YOU WANT TO SAMPLE AGAIN? OVERWRITES LOCAL FILE, REMOTE FILE, AND DB!!"
    )
    if prompt != "yes":
        print("Changing nothing. Exiting.")
        exit(0)

    sample = clean.sample(15_000, seed=42)
    with open("outputs/dump/clean_sample.txt", "w") as f:
        f.writelines(
            f"\n {'-'*256} \n".join(sample.select("text").to_series().to_numpy())
        )

    sample.to_parquet("data/raw/sample.parquet")
    bucket.upload_file("data/raw/sample.parquet", "data/raw/sample.parquet")

    dicts = sample.to_dicts()
    for dd in dicts:
        dd["label"] = ""

    DB.thesis.labeled_tweets.insert_many(dicts)

#%%
