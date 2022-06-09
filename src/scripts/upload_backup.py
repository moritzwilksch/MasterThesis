from src.utils.storage import bucket

# bucket.upload_file("labeled_tweets_backup.gz", "labeled_tweets_backup.gz")
bucket.upload_file(
    "data/labeled/labeled_tweets.parquet", "data/labeled/labeled_tweets.parquet"
)
bucket.upload_file("tuning/optuna.db", "tuning/optuna.db")
print("done.")
