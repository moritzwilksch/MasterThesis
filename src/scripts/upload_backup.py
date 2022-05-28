from src.utils.storage import bucket

bucket.upload_file("labeled_tweets_backup.gz", "labeled_tweets_backup.gz")
print("done.")
