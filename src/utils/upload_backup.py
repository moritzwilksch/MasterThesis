from src.utils.storage import bucket

bucket.upload_file("backup.gz", "backup.gz")
print("done.")