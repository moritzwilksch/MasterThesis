import os

import boto3
import dotenv

dotenv.load_dotenv()

bucket = boto3.resource(
    "s3",
    endpoint_url="https://s3.us-west-004.backblazeb2.com",
    aws_access_key_id=os.getenv("BB_KEYID"),
    aws_secret_access_key=os.getenv("BB_APPKEY"),
).Bucket("MasterThesis")
