import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def get_client():
    client = MongoClient(
        f"mongodb://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASSWD')}@{os.getenv('MONGO_HOST')}:27017/thesis",
        authSource="thesis",
    )
    return client
