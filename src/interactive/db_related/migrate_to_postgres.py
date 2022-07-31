#%%
import datetime
import os

import dotenv
import ray
from sqlalchemy import (Column, Date, Integer, MetaData, String, Table,
                        create_engine)
from sqlalchemy.dialects.postgresql import ARRAY, JSON

ray.init(ignore_reinit_error=True)

dotenv.load_dotenv()

engine = create_engine(
    f"postgresql://postgres:{os.getenv('POSTGRES_PASSWD')}@localhost:5432/thesis"
)

metadata_obj = MetaData(bind=engine)
tweets = Table(
    "tweets",
    metadata_obj,
    Column("id", Integer, primary_key=True, auto_increment=True),
    Column("created_at", Date),
    Column("twitter_id", String),
    Column("entities", JSON),
    Column("author_id", String),
    Column("text", String),
    Column("retweet_count", Integer),
    Column("reply_count", Integer),
    Column("like_count", Integer),
    Column("quote_count", Integer),
)

metadata_obj.create_all()
#%%
@ray.remote
def mongo_doc_to_sql(doc: dict):
    return {
        "created_at": datetime.datetime.fromisoformat(doc["created_at"][:-1]),
        "twitter_id": doc["id"],
        "entities": doc.get("entities"),
        "author_id": doc.get("author_id"),
        "text": doc.get("text"),
        "retweet_count": doc["public_metrics"].get("retweet_count"),
        "reply_count": doc["public_metrics"].get("reply_count"),
        "like_count": doc["public_metrics"].get("like_count"),
        "quote_count": doc["public_metrics"].get("quote_count"),
    }


#%%
from src.utils.db import get_client

DB = get_client()
result = DB["thesis"]["prod_tweet"].find({}).limit(100_000)
result = list(result)


#%%
flattened_result_futures = [mongo_doc_to_sql.remote(doc) for doc in result]
flattened_results = ray.get(flattened_result_futures)

#%%
with engine.connect() as conn:
    conn.execute(tweets.insert(), flattened_results)

#%%
ray.shutdown()
