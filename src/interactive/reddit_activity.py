#%%
import pandas as pd
import polars as pl
import requests

#%%
url = "https://api.pushshift.io/reddit/submission/search"

params = {
    "subreddit": "Investing",
    "size": 1000,
}
response = requests.get(url, params=params)
# print(response.json())
rawdf: pd.DataFrame = pd.DataFrame(response.json()["data"])
#%%
df = pl.from_pandas(rawdf["selftext"]).to_frame()

#%%
df.filter(~pl.col("selftext").is_in(pl.Series(["", "[removed]", "[deleted]"]))).select(
    pl.col("selftext").str.lengths().mean()
)
