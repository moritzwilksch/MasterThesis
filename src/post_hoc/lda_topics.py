#%%
import pandas as pd
import polars as pl
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from src.utils.preprocessing import Preprocessor

#%%
df = pd.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
df = prepper.process(pl.from_pandas(df)).to_pandas()

#%%
lda = LatentDirichletAllocation(n_components=5, n_jobs=-1)
cv = CountVectorizer()
cv.fit(df["text"])
lda.fit(cv.transform(df["text"]))

#%%
mapper = {v: k for k, v in cv.vocabulary_.items()}
top_idxs = lda.components_.argsort(axis=1)[:, -20:]

for row in top_idxs:
    print("--- TOPIC ---")
    for elem in row:
        print(mapper[elem])
