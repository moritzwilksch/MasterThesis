#%%
import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.preprocessing import Preprocessor
import polars as pl


#%%
df = pd.read_parquet("data/labeled/labeled_tweets.parquet")
CLASS_MAPPING = {
    "0": "nospam",
    "1": "nospam",
    "2": "spam",
    "3": "nospam",
}
prepper = Preprocessor()
df = prepper.process(pl.DataFrame(df)).to_pandas()
df = df.assign(label=df["label"].map(CLASS_MAPPING).astype("category"))

X = df["text"]
y = df["label"]


#%%
pipe = Pipeline(
    [
        ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))),
        ("model", LogisticRegression(C=1.7)),
    ]
)

scores = cross_val_score(
    pipe, X, y, scoring=make_scorer(roc_auc_score, needs_proba=True), n_jobs=5
)

print(np.mean(scores))

#%%
pipe.fit(X, y)
preds = pipe.predict(X)

#%%
roc_auc_score(y, pipe.predict_proba(X)[:, 1])