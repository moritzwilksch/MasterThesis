#%%
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.utils.preprocessing import Preprocessor

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
# df = df.assign(label=df["label"].map(CLASS_MAPPING).astype("category"))

X = df["text"]
y = df["label"]


#%%
pipe = Pipeline(
    [
        ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))),
        ("model", LogisticRegression(C=1.7, max_iter=450)),
    ]
)

scores = cross_val_score(
    pipe,
    X,
    y,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=5,
)

print(np.mean(scores))

#%%
pipe.fit(X, y)
preds = pipe.predict(X)

#%%
roc_auc_score(y, pipe.predict_proba(X)[:, 1])

#%%
from sklearn.model_selection import cross_val_predict

preds = cross_val_predict(pipe, X, y)

#%%
from sklearn.metrics import classification_report
print(classification_report(y, preds))
