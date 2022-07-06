#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sqlalchemy import all_

from src.utils.plotting import Colors, set_style
from src.utils.preprocessing import Preprocessor

set_style()

#%%
df = pd.read_parquet("data/labeled/labeled_tweets.parquet")
df = df.assign(label=lambda d: d["label"].replace("0", "2"))
finsome = pd.read_csv("data/finSoMe/finsome.csv")

#%%
def build_data(df, finsome):
    all_data = pd.DataFrame(
        {
            "text": df["text"].to_list() + finsome["tweet"].to_list(),
            "src": ["twitter"] * len(df) + ["stocktwits"] * len(finsome),
        }
    )

    return all_data.assign(
        text=all_data["text"].astype("string"), src=all_data["src"].astype("category")
    )


all_data = build_data(df, finsome)

prepper = Preprocessor()
all_data = prepper.process(pl.DataFrame(all_data)).to_pandas()

#%%
pipeline = Pipeline(
    [("vectorizer", TfidfVectorizer()), ("model", LogisticRegression())]
)

scores = cross_val_score(
    pipeline,
    all_data["text"],
    all_data["src"],
    scoring=make_scorer(roc_auc_score, needs_proba=True),
    n_jobs=5,
    cv=KFold(shuffle=True),
)

print(scores)

#%%
pipeline.fit(
    all_data["text"],
    all_data["src"],
)

#%%
idxs = pipeline["model"].coef_.argsort()
mapper = {v: k for k, v in pipeline["vectorizer"].vocabulary_.items()}

N = 30
print("Top Positive")
print("-" * 20)
for item in idxs.ravel()[:-N:-1]:
    print(mapper[item])

print("Top Negative")
print("-" * 20)
for item in idxs.ravel()[:N]:
    print(mapper[item])

#%%
from collections import Counter

df_words = (
    pd.DataFrame(
        Counter(
            " ".join(all_data.query("src == 'twitter'")["text"].to_list()).split(" ")
        ),
        index=[0],
    )
    .T.reset_index()
    .rename({0: "frequency", "index": "word"}, axis=1)
    .assign(
        frequency=lambda d: d["frequency"] / d["frequency"].count(),
        word=lambda d: d["word"].astype("string"),
    )
)

finsome_words = (
    pd.DataFrame(
        Counter(
            " ".join(all_data.query("src == 'stocktwits'")["text"].to_list()).split(" ")
        ),
        index=[0],
    )
    .T.reset_index()
    .rename({0: "frequency", "index": "word"}, axis=1)
    .assign(
        frequency=lambda d: d["frequency"] / d["frequency"].count(),
        word=lambda d: d["word"].astype("string"),
    )
)

#%%
freqs = pd.merge(
    left=df_words,
    right=finsome_words,
    how="outer",
    on="word",
    suffixes=["_twitter", "_stocktwits"],
).dropna()

#%%
freqs.assign(
    delta=np.abs(freqs["frequency_twitter"] / freqs["frequency_stocktwits"])
).sort_values("delta").head(50)
