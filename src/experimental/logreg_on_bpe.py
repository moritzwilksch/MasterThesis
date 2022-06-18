#%%
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl
from black import Line
from bpemb import BPEmb

from src.utils.preprocessing import Preprocessor
import torchtext

#%%
tokenizer = BPEmb(lang="en")
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")

prepper = Preprocessor()
pyfin_senti_data = prepper.process(df)


# TODO: should we merge labels? drop 2?
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)
#%%
def tokenize(s):
    v = np.zeros((len(s), 3_000), dtype=np.int32)
    ids = tokenizer(s)
    counters = [Counter(doc) for doc in ids]

    for doc_idx, c in enumerate(counters):
        v[doc_idx, list(c.keys())] = list(c.values())

    return v


#%%
tokenizer = torchtext.data.functional.sentencepiece_numericalizer(
    torchtext.data.functional.load_sp_model(
        f"outputs/tokenizers/retraining_trainval.model"
    )
)
X = tokenize(df["text"])
y = df["label"].to_list()

from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# tfidf = TfidfTransformer()
tfidf = TfidfVectorizer(ngram_range=(4, 6), analyzer="char_wb", stop_words="english")

X = np.array(df["text"].to_list())
X_tfidf = tfidf.fit_transform(X)

from sklearn.feature_selection import SelectKBest

kb = SelectKBest(k=5000)
X_tfidf = kb.fit_transform(X_tfidf, y)

#%%

scores = cross_val_score(
    # LogisticRegression(C=1, random_state=42, max_iter=350),
    # RandomForestClassifier(n_estimators=100,  min_samples_leaf=3, n_jobs=3, max_features=0.5),
    # MultinomialNB(),
    LGBMClassifier(n_estimators=100, n_jobs=3, num_leaves=16),
    X_tfidf,
    y,
    cv=5,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=-1,
)

print(f"Mean score: {np.mean(scores):.3f} (SD: {np.std(scores):.3f}), {scores}")

#%%
model = LGBMClassifier(n_estimators=100, n_jobs=3, num_leaves=16)
model.fit(X_tfidf, y)

#%%
finsome = pd.read_json("data/finSoMe/fin-SoMe.json")
finsome = pd.DataFrame(
    {
        "text": finsome["tweet"],
        "label": finsome["market_sentiment"].map(
            {"Unsure": 2, "Bearish": 3, "Bullish": 1}
        ),
    }
)

prepper = Preprocessor()
finsome = prepper.process(pl.from_pandas(finsome)).to_pandas()

#%%
# finsomeX = tfidf.transform(np.array(["buy puts"]))
finsomeX = tfidf.transform(finsome["text"])
finsomeX = kb.transform(finsomeX)
preds = model.predict_proba(finsomeX)
print(preds)
#%%
roc_auc_score(finsome["label"], preds, multi_class="ovr")