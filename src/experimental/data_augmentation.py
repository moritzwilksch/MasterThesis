#%%
from collections import Counter
from tkinter import Label

import mlflow
import numpy as np
import pandas as pd
import polars as pl
import torchtext
from bpemb import BPEmb
from lightgbm import LGBMClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.preprocessing import Preprocessor

# _experiment_id = mlflow.create_experiment("other-fast-models")
mlflow.set_experiment("other-fast-models")

#%%

TOK = "custom"

if TOK == "bpe":
    tokenizer = BPEmb(lang="en", vs=5_000).encode_ids
elif TOK == "custom":
    tokenizer = torchtext.data.functional.sentencepiece_numericalizer(
        torchtext.data.functional.load_sp_model(
            f"outputs/tokenizers/retraining_trainval.model"
        )
    )

df = pl.read_parquet("data/labeled/labeled_tweets.parquet")

prepper = Preprocessor()
df = prepper.process(df)

df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)
#%%
def tokenize(s):
    v = np.zeros((len(s), 10_000), dtype=np.int32)
    ids = tokenizer(s)
    counters = [Counter(doc) for doc in ids]

    for doc_idx, c in enumerate(counters):
        v[doc_idx, list(c.keys())] = list(c.values())

    return v


#%%
X = tokenize(df["text"])
y = df["label"].to_numpy()

from nltk.corpus import stopwords
#%%
# X = tokenize(df["text"])
from nltk.stem import PorterStemmer
# stemmer = PorterStemmer()
# df = df.with_column(
#     pl.col("text")
#     .str.split(" ")
#     .arr.eval(pl.element().apply(lambda x: stemmer.stem(x)))
#     .arr.join(" ")
# )
# X = df["text"].to_pandas()
# y = df["label"].to_list()
from textaugment import EDA

dftrainval, dftest = train_test_split(df.to_pandas())

t = EDA()


def augment_text(text: str) -> str:
    # print(text)
    text = t.synonym_replacement(text, n=2)
    # text = t.random_insertion(text, n=2)
    # text = t.random_deletion(text)
    return text


# augmented = pl.from_pandas(dftrainval).select(
#     [pl.col("text").apply(augment_text).alias("text"), pl.col("label")]
# )

# dftrainval = pd.concat([dftrainval.drop("id", axis=1), augmented.to_pandas()])

#%%
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import FunctionTransformer


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_leaves: int = 32):
        self.num_leaves = num_leaves
        self.classes_ = ["1", "2", "3"]
        self.model = LGBMClassifier(n_estimators=250, num_leaves=self.num_leaves)

    def fit(self, X, y):
        xtrain, xval, ytrain, yval = train_test_split(X, y, test_size=0.1)
        self.model.fit(
            xtrain,
            ytrain,
            eval_set=[(xval, yval)],
            early_stopping_rounds=25,
            eval_metric="auc_mu",
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


with mlflow.start_run():
    KBEST = 3_000

    model = Pipeline(
        [
            ("vectorizer", TfidfVectorizer()),
            # ("vectorizer", TfidfTransformer()),
            # ("kbest", SelectKBest()),
            # ("model", LogisticRegression()),
            ("model", LGBMWrapper()),
        ]
    )

    params = {
        "vectorizer__analyzer": "char_wb",
        "vectorizer__ngram_range": (4, 4),
        "vectorizer__max_features": 10_000,
        # "kbest__k": KBEST,
        "model__C": 1.7,
        "model__n_jobs": 3,
        "model__max_iter": 350,
        # "model__num_leaves": 32,
    }

    mlflow.log_params(params)
    mlflow.log_param("modeltype", model["model"].__class__.__name__)

    scores = cross_val_score(
        # MultinomialNB(),
        model.set_params(**params),
        dftrainval["text"],
        dftrainval["label"],
        # X,
        # y,
        cv=5,
        scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
        n_jobs=-1,
    )

    # scores = []
    # for trainidx, validx in KFold(n_splits=5, shuffle=True).split(dftrainval):
    #     _train = dftrainval.iloc[trainidx]
    #     _val = dftrainval.iloc[validx]

    #     augmented = pl.from_pandas(_train).select(
    #         [pl.col("text").apply(augment_text).alias("text"), pl.col("label")]
    #     )

    #     _train = pd.concat([_train.drop("id", axis=1), augmented.to_pandas()])

    #     model.set_params(**params)
    #     model.fit(_train["text"], _train["label"])

    #     y_pred = model.predict_proba(_val["text"])
    #     scores.append(roc_auc_score(_val["label"], y_pred, multi_class="ovr"))

    print(f"Mean score: {np.mean(scores):.3f} (SD: {np.std(scores):.3f}), {scores}")
    # mlflow.lightgbm.log_model(model, "model")
    mlflow.log_metric("val_auc", np.mean(scores))


#%%
from sklearn.model_selection import train_test_split

# xtrain, xtest, ytrain, ytest = train_test_split(X, y)
# model.fit(xtrain, ytrain)

model.fit(dftrainval["text"], dftrainval["label"])

preds = model.predict_proba(dftest["text"])
print(roc_auc_score(dftest["label"], preds, multi_class="ovr"))

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
finsome = prepper.process(pl.from_pandas(finsome))  # .to_pandas()
# finsome = finsome.with_column(
#     pl.col("text")
#     .str.split(" ")
#     .arr.eval(pl.element().apply(lambda x: stemmer.stem(x)))
#     .arr.join(" ")
# )
preds = model.predict_proba(finsome["text"])
print(roc_auc_score(finsome["label"], preds, multi_class="ovr"))

#%%
tokens = model["vectorizer"].transform(["lol, happy I'm short $AAPL at 4/20 $59"])
ids = np.argwhere(tokens[0, :] != 0)[:, -1]
mapping = {v: k for k, v in model["vectorizer"].vocabulary_.items()}
for ii in ids:
    print(mapping[ii])
