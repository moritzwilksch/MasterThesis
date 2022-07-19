#%%
import string

import nltk
import numpy as np
import pandas as pd
import polars as pl
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.utils.preprocessing import Preprocessor

stopwords = list(stopwords.words("english"))

#%%
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)

#%%
meta_features = df.select(
    [
        pl.col("text").str.count_match(prepper.CASHTAG_REGEX).alias("n_cashtags"),
        pl.col("text").str.count_match(prepper.MENTION_REGEX).alias("n_mentions"),
        (pl.col("text").str.count_match(r"[A-Z]") / pl.col("text").str.lengths()).alias(
            "pct_capitalized"
        ),
        (pl.col("text").str.count_match(r"\d") / pl.col("text").str.lengths()).alias(
            "pct_digits"
        ),
        (
            pl.col("text")
            .str.split(" ")
            .arr.eval(pl.element().is_in(stopwords).sum())
            .arr.first()
            / pl.col("text").str.split(" ").arr.lengths()
        ).alias("pct_stopwords"),
        (
            pl.col("text")
            .str.split("")  # single chars
            .arr.eval(pl.element().is_in(string.punctuation).sum())
            .arr.first()
            / pl.col("text").str.split(" ").arr.lengths()
        ).alias("pct_punctuation"),
    ]
)

#%%

#%%


meta_features
#%%
df = prepper.process(df)
#%%
X = pl.concat([df.select("text"), meta_features], how="horizontal")
from catboost import CatBoostClassifier, CatboostError
from lightgbm import LGBMClassifier
#%%
from sklearn.base import BaseEstimator, ClassifierMixin


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, num_leaves: int = 32):
        self.num_leaves = num_leaves
        self.classes_ = ["1", "2", "3"]
        self.model = LGBMClassifier(
            boosting_type="gbdt", n_estimators=200, num_leaves=self.num_leaves
        )
        # self.model = CatBoostClassifier()

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


#%%

model = Pipeline(
    [
        (
            "ct",
            ColumnTransformer(
                [
                    (
                        "text_pipe",
                        Pipeline(
                            [
                                (
                                    "vectorizer",
                                    TfidfVectorizer(
                                        analyzer="char_wb", ngram_range=(4, 4)
                                    ),
                                ),
                                # ("kbest", SelectKBest(k=10_000)),
                            ]
                        ),
                        "text",
                    )
                ],
                remainder="passthrough",
            ),
        ),
        (
            "model",
            # LogisticRegression(random_state=42, n_jobs=-1, max_iter=550, C=1.42),
            LGBMWrapper(num_leaves=32),
        ),
    ]
)


score = cross_val_score(
    model,
    X=X.to_pandas(),
    y=df["label"].to_pandas(),
    cv=5,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=5,
)

print(np.mean(score), score)

#%%
model = model.fit(X.to_pandas(), df["label"].to_pandas())
#%%
from src.utils.plotting import set_style

set_style()
from lightgbm.plotting import plot_importance, plot_tree

plot_tree(model["model"].model, dpi=800)

#%%
preds = model.predict(X.to_pandas())
preddf = df.to_pandas().copy()
preddf["prediction"] = preds
preddf["prediction"] = preddf["prediction"].astype("string")

wrong = preddf.query("(label=='3' & prediction=='1')")
# wrong = df.query("label != prediction")
wrong
