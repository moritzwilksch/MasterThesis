#%%
import polars as pl
import spacy

from src.utils.preprocessing import Preprocessor

#%%
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
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])

#%%
doc = nlp("big bigger biggest $12.34, +4%, and, this ...")
for token in doc:
    print(token.text, token.lemma_, token.is_punct)

#%%


#%%
not_stop_words = [
    "up",
    "down",
    "above",
    "below",
    "against",
    "between",
    "bottom",
    "top",
    "call",
    "put",
    "least",
    "most",
    "much",
    "n't",
    "off",
    "under",
    "over"
]
for elem in not_stop_words:
    try:
        nlp.Defaults.stop_words.remove(elem)
    except KeyError:
        pass

preprocessed_docs = []
for doc in nlp.pipe(df["text"].to_list(), batch_size=64, n_process=-1):
    preprocessed_docs.append(" ".join(w.lemma_ for w in doc))


#%%
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMClassifier


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
            verbose=10
        )

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


model = Pipeline(
    [
        ("vectorizer", TfidfVectorizer()),
        # ("vectorizer", TfidfTransformer()),
        ("kbest", SelectKBest(score_func=chi2)),
        # ("model", LogisticRegression()),
        ("model", LGBMWrapper()),
    ]
)

params = {
    "vectorizer__analyzer": "char_wb",
    "vectorizer__ngram_range": (3, 5),
    # "vectorizer__max_features": 10_000,
    "kbest__k": 10_000,
    # "model__C": 2,
    # "model__n_jobs": 5,
    # "model__max_iter": 350,
    "model__num_leaves": 32,
}

scores = cross_val_score(
    # MultinomialNB(),
    model.set_params(**params),
    preprocessed_docs,
    df["label"].to_list(),
    # X,
    # y,
    cv=5,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=-1,
)

print(np.mean(scores))

#%%
preds = cross_val_predict(
    model.set_params(**params), preprocessed_docs, df["label"].to_list(), n_jobs=-1
)

#%%
import pandas as pd
pred_df = pd.DataFrame(
    {"text": preprocessed_docs, "label": df["label"].to_list(), "pred": preds}
)

#%%
wrong = pred_df.query("label != pred")

#%%
model.fit(preprocessed_docs, df["label"].to_list())
#%%
import eli5
from eli5.lime import TextExplainer

s = "🎯 ticker price target update 🔴 target lowered by royal bank of canada from $ 999.99 to $ 999.99 day quote / change : $ 999.99 ( 9.99 % ) target upside : 99.99 % published : april 99 , 9999"
te = TextExplainer(random_state=42, char_based=True)
te.fit(s, model.predict_proba)
te.explain_prediction()
