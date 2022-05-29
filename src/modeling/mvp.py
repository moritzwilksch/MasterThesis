#%%
from matplotlib import projections
import polars as pl
from src.utils.db import get_client
import pandas as pd
import numpy as np

DB = get_client()

#%%
df = pl.from_dicts(
    list(
        DB.thesis.labeled_tweets.find(
            {"label": {"$ne": ""}}, projection={"text": True, "label": True, "_id": False}
        )
    )
)


df = df.with_column(pl.col("text").str.replace(r"\d", "9")).filter(pl.col("label") != "2")

df = df.to_pandas()
#%%
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accs = []
for train_idx, val_idx in kf.split(df["text"], df["label"]):
    xtrain, xval = df.loc[train_idx, "text"], df.loc[val_idx, "text"]
    ytrain, yval = df.loc[train_idx, "label"], df.loc[val_idx, "label"]

    # vectorizer = CountVectorizer()
    vectorizer = TfidfVectorizer()
    xtrain = vectorizer.fit_transform(xtrain)
    xval = vectorizer.transform(xval)

    # model = LogisticRegression()
    # model = MultinomialNB()
    model = SVC()
    model.fit(xtrain, ytrain)

    accs.append(accuracy_score(yval, model.predict(xval)))

print(f"acc = {np.mean(accs):.2%} (SD={np.std(accs):.4})")

#%%
from sklearn.metrics import classification_report
print(classification_report(yval, model.predict(xval)))
