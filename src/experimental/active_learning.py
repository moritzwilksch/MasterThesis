#%%
from numpy import vectorize
import polars as pl
from abc import ABC, abstractmethod
import numpy as np
from sklearn.linear_model import LogisticRegression

#%%
df = pl.read_csv("data/raw/OLD_dataset.csv", columns=["tweet", "sentiment"])

#%%
from sklearn.feature_extraction.text import TfidfVectorizer


class Oracle:
    def __init__(self, model, vectorizer, data):
        self.model = model
        self.vectorizer = vectorizer
        self.vectorizer.fit(data["tweet"].to_pandas())

    def train_on_labeled(self, data):
        subset = data.filter(pl.col("assigned").is_not_null()).to_pandas()
        X = self.vectorizer.transform(subset["tweet"])
        y = subset["assigned"]

        self.model.fit(X, y)
        return self.model

    def sample_next_batch(self, data):
        model = self.train_on_labeled(data)
        preds = model.predict_proba(self.vectorizer.transform(data["tweet"]))
        entropies = (-np.log(preds) * preds).sum(axis=1)
        data = data.with_column(pl.Series("entropy", entropies))

        return (
            data.filter(pl.col("assigned").is_null())
            .sort("entropy")
            .select("id")
            .tail(100)
            .to_series()
            .to_list()
        )


class ActiveLearner:
    def __init__(self, data, oracle):
        self.data = data.with_columns(
            [pl.Series(range(data.height)).alias("id"), pl.lit(None).alias("assigned")]
        )

        first_batch = np.random.randint(0, data.height, size=100)
        self.data = self.label_batch(self.data, first_batch)

        self.oracle = oracle

    def label_batch(self, data, idxs):
        return data.with_column(
            pl.when(pl.col("id").is_in(pl.Series(idxs)) & pl.col("assigned").is_null())
            .then(pl.col("sentiment"))
            .otherwise(pl.col("assigned"))
            .alias("assigned")
        )

    def learn(self):
        for i in range(20):
            next_batch = self.oracle.sample_next_batch(self.data)
            self.data = self.label_batch(self.data, next_batch)
            print(self.data.select(pl.col("assigned").is_null().sum()))
            print(next_batch)
            input()


oracle = Oracle(LogisticRegression(), TfidfVectorizer(), df)
al = ActiveLearner(df, oracle)
al.learn()
