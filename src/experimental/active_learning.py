#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#%%
all_data = pl.read_csv(
    "data/raw/OLD_dataset.csv", columns=["tweet", "sentiment"]
).to_pandas()
xtrain, xtest, ytrain, ytest = train_test_split(
    all_data["tweet"], all_data["sentiment"], random_state=42
)
df = pl.from_pandas(pd.concat([xtrain, ytrain], axis=1))
test_df = pl.from_pandas(pd.concat([xtest, ytest], axis=1))

#%%


class Oracle:
    def __init__(
        self, data, model: LogisticRegression, vectorizer: TfidfVectorizer
    ) -> None:
        self.data = data
        self.data = self.data.with_columns(
            [pl.Series(range(data.height)).alias("id"), pl.lit(None).alias("assigned")]
        )

        self.model = model
        self.vectorizer = vectorizer
        self.vectorizer.fit(self.data["tweet"])

    def label_batch(self, indeces):
        self.data = self.data.with_column(
            pl.when(pl.col("id").is_in(pl.Series(indeces)))
            .then(pl.col("sentiment"))
            .otherwise(pl.col("assigned"))
            .alias("assigned")
        )

    def train_on_labeled(self):
        subset = self.data.filter(pl.col("assigned").is_not_null())
        X = self.vectorizer.transform(subset["tweet"])
        y = subset["assigned"]
        self.model.fit(X, y)

    def sample_next_batch(self):
        X = self.vectorizer.transform(self.data["tweet"])
        preds = self.model.predict_proba(X)
        entropies = -(preds * np.log(preds)).sum(axis=1)
        temp_df_with_entropy = self.data.with_column(pl.Series("entropy", entropies))

        return (
            temp_df_with_entropy.filter(pl.col("assigned").is_null())
            .sort("entropy")
            .tail(100)
            .select("id")
            .to_series()
            .to_list()
        )


class RandomSamplingOracle(Oracle):
    def sample_next_batch(self):
        return np.random.choice(
            self.data.filter(pl.col("assigned").is_null())
            .select("id")
            .to_series()
            .to_list(),
            size=100,
        )


class ActiveLearner:
    def __init__(self, oracle: Oracle) -> None:
        self.oracle = oracle

    def run(self):
        first_batch = np.random.choice(
            self.oracle.data.select("id").to_series().to_list(), size=100
        )
        self.oracle.label_batch(first_batch)

        accs = []
        for ii in range(self.oracle.data.height // 100):
            self.oracle.train_on_labeled()

            preds = self.oracle.model.predict(self.oracle.vectorizer.transform(xtest))
            iter_acc = accuracy_score(ytest, preds)
            accs.append(iter_acc)

            next_batch = self.oracle.sample_next_batch()
            self.oracle.label_batch(next_batch)

        return accs


#%%
oracle1 = Oracle(df, LogisticRegression(random_state=42), TfidfVectorizer())
oracle2 = RandomSamplingOracle(
    df, LogisticRegression(random_state=42), TfidfVectorizer()
)

al1 = ActiveLearner(oracle1)
accs1 = al1.run()

al2 = ActiveLearner(oracle2)
accs2 = al2.run()


#%%
fig, ax = plt.subplots()
ax.plot(np.array(range(len(accs1))) * 100, accs1, label="Active Learning", color="blue")
ax.plot(np.array(range(len(accs2))) * 100, accs2, label="Random Sampling", color="grey")
ax.legend()
ax.set(xlabel="# data point", ylabel="Accuracy")
sns.despine()
