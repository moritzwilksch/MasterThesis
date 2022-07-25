#%%
import pandas as pd
from sklearn import multiclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import set_style
import ray

set_style()

all_data = (
    pd.read_parquet("data/labeled/labeled_tweets.parquet")
    .assign(pseudo_label=None, entropy=0)
    .convert_dtypes()
    .assign(pseudo_label=lambda d: d.pseudo_label.astype("string"))
    .assign(label=lambda d: d["label"].map({"0": "2", "1": "1", "2": "2", "3": "3"}))
)
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

#%%
def entropy(y):
    return -(y * np.log(y)).sum(axis=1)


#%%
from enum import Enum


class SamplingStrategy(Enum):
    random = "random"
    entropy = "entropy"


@ray.remote(num_returns=2)
def run_experiment(
    train_data: pd.DataFrame,
    strategy: SamplingStrategy = SamplingStrategy.random,
    batch_size: int = 500,
) -> tuple[list, list]:
    train_data = train_data.copy()

    pipe = Pipeline(
        [
            ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))),
            ("model", LogisticRegression(C=1.7, max_iter=350, n_jobs=-1)),
        ]
    )

    train_aucs = []
    test_aucs = []

    unlabeled = train_data[train_data["pseudo_label"].isnull()]
    to_be_labeled_idx = unlabeled.sample(batch_size).index

    for ii in range(8_000 // batch_size - 1):

        train_data.loc[to_be_labeled_idx, "pseudo_label"] = train_data.loc[
            to_be_labeled_idx, "label"
        ]

        labeled_data = train_data.loc[~train_data["pseudo_label"].isna()]
        pipe.fit(labeled_data["text"], labeled_data["pseudo_label"])

        train_preds = pipe.predict_proba(labeled_data["text"])
        train_aucs.append(
            roc_auc_score(labeled_data["pseudo_label"], train_preds, multi_class="ovr")
        )

        test_preds = pipe.predict_proba(test_data["text"])
        test_aucs.append(roc_auc_score(test_data["label"], test_preds, multi_class="ovr"))

        train_data = train_data.assign(
            entropy=entropy(pipe.predict_proba(train_data["text"]))
        )

        if strategy == SamplingStrategy.random:
            to_be_labeled_idx = (
                train_data[train_data["pseudo_label"].isnull()].sample(batch_size).index
            )
        elif strategy == SamplingStrategy.entropy:
            to_be_labeled_idx = (
                train_data[train_data["pseudo_label"].isnull()]
                .sort_values("entropy", ascending=False)
                .index[:batch_size]
            )

    return train_aucs, test_aucs


#%%
ray.init(ignore_reinit_error=True)

BATCH_SIZE = 100
results_random = run_experiment.remote(
    train_data, SamplingStrategy.random, batch_size=BATCH_SIZE
)
results_entropy = run_experiment.remote(
    train_data, SamplingStrategy.entropy, batch_size=BATCH_SIZE
)

#%%
train_aucs_random, test_aucs_random = ray.get(results_random)
train_aucs_entropy, test_aucs_entropy = ray.get(results_entropy)

#%%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(
    np.arange(len(test_aucs_random)) * BATCH_SIZE,
    test_aucs_random,
    label="Random",
    color="k",
)
ax.plot(
    np.arange(len(test_aucs_entropy)) * BATCH_SIZE,
    test_aucs_entropy,
    label="Entropy",
    color="blue",
)
ax.set_title("Random vs. Entropy Sampling (batch_size = 100)", weight="bold")
ax.set_xlabel("Number of labeled samples", weight="bold")
ax.set_ylabel("Test AUC", weight="bold")
ax.legend()
sns.despine()
fig.savefig("outputs/plots/active_learning.pdf", bbox_inches="tight")
