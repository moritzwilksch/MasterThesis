#%%
import pandas as pd
from sklearn import multiclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

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

pipe = Pipeline(
    [
        ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))),
        ("model", LogisticRegression(C=1.7, max_iter=350, n_jobs=-1)),
    ]
)

train_aucs = []
test_aucs = []

unlabeled = train_data[train_data["pseudo_label"].isnull()]
to_be_labeled_idx = unlabeled.sample(500).index

for ii in range(10):

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

    train_data = train_data.assign(entropy=entropy(pipe.predict_proba(train_data["text"])))
    to_be_labeled_idx = train_data[train_data["pseudo_label"].isnull()].sample(500).index

print(train_aucs)
print(test_aucs)
