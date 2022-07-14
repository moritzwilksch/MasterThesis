#%%
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.utils.preprocessing import Preprocessor

#%%
df = pd.read_parquet("data/labeled/labeled_tweets.parquet")
CLASS_MAPPING = {
    "0": "some_sentiment",
    "1": "some_sentiment",
    "2": "no_sentiment",
    "3": "some_sentiment",
}
prepper = Preprocessor()
df = prepper.process(pl.DataFrame(df))
df = df.to_pandas()
df = df.assign(label=df["label"].map(CLASS_MAPPING).astype("category"))

X = df["text"]
y = df["label"]


#%%
pipe = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                analyzer="char_wb", ngram_range=(4, 4), min_df=1.021498227938179e-06
            ),
        ),
        ("model", LogisticRegression(C=1.4281305851678692, max_iter=450, n_jobs=-1)),
    ]
)

scores = cross_val_score(
    pipe,
    X,
    y,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=5,
)

print(np.mean(scores))

#%%
pipe.fit(X, y)
preds = pipe.predict(X)

#%%
roc_auc_score(y, pipe.predict_proba(X)[:, 1])

#%%
from sklearn.model_selection import cross_val_predict

preds = cross_val_predict(pipe, X, y, method="predict")

#%%
from sklearn.metrics import classification_report

print(classification_report(y, preds))
# print(roc_auc_score(y.values, preds[:, 1]))

#%%
def classification_report_to_tex(cr):
    lines = [
        " & "
        + " & ".join(
            [
                "\\textbf{Precision}",
                "\\textbf{Recall}",
                "\\textbf{F1-Score}",
                "\\textbf{Support}",
            ]
        )
    ]
    for cat, data in cr.items():
        if isinstance(data, float):
            continue

        lines.append(
            f"\\textbf{{{cat}}} & {data['precision']:.3f} & {data['recall']:.3f} & {data['f1-score']:.3f} & {data['support']:,d}"
        )

    return "\\\\\n".join(lines) + "\\\\"


#%%
print(classification_report_to_tex(classification_report(y, preds, output_dict=True)))
