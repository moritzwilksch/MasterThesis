#%%
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import TransformerSAModel
from src.utils.plotting import Colors, set_style, scale_lightness
from src.utils.preprocessing import Preprocessor

set_style()

#%%
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
df = prepper.process(df)
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
).to_pandas()

#%%
model = joblib.load("outputs/models/final_LogisticRegressionModel.gz")

#%%

train_sizes, train_scores, test_scores = learning_curve(
    model,
    df["text"],
    df["label"],
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=-1,
    train_sizes=np.arange(1_000, 8_000 + 1, 1_000),
)

#%%
fig, ax = plt.subplots(figsize=(12, 6))
plot_df = pd.DataFrame(
    {"train_size": np.tile(train_sizes, 5), "test_score": test_scores.T.ravel()}
)
sns.lineplot(
    data=plot_df,
    x="train_size",
    y="test_score",
    ax=ax,
    color=Colors.DARKBLUE.value,
    lw=3,
    zorder=10,
)
sns.scatterplot(
    data=plot_df.groupby("train_size", as_index=False).mean(),
    x="train_size",
    y="test_score",
    color=Colors.DARKBLUE.value,
    s=75,
    ec=scale_lightness(sns.desaturate(Colors.DARKBLUE.value, 1), 0.5),
    zorder=20,
)

ax.grid(True, which="major", axis="y", ls="--")
ax.xaxis.set_major_formatter("{x:,.0f}")
sns.despine()
ax.set_xlabel("Size of Training Dataset", weight="bold", labelpad=10)
ax.set_ylabel("Out-of-sample ROC AUC", weight="bold", labelpad=10)
fig.savefig("outputs/plots/learning_curve.pdf", bbox_inches="tight")
