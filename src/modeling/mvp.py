#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import shap
import toml
from matplotlib import projections
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

from src.utils.db import get_client
from src.utils.preprocessing import Preprocessor

DB = get_client()

#%%
stopwords = toml.load("config/stopwords.toml").get("stopwords")


#%%
df = pl.from_dicts(
    list(
        DB.thesis.labeled_tweets.find(
            {"label": {"$ne": ""}},
            projection={"text": True, "label": True, "_id": False},
        )
    )
)

# sw_remove_expr = pl.col("text").str.to_lowercase()
# for word in stopwords:
#     sw_remove_expr = sw_remove_expr.str.replace_all(f" {word} ", "")

prepper = Preprocessor()
df = prepper.process(df)

df = df.filter(pl.col("label") != "2")
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)
df = df.to_pandas()


# #%%

# explainer = shap.Explainer(
#     model,
#     xtrain,
#     feature_names=vectorizer.get_feature_names_out(),
#     output_names=["unc", "pos", "neu", "neg"],
# )
# shap_values = explainer(xval)

# #%%


# def plot_top_bottom(shap_values, class_idx, n):
#     averaged_shap_values = shap_values[:, :, class_idx].mean(0).values

#     top_idxs = averaged_shap_values.argsort()[-n:]
#     # bottom_idxs = shap_values[:, :, 1].mean(0).values.argsort()[:n]

#     top_words = vectorizer.get_feature_names_out()[top_idxs]
#     # bottom_words = vectorizer.get_feature_names_out()[bottom_idxs]

#     fig, axes = plt.subplots(1, 1, figsize=(10, 5))
#     axes.barh(y=top_words, width=averaged_shap_values[top_idxs])
#     # axes[1].barh(y=bottom_words, width=averaged_shap_values[bottom_idxs])
#     fig.tight_layout()
#     sns.despine()


# plot_top_bottom(shap_values, 0, 20)

#%%
from src.modeling.models import LogisticRegressionModel
from sklearn.model_selection import train_test_split

#%%
lrm = LogisticRegressionModel(df)
lrm.run_optuna()


#%%
xtrain, xval, ytrain, yval = train_test_split(df["text"], df["label"])
lrm.refit_best_model(xtrain, ytrain)

#%%
print(classification_report(yval, lrm.model.predict(xval)))
print(confusion_matrix(yval, lrm.model.predict(xval)))
