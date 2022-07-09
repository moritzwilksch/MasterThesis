#%%
from random import setstate
import joblib
import pandas as pd
import polars as pl
from src.ml_modeling.models import LogisticRegressionModel
from src.utils.preprocessing import Preprocessor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import Colors, set_style
import numpy as np

set_style()

#%%
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
df = prepper.process(df)


# TODO: should we merge labels? drop 2?
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)


df = df.to_pandas()

#%%
pipe = LogisticRegressionModel(None, None).get_pipeline()
pipe.set_params(**LogisticRegressionModel.FINAL_BEST_PARAMS)

#%%

preds = cross_val_predict(
    pipe,
    df["text"],
    df["label"],
    n_jobs=-1,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
)


#%%
print(classification_report(df["label"], preds))
print(confusion_matrix(df["label"], preds, normalize="pred"))

#%%
mtx = pd.DataFrame(confusion_matrix(df["label"], preds, normalize="pred"))
s_mtx = mtx.style.apply(
    lambda s: np.where(s == s.max(), "background-color:#DDDDDD", ""), axis=1
)
s_mtx


#%%
print(s_mtx.to_latex(convert_css=True))

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


cr = classification_report(
    df["label"], preds, output_dict=True, target_names=["POS", "NEU", "NEG"]
)
print(classification_report_to_tex(cr))

#%%
cr = classification_report(df["label"], preds)
cr = cr.replace(" " * 6, " & ").replace("\n", "\\\\ \n")
cr = "\n".join(line for line in cr.split("\n") if line != "\\\\ ")
print(cr)


#%%
def confusion_matrix_to_tex(mtx):
    CATS = ["POS", "NEU", "NEG"]
    lines = []

    for idx, cat in enumerate(CATS):
        line = []
        line.append(f"\\textbf{{{cat}}}")
        for elem in mtx[idx, :]:
            line.append(f"{elem:.0%}")
        lines.append(" & ".join(line))

    return " \\\\\n".join(l.replace("%", "\\%") for l in lines) + " \\\\"


print(confusion_matrix_to_tex(confusion_matrix(df["label"], preds, normalize="true")))

#%%
df["prediction"] = preds
# wrong = df.query("(label=='3' & prediction=='1') | (label=='1' & prediction=='3')")
wrong = df.query("(label=='3' & prediction=='1')")
# wrong = df.query("label != prediction")
wrong

#%%
