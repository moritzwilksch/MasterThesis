#%%
import os

import polars as pl
import torchtext
from sklearn.model_selection import KFold, train_test_split

from src.utils.preprocessing import Preprocessor

all_data = pl.read_parquet("data/labeled/labeled_tweets.parquet")
all_data = all_data.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .cast(pl.Int32)
    .alias("label")
).with_column(pl.col("label") - 1)

prepper = Preprocessor()
all_data = prepper.process(all_data).to_pandas()

xtrainval, xtest, ytrainval, ytest = train_test_split(
    all_data["text"],
    all_data["label"],
    shuffle=True,
    random_state=42,
    test_size=0.25,  # hold-out test set
)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for idx, (train_idx, val_idx) in enumerate(kfold.split(xtrainval)):
    xtrainval.iloc[train_idx].to_csv("data/temp.csv", index=False, header=False)
    torchtext.data.functional.generate_sp_model(
        "data/temp.csv",
        vocab_size=3_000,
        model_type="unigram",  # outperforms BPE
        model_prefix=f"outputs/tokenizers/split_{idx}",
    )
os.remove("data/temp.csv")
