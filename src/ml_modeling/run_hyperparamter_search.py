#%%
from enum import Enum

import numpy as np
import pandas as pd
import polars as pl
from src.ml_modeling.models import SVMModel
from src.utils.db import get_client
from src.utils.preprocessing import Preprocessor
from src.ml_modeling.experiment import Experiment

DB = get_client()

# TODO: replace by loading from parquet file once labeling is done
df = pl.from_dicts(
    list(
        DB.thesis.labeled_tweets.find(
            {"label": {"$ne": ""}},
            projection={"text": True, "label": True, "_id": False, "id": True},
        )
    )
)

# pre-processing
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
experiment = Experiment("SVMModel", SVMModel, df)
# experiment.run(n_trials=100)

#%%
val_scores, test_scores, best_params, times_taken = experiment.load()
