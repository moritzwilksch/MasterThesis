#%%
import numpy as np
import polars as pl

from src.modeling.experiment import (Experiment, FinBERTBenchmark,
                                     TwitterRoBERTaBenchmark, VaderBenchmark)
from src.modeling.models import LogisticRegressionModel
from src.utils.db import get_client
from src.utils.preprocessing import Preprocessor

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
# experiment01 = Experiment("LogisticRegression", LogisticRegressionModel, df)
# # experiment01.run(n_trials=100)
# val_scores, test_scores, best_params, times_taken = experiment01.load()
# print(test_scores)
# print(times_taken)

#%%

# vaderbenchmark = VaderBenchmark(df)
# test_scores, times_taken = vaderbenchmark.load()
# print(test_scores)
# print(times_taken)

#%%
finbertbenchmark = FinBERTBenchmark(df)
test_scores, times_taken = finbertbenchmark.load()
print(test_scores)
print(times_taken)

#%%
# twitter_roberta_benchmark = TwitterRoBERTaBenchmark(df)
# test_scores, times_taken = twitter_roberta_benchmark.load()
# print(test_scores)
# print(times_taken)


#%%
# print(val_scores)
# print(test_scores)
# print(best_params)
