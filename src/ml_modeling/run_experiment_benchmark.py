#%%
from enum import Enum

import numpy as np
import pandas as pd
import polars as pl
from black import Mode

from src.ml_modeling.experiment import (Experiment, FinBERTBenchmark,
                                        NTUSDMeBenchmark,
                                        TwitterRoBERTaBenchmark,
                                        VaderBenchmark)
from src.ml_modeling.models import LogisticRegressionModel, SVMModel
from src.utils.db import get_client
from src.utils.preprocessing import Preprocessor

DB = get_client()

# TODO: replace by loading from parquet file once labeling is done
pyfin_senti_data = pl.from_dicts(
    list(
        DB.thesis.labeled_tweets.find(
            {"label": {"$ne": ""}},
            projection={"text": True, "label": True, "_id": False, "id": True},
        )
    )
)

# pre-processing
prepper = Preprocessor()
pyfin_senti_data = prepper.process(pyfin_senti_data)


# TODO: should we merge labels? drop 2?
pyfin_senti_data = pyfin_senti_data.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)


pyfin_senti_data = pyfin_senti_data.to_pandas()


finsome = pd.read_json("data/finSoMe/fin-SoMe.json")
finsome = pd.DataFrame(
    {
        "text": finsome["tweet"],
        "label": finsome["market_sentiment"].map(
            {"Unsure": 2, "Bearish": 3, "Bullish": 1}
        ),
    }
)

prepper = Preprocessor()
finsome = prepper.process(pl.from_pandas(finsome)).to_pandas()


#%%
class Dataset(Enum):
    FINSOME = "finsome"
    PYFIN_SENTI = "pyfin_sentiment_data"


class Model(Enum):
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    VADER = "vader"
    FINBERT = "finbert"
    TWITTER_ROBERTA = "twitter_roberta"
    NTUSD = "ntusd"


########################
DATASET = Dataset.PYFIN_SENTI
MODEL = Model.SVM
########################

if DATASET == Dataset.FINSOME:
    data = finsome
else:
    data = pyfin_senti_data

print(f"Benchmark: {MODEL.value} on {DATASET.value}")

if MODEL == Model.LOGISTIC_REGRESSION:
    experiment = Experiment("LogisticRegression", LogisticRegressionModel, data)
    if DATASET == DATASET.PYFIN_SENTI:
        experiment.fit_final_best_model(data)
        val_scores, test_scores, best_params, times_taken = experiment.load()
    else:
        test_scores = experiment.apply_to_other_data(finsome)
        times_taken = "N/A"

if MODEL == Model.SVM:
    experiment = Experiment("SVM", SVMModel, data)
    if DATASET == DATASET.PYFIN_SENTI:
        experiment.fit_final_best_model(data)
        val_scores, test_scores, best_params, times_taken = experiment.load()
    else:
        test_scores = experiment.apply_to_other_data(finsome)
        times_taken = "N/A"

if MODEL == Model.VADER:
    vaderbenchmark = VaderBenchmark(data)
    test_scores, times_taken = vaderbenchmark.load()

if MODEL == Model.FINBERT:
    finbertbenchmark = FinBERTBenchmark(data)
    test_scores, times_taken = finbertbenchmark.load()

if MODEL == Model.TWITTER_ROBERTA:
    twitter_roberta_benchmark = TwitterRoBERTaBenchmark(data)
    test_scores, times_taken = twitter_roberta_benchmark.load()

if MODEL == Model.NTUSD:
    ntusd = NTUSDMeBenchmark(data, path_to_ntusd_folder="data/NTUSD-Fin")
    test_scores, times_taken = ntusd.load()


print(test_scores)
print(times_taken)
