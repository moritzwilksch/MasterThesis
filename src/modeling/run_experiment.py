#%%
import numpy as np
import polars as pl

from src.modeling.experiment import Experiment, VaderBenchmark
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
experiment01 = Experiment("LogisticRegression", LogisticRegressionModel, df)
# experiment01.run()
# val_scores, test_scores, best_params = experiment01.load()
model = experiment01.fit_final_best_model(df)
preds = model.predict(df["text"])
probas = model.predict_proba(df["text"])

#%%
mask = probas.max(axis=1) > 0.8
preddf = df.assign(pred=preds)[mask].loc[lambda d: d["pred"] != d["label"]]
preddf.style.set_properties(subset=['text'], **{'width': '500px'})

#%%
preddf["id"].to_list()
#%%

vaderbenchmark = VaderBenchmark(df)
vaderbenchmark.load()

#%%
print(val_scores)
print(test_scores)
print(best_params)
