#%%
import joblib
import pandas as pd
import polars as pl

from src.utils.preprocessing import Preprocessor

pipe = joblib.load("outputs/models/final_LogisticRegressionModel.gz")
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
from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import TransformerSAModel
import torch
import torch.nn.functional as F
import numpy as np

prepper = Preprocessor()

data = TweetDataModule(split_idx="retrain", batch_size=64, model_type="transformer")
model = TransformerSAModel.load_from_checkpoint("outputs/models/transformer_final.ckpt")
model.eval()


def predict_one(s: str):
    if len(s) == 0 or set(s) == set(" "):
        return np.array([[0, 1, 0]])
    df = prepper.process(pl.DataFrame({"text": s}))
    x = torch.Tensor(list(data.tokenizer(df["text"]))[0]).long().reshape(-1, 1)
    logits = model(x, masks=torch.full((len(x),), False).unsqueeze(0))
    return F.softmax(logits, dim=1).detach().numpy()


def predict(arr):
    preds = []
    for elem in arr:
        preds.append(predict_one(elem))
    preds = np.vstack(preds)
    return preds / preds.sum(axis=1, keepdims=True)


predict(["long", "short"])

#%%
predict(["long", "short", "", " ", "hello this is a longer test 00", "ÖÖÖ"]).sum(axis=1)

#%%
import eli5
from eli5.lime import TextExplainer

te = TextExplainer(random_state=42, char_based=False)
te.fit("long ticker i'm buying this is green", pipe.predict_proba)
te.explain_prediction()
