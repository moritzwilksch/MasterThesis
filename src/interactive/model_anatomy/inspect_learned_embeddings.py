#%%
import time

import numpy as np
import polars as pl
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity

from src.dl_modeling.data import (BERTTensorDataModule, BertTensorDataSet,
                                  TweetDataModule)
from src.dl_modeling.models import (BERTSAModel, RecurrentSAModel,
                                    TransformerSAModel)
from src.utils.preprocessing import Preprocessor

#%%
USE_MODEL = "transformer"

prepper = Preprocessor()
if USE_MODEL == "transformer":
    data = TweetDataModule(split_idx="retrain", batch_size=64, model_type="transformer")
    model = TransformerSAModel.load_from_checkpoint(
        "outputs/models/transformer_final.ckpt"
    )
    model.eval()
elif USE_MODEL == "recurrent":
    data = TweetDataModule(split_idx="retrain", batch_size=64, model_type="recurrent")
    model = RecurrentSAModel.load_from_checkpoint("outputs/models/gru_final.ckpt")
elif USE_MODEL == "bert":
    data = BERTTensorDataModule(
        split_idx="retrain", data_prefix="prep_pyfin", batch_size=64
    )
    model = BERTSAModel.load_from_checkpoint("outputs/models/bert_final.ckpt")
model.eval()

#%%
embs = model.embedding.weight.detach().numpy()

sims = cosine_similarity(embs, embs)

with open("outputs/tokenizers/retraining_trainval.vocab") as f:
    vocab = [line.split("\t")[0] for line in f.readlines()]

#%%


def get_nearest(sims, word):
    df = prepper.process(pl.DataFrame({"text": [word]}))
    token = np.array(list(data.tokenizer(df["text"])))

    nearest = sims[token].ravel().argsort()[:-10:-1]
    print(", ".join(np.array(vocab)[nearest].ravel()))


get_nearest(sims, "down")

#%%
