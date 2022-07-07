#%%
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import projections
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)
from sklearn.metrics.pairwise import cosine_similarity

from src.dl_modeling.data import (BERTTensorDataModule, BertTensorDataSet,
                                  TweetDataModule)
from src.dl_modeling.models import (BERTSAModel, RecurrentSAModel,
                                    TransformerSAModel)
from src.utils.plotting import Colors, scale_lightness, set_style
from src.utils.preprocessing import Preprocessor

set_style()
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

# load vocab + embeddings
with open("outputs/tokenizers/retraining_trainval.vocab") as f:
    vocab = [line.split("\t")[0] for line in f.readlines()]
embs = model.embedding.weight.detach().numpy()

#%%
sims = cosine_similarity(embs, embs)


def get_nearest(sims, word):
    df = prepper.process(pl.DataFrame({"text": [word]}))
    token = np.array(list(data.tokenizer(df["text"])))

    nearest = sims[token].ravel().argsort()[:-10:-1]
    print(", ".join(np.array(vocab)[nearest].ravel()))


# get_nearest(sims, "down")

import umap
#%%
from sklearn.manifold import TSNE


def visualize_embeddings(embs, projector: str):
    positive = [
        "up",
        "green",
        "long",
        "bullish",
        "call",
        "buy",
        "+9.9",
    ]
    negative = ["sell", "bearish", "put", "short", "red", "down", "-9.99"]
    neutral = ["sure", "chat", "question", "bitcoin", "ceo", "$999"]

    words = positive + negative + neutral

    df = prepper.process(pl.DataFrame({"text": words}))
    tokens = np.array(list(data.tokenizer(df["text"])))
    if not tokens.shape == (len(words), 1):
        raise ValueError(f"One word evaluated to more than one token! {tokens =}")

    chosen_embeddings = embs[tokens].squeeze()

    if projector == "tsne":
        # low_dim_embeddings = TSNE().fit_transform(embs)
        # joblib.dump(low_dim_embeddings, "outputs/dump/low_dim_tsne.joblib")
        low_dim_embeddings = joblib.load("outputs/dump/low_dim_tsne.joblib")
    elif projector == "umap":
        # low_dim_embeddings = umap.UMAP().fit_transform(embs)
        # joblib.dump(low_dim_embeddings, "outputs/dump/low_dim_umap.joblib")
        low_dim_embeddings = joblib.load("outputs/dump/low_dim_umap.joblib")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = (
        [Colors.GREEN.value] * len(positive)
        + [Colors.RED.value] * len(negative)
        # + ["0.4"] * len(neutral)
        + [Colors.DARKBLUE.value] * len(neutral)
    )

    ax.scatter(
        low_dim_embeddings[tokens, 0],
        low_dim_embeddings[tokens, 1],
        # color=Colors.DARKBLUE.value,
        c=colors,
        ec=[scale_lightness(sns.desaturate(c, 1), 0.5) for c in colors],
    )

    _position_override = {
        "put": (0, -0.25),
        "down": (0, -0.25),
        "call": (0, -0.2),
        "bullish": (-0.275, 0),
        "green": (0.25, 0),
        "buy": (-0.17, -0.1),
        "down": (0.25, 0),
        "-9.99": (0, -0.3),
        "red": (0.05, 0.05),
    }

    _va_center = ["green", "call", "bullish", "down"]

    for idx, text in enumerate(words):
        if text not in _position_override:
            xy = (
                low_dim_embeddings[tokens[idx], 0],
                low_dim_embeddings[tokens[idx], 1] + 0.1,
            )
        else:
            xy = (
                low_dim_embeddings[tokens[idx], 0] + _position_override[text][0],
                low_dim_embeddings[tokens[idx], 1] + _position_override[text][1],
            )

        ax.annotate(
            text,
            xy,
            size=14,
            ha="center",
            va="center" if text in _va_center else "bottom",
        )

    # ax.autoscale(False)
    # ax.set_xlim(4, 9)
    # ax.set_ylim(2, 7)
    ax.margins(0.1)
    ax.set_xlabel("Dimension 0", weight="bold")
    ax.set_ylabel("Dimension 1", weight="bold")

    sns.despine()
    fig.tight_layout()
    fig.savefig("outputs/plots/word_embeddings.pdf", bbox_inches="tight")


visualize_embeddings(
    embs,
    projector="umap",
)

# %%
