#%%
import io

import pandas as pd
import torchtext

#%%
torchtext.data.functional.generate_sp_model(
    "data/raw/OLD_dataset.csv",
    vocab_size=10_000,
    model_type="bpe",
    model_prefix="data/xyz",
)
sp_model = torchtext.data.functional.load_sp_model("m_user.model")
tokenizer = torchtext.data.functional.sentencepiece_tokenizer(sp_model=sp_model)

import numpy as np

#%%
import torch

list(tokenizer(["hello world", "looong $AAPL stock at $123 89P +17% YTD.! 😂😂😂"]))
