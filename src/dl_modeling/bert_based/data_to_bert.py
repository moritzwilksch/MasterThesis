#%%
import time

import numpy as np
import polars as pl
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

from src.dl_modeling.data import TweetDataModule
from src.utils.preprocessing import Preprocessor

model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

import pandas as pd

df = pd.read_parquet("data/labeled/labeled_tweets.parquet")

# finsome = pd.read_json("data/finSoMe/fin-SoMe.json")
# finsome = pd.DataFrame(
#     {
#         "text": finsome["tweet"],
#         "label": finsome["market_sentiment"].map(
#             {"Unsure": 2, "Bearish": 3, "Bullish": 1}
#         ),
#     }
# )

# prepper = Preprocessor()
# finsome = prepper.process(pl.from_pandas(finsome)).to_pandas()
# df = prepper.process(pl.from_pandas(df)).to_pandas()
#%%

### ATTENTION vvvv
# df = finsome

for idx, batch in enumerate(np.array_split(df, 20)):
    print(f"Processing batch #{idx}...")
    texts = batch["text"].to_list()

    tic = time.perf_counter()
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    out = model(**tokens).last_hidden_state
    representations = torch.mean(out, dim=1)
    tac = time.perf_counter()
    print(f"Time taken: {tac - tic} for {len(texts)} texts")
    # torch.save(representations, f"data/distilbert/noprep_pyfin_{idx}.pt")
