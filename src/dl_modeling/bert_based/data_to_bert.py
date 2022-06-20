#%%
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

from src.dl_modeling.data import TweetDataModule

model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

import pandas as pd

df = pd.read_parquet("data/labeled/labeled_tweets.parquet")


#%%

for idx, batch in enumerate(np.array_split(df, 20)):
    print(f"Processing batch #{idx}...")
    texts = batch["text"].to_list()

    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    out = model(**tokens).last_hidden_state
    representations = out[:, 0, :] #torch.mean(out, dim=1)
    torch.save(representations, f"data/distilbert/clstoken_representations_{idx}.pt")
