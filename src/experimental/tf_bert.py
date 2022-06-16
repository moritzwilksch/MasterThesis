#%%
import torch
from transformers import DistilBertModel, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

#%%
tokens = tokenizer(
    ["hello world", "this is a short test"],
    return_tensors="pt",
    truncation=True,
    padding=True,
)
tokens


#%%
model(tokens)
