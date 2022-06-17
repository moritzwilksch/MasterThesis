#%%
import torch
from transformers import pipeline, AutoModel, AutoTokenizer

# bert = pipeline("feature-extraction", "bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
tokens = tokenizer(
    [
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
    ],
    return_tensors="pt",
    padding=True,
    truncation=True,
)

tokens
#%%
out = model(**tokens)

#%%
model(tokens)
