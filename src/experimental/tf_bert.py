#%%
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

# bert = pipeline("feature-extraction", "bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#%%
import time

tic = time.perf_counter()
tokens = tokenizer(
    [
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
        "tesla battery supplier catl plans $9 billion share sale to boost capacity ticker #xglobalmarkets #tesla ticker",
        "insider alice l walton reports selling 999,999 shares of ticker for a total cost of $99,999,999.99 #fntl",
    ]
    * 4,
    return_tensors="pt",
    padding=True,
    truncation=True,
)
out = model(**tokens)
tac = time.perf_counter()
print(tac - tic)
tokens
#%%

#%%
model(tokens)
