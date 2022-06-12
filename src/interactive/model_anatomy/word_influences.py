#%%
import joblib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

#%%
model: Pipeline = joblib.load("outputs/models/final_LogisticRegressionModel.gz")

#%%

top_per_class = model["model"].coef_.argsort(axis=1)[:, -15:]
idx_to_token_map = {v: k for k, v in model["vectorizer"].vocabulary_.items()}

for row in top_per_class:
    for entry in row:
        print(idx_to_token_map[entry], end=" | ")
    print()
    print("-" * 80)
