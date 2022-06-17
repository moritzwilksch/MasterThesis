#%%
import pandas as pd
import torch

#%%
representations = []
for idx in range(20):
    print(f"Processing batch #{idx}...")
    representations.append(torch.load(f"data/representations_{idx}.pt"))

#%%
X = torch.vstack(representations).detach().numpy()
y = pd.read_parquet("data/labeled/labeled_tweets.parquet")["label"].replace("0", "2")


#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

cross_val_score(
    # LogisticRegression(max_iter=350, random_state=42, solver="sag", C=1.7),
    GradientBoostingClassifier(n_estimators=100, min_samples_leaf=3),
    X,
    y,
    cv=5,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
    n_jobs=-1,
)


#%%
model = LogisticRegression(
    max_iter=350, C=3, random_state=42, n_jobs=-1, solver="liblinear"
)
model.fit(X, y)

#%%
preds = model.predict_proba(X)

#%%
roc_auc_score(y, preds, multi_class="ovr")
