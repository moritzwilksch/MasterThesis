#%%
import snorkel

import pandas as pd
from src.utils.db import get_client

DB = get_client()

# TODO: replace by loading from parquet file once labeling is done
df = pd.DataFrame(
    list(
        DB.thesis.labeled_tweets.find(
            {"label": {"$ne": ""}},
            projection={"text": True, "label": True, "_id": False},
        )
    )
)

#%%
test_idx = df.sample(frac=0.1, random_state=42).index

df_train = df[~df.index.isin(test_idx)].copy().reset_index(drop=True)
df_test = df[df.index.isin(test_idx)].copy().reset_index(drop=True).drop("label", axis=1)

#%%
from snorkel.labeling import labeling_function

@labeling_function()
def short(x):
    return 3 if "short $" in x.text.lower() else -1

@labeling_function()
def bear(x):
    return 3 if "bearish" in x.text.lower() else -1

@labeling_function()
def sell(x):
    return 3 if "selling" in x.text.lower() else -1

@labeling_function()
def sold(x):
    return 3 if "sold" in x.text.lower() else -1


@labeling_function()
def buying(x):
    return 1 if "buying" in x.text.lower() else -1

@labeling_function()
def bought(x):
    return 1 if "bought" in x.text.lower() else -1

@labeling_function()
def long_(x):
    return 1 if "long" in x.text.lower() else -1

@labeling_function()
def calls(x):
    return 1 if "calls" in x.text.lower() else -1


@labeling_function()
def interesting(x):
    return 0 if "interesting" in x.text.lower() else -1


@labeling_function()
def spam1(x):
    return 2 if "chat" in x.text.lower() else -1

@labeling_function()
def spam2(x):
    return 2 if "join " in x.text.lower() else -1


from snorkel.labeling import PandasLFApplier

lfs = [short, bear, sell, sold, buying, bought, long_, calls, interesting, spam1, spam2]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)

#%%
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)

#%%
from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=4, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

#%%
from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter(cardinality=4)
preds_train = majority_model.predict(L=L_train)
#%%
majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")[
    "accuracy"
]
print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.1f}%")