#%%
import string

import nltk
import numpy as np
import pandas as pd
import polars as pl
from nltk.corpus import stopwords
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import TransformerSAModel
from src.utils.preprocessing import Preprocessor
import numpy as np

stopwords = list(stopwords.words("english"))

#%%
df = pl.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
df = prepper.process(df)
df = df.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)

data = TweetDataModule(split_idx="retrain", batch_size=64, model_type="transformer")
model = TransformerSAModel.load_from_checkpoint("outputs/models/transformer_final.ckpt")
model.eval()

#%%
df_as_tokens = list(data.tokenizer(df["text"]))
X = []
for row in df_as_tokens:
    X.append(model.embedding.weight[row].detach().numpy().mean(axis=0))
X = np.vstack(X)

#%%
xtrain, xtest, ytrain, ytest = train_test_split(
    X,
    df["label"].to_pandas(),
    shuffle=True,
    random_state=42,
    test_size=0.25,  # hold-out test set
)

#%%
lr = LogisticRegression(C=2)
lr.fit(xtrain, ytrain)

preds = lr.predict_proba(xtest)
print(roc_auc_score(ytest, preds, multi_class="ovr"))
