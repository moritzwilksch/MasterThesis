#%%
import os

import polars as pl
import torchtext
from sklearn.model_selection import KFold, train_test_split

from src.utils.preprocessing import Preprocessor

all_data = pl.read_parquet("data/labeled/labeled_tweets.parquet")
all_data = all_data.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .cast(pl.Int32)
    .alias("label")
).with_column(pl.col("label") - 1)

prepper = Preprocessor()
all_data = prepper.process(all_data).to_pandas()



# ###################################
# import spacy
# nlp = spacy.load("en_core_web_sm", exclude=["parser", "ner", "textcat"])
# not_stop_words = [
#     "up",
#     "down",
#     "above",
#     "below",
#     "against",
#     "between",
#     "bottom",
#     "top",
#     "call",
#     "put",
#     "least",
#     "most",
#     "much",
#     "n't",
#     "off",
#     "under",
#     "over"
# ]
# for elem in not_stop_words:
#     try:
#         nlp.Defaults.stop_words.remove(elem)
#     except KeyError:
#         pass

# preprocessed_docs = []
# for doc in nlp.pipe(all_data["text"].to_list(), batch_size=64, n_process=-1):
#     preprocessed_docs.append(" ".join(w.text for w in doc))
# all_data["text"] = preprocessed_docs
# ###################################









xtrainval, xtest, ytrainval, ytest = train_test_split(
    all_data["text"],
    all_data["label"],
    shuffle=True,
    random_state=42,
    test_size=0.25,  # hold-out test set
)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for idx, (train_idx, val_idx) in enumerate(kfold.split(xtrainval)):
    xtrainval.iloc[train_idx].to_csv("data/temp.csv", index=False, header=False)
    torchtext.data.functional.generate_sp_model(
        "data/temp.csv",
        vocab_size=3_000,
        model_type="unigram",  # outperforms BPE
        model_prefix=f"outputs/tokenizers/split_{idx}",
    )

# for re-training:
xtrainval.to_csv("data/temp.csv", index=False, header=False)
torchtext.data.functional.generate_sp_model(
    "data/temp.csv",
    vocab_size=3_000,
    model_type="unigram",  # outperforms BPE
    model_prefix=f"outputs/tokenizers/retraining_trainval",
)
os.remove("data/temp.csv")
