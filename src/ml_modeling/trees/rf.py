#%%
import polars as pl
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.utils.preprocessing import Preprocessor

#%%

pyfin_senti_data = pl.read_parquet("data/labeled/labeled_tweets.parquet")
prepper = Preprocessor()
pyfin_senti_data = prepper.process(pyfin_senti_data)


# TODO: should we merge labels? drop 2?
pyfin_senti_data = pyfin_senti_data.with_column(
    pl.when(pl.col("label") == "0")
    .then(pl.lit("2"))
    .otherwise(pl.col("label"))
    .alias("label")
)


pyfin_senti_data = pyfin_senti_data.to_pandas()

#%%

model = Pipeline(
    [
        ("vectorizer", TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100, min_samples_leaf=3, class_weight="balanced"
            ),
        ),
    ]
)

cross_val_score(
    model,
    pyfin_senti_data["text"],
    pyfin_senti_data["label"],
    n_jobs=-1,
    scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
)
