#%%
from numpy import isin
from src.ml_modeling.models import LogisticRegressionModel, SVMModel
from src.dl_modeling.models import RecurrentSAModel, TransformerSAModel, BERTSAModel

#%%
models = [
    LogisticRegressionModel,
    SVMModel,
    RecurrentSAModel,
    TransformerSAModel,
    BERTSAModel,
]
data = dict()
for model in models:
    try:
        data[model.__name__] = model.BEST_PARAMS
    except AttributeError:
        data[model.__name__] = model.FINAL_BEST_PARAMS

#%%
all_params = [
    "vectorizer__analyzer",
    "vectorizer__ngram_range",
    "vectorizer__min_df",
    "model__C",
    "model__kernel",
    "model__degree",
    "token_dropout",
    "embedding_dim",
    "hidden_dim",
    "dropout",
    "gru_hidden_dim",
    "dim_ff",
]

#%%
def texify(s: str):
    return s.replace("__", "_").replace("_", "\_")

#%%
lines = []
for param in all_params:
    current_line = [param]
    for model in models:
        value = data.get(model.__name__).get(param, "--")

        if isinstance(value, float):
            value = round(value, 3) if value > 0.001 else f"{value:.2e}"
        current_line.append(str(value))
    
    lines.append(" & ".join(current_line) + " \\\\")

for line in lines:
    if "token" in line and "dropout" in line:
        print("\\midrule")
    print(texify(line))

