from abc import ABC

import numpy as np
import optuna
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


class LogisticRegressionModel:
    def __init__(self, train_val_data):
        self.train_val_data = train_val_data
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        self.study = optuna.delete_study(
            storage="sqlite:///tuning/optuna.db",
            study_name="LogisticRegression",
        )
        self.study = optuna.create_study(
            storage="sqlite:///tuning/optuna.db",
            study_name="LogisticRegression",
            direction="maximize",
            load_if_exists=True,
        )

    def get_pipeline(self):
        """Returns pipeline"""
        return Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(),  # params will be set later
                ),
                (
                    "model",
                    LogisticRegression(random_state=42, n_jobs=-1, max_iter=250),
                ),
            ]
        )

    def refit_best_model(self, refit_xtrain, refit_ytrain):
        """Fits model to data"""

        self.model = self.get_pipeline()

        params = self.study.best_params.copy()
        if params["vectorizer__analyzer"] == "word":
            params["vectorizer__ngram_range"] = (1, params["vectorizer__ngram_range"])

        else:
            params["vectorizer__ngram_range"] = (
                params["vectorizer__ngram_range"],
                params["vectorizer__ngram_range"],
            )

        self.model.set_params(**params)

        self.model.fit(refit_xtrain, refit_ytrain)

    def run_optuna(self, n_trials: int = 100):
        """Run a hyperparameter optimization"""
        self.study.optimize(self.optuna_trial, n_trials=n_trials)

    def optuna_trial(self, trial):
        """One Optuna trial run"""
        params = {
            "model__C": trial.suggest_loguniform("model__C", 1e-5, 100),
            "vectorizer__analyzer": trial.suggest_categorical(
                "vectorizer__analyzer", ["char_wb", "word"]
            ),
        }

        params = self.add_ngram_range_tuple_to_params(trial, params)

        pipe = self.get_pipeline()
        pipe.set_params(**params)  # set chosen hyperparameters

        score = cross_val_score(
            pipe,
            X=self.train_val_data["text"],
            y=self.train_val_data["label"],
            cv=self.kfold,
            scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
            n_jobs=5,
        )
        return score.mean()

    def add_ngram_range_tuple_to_params(self, trial, params):
        if params["vectorizer__analyzer"] == "word":
            params["vectorizer__ngram_range"] = trial.suggest_int(
                "vectorizer__ngram_range", 1, 3
            )
            params["vectorizer__ngram_range"] = (1, params["vectorizer__ngram_range"])

        else:
            params["vectorizer__ngram_range"] = trial.suggest_int(
                "vectorizer__ngram_range", 3, 6
            )
            params["vectorizer__ngram_range"] = (
                params["vectorizer__ngram_range"],
                params["vectorizer__ngram_range"],
            )
        return params
