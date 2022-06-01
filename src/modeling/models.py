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
        # self.study = optuna.delete_study(
        #     storage="sqlite:///tuning/optuna.db",
        #     study_name="LogisticRegression",
        # )
        self.study = optuna.create_study(
            storage="sqlite:///tuning/optuna.db",
            study_name="LogisticRegression",
            direction="maximize",
            load_if_exists=True,
        )

    def get_pipeline(self, params, prep_params):
        """Returns pipeline"""
        return Pipeline(
            [
                (
                    "vectorizer",
                    TfidfVectorizer(
                        analyzer=prep_params["analyzer"],
                        ngram_range=prep_params["ngram_range"],
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        **params, random_state=42, n_jobs=-1, max_iter=250
                    ),
                ),
            ]
        )

    def refit_best_model(self, refit_xtrain, refit_ytrain):
        """Fits model to data"""
        params = {"C": 1.3888279023427723}
        prep_params = {"analyzer": "char_wb", "ngram_range": (4, 4)}

        self.model = self.get_pipeline(params, prep_params)
        self.model.fit(refit_xtrain, refit_ytrain)

    def run_optuna(self, n_trials: int = 100):
        """Run a hyperparameter optimization"""
        self.study.optimize(self.optuna_trial, n_trials=n_trials)

    def optuna_trial(self, trial):
        """One Optuna trial run"""
        params = {
            "C": trial.suggest_loguniform("C", 1e-5, 100),
        }

        prep_params = {
            "analyzer": trial.suggest_categorical("analyzer", ["char_wb", "word"]),
        }

        if prep_params["analyzer"] == "word":
            prep_params["ngram_range"] = trial.suggest_int("ngram_range", 1, 3)
            prep_params["ngram_range"] = (1, prep_params["ngram_range"])

        else:
            prep_params["ngram_range"] = trial.suggest_int("ngram_range", 3, 6)
            prep_params["ngram_range"] = (
                prep_params["ngram_range"],
                prep_params["ngram_range"],
            )

        pipe = self.get_pipeline(params, prep_params)

        score = cross_val_score(
            pipe,
            X=self.train_val_data["text"],
            y=self.train_val_data["label"],
            cv=self.kfold,
            scoring=make_scorer(roc_auc_score, needs_proba=True, multi_class="ovr"),
            n_jobs=5,
        )
        return score.mean()
        # aucs = []
        # for train_idx, val_idx in self.kfold.split(self.train_val_data):
        #     # model definition
        #     model = LogisticRegression(
        #         **params, random_state=42, n_jobs=-1, max_iter=250
        #     )

        #     # data split definition
        #     xtrain, xval = (
        #         self.train_val_data.loc[train_idx, "text"],
        #         self.train_val_data.loc[val_idx, "text"],
        #     )
        #     ytrain, yval = (
        #         self.train_val_data.loc[train_idx, "label"],
        #         self.train_val_data.loc[val_idx, "label"],
        #     )

        #     # preprocessing
        #     self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4))
        #     xtrain = self.vectorizer.fit_transform(xtrain)
        #     xval = self.vectorizer.transform(xval)

        #     # fit & eval
        #     model.fit(xtrain, ytrain)
        #     aucs.append(roc_auc_score(yval, model.predict_proba(xval), multi_class="ovr"))
