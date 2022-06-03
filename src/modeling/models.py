from abc import ABC, abstractmethod

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


class BaseSklearnSAModel(ABC):
    """
    Abstract base class for all sklearn-based SA models

    Args:
        name (str): _description_
        split_idx (int): _description_
        train_val_data (pd.DataFrame): _description_
    """

    def __init__(
        self,
        name: str,
        split_idx: int,
        train_val_data: pd.DataFrame,
    ):
        self.name = name
        self.train_val_data = train_val_data
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        self.study = optuna.create_study(
            storage="sqlite:///tuning/optuna.db",
            study_name=f"{self.name}-split-{split_idx}",
            direction="maximize",
            load_if_exists=True,
        )

    @abstractmethod
    def optuna_trial(self, trial: optuna.Trial) -> float:
        """One optuna trial which suggests HPs and returns a score

        Args:
            trial (optuna.Trial): optuna trial object

        Returns:
            float: score for this trial
        """
        pass

    @abstractmethod
    def get_pipeline(self) -> Pipeline:
        """Returns pipeline that can be fit to a text column of a dataframe

        Returns:
            Pipeline: sklearn pipeline that can be fit to a text column of a dataframe
        """
        pass

    def run_optuna(self, n_trials: int = 100) -> None:
        """Runs a hyper param sweep on train_val_data

        Args:
            n_trials (int, optional): number of trials. Defaults to 100.
        """
        self.study.optimize(self.optuna_trial, n_trials=n_trials)

    def refit_best_model(self, refit_xtrain: pd.DataFrame, refit_ytrain: pd.DataFrame):
        """Refit best model from this study to given data

        Args:
            refit_xtrain (pd.DataFrame): X
            refit_ytrain (pd.DataFrame): y
        """
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


class LogisticRegressionModel(BaseSklearnSAModel):
    def __init__(self, split_idx, train_val_data):
        # self.study = optuna.delete_study(
        #     storage="sqlite:///tuning/optuna.db",
        #     study_name="LogisticRegression",
        # )

        super().__init__(
            "LogisticRegression", split_idx=split_idx, train_val_data=train_val_data
        )

    def get_pipeline(self):
        """Overrides BaseSklearnSAModel.get_pipeline()"""

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

    def optuna_trial(self, trial):
        """Overrides BaseSklearnSAModel.optuna_trial()"""
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