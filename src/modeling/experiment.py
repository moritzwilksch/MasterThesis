from abc import ABC, abstractmethod

import pandas as pd
from rich.prompt import Prompt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.modeling.models import BaseSklearnSAModel, LogisticRegressionModel


class Experiment:
    """An experiment represents a nested CV run of a model."""

    def __init__(
        self, study_name: str, model: BaseSklearnSAModel, all_data: pd.DataFrame
    ):
        """Init.

        Args:
            study_name (str): name of the optuna study
            model (BaseSklearnSAModel): model to use
            all_data (pd.DataFrame): data frame of train and val data for inner CV
        """
        self.study_name = study_name
        self.model = model
        self.data = all_data
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    def run(self) -> None:
        """Runs the hyperparameter search for each outer split. Holds out test data."""
        answer = Prompt.ask(
            "Do you really want to re-run the entire hyperparameter sweep?",
            choices=["y", "n"],
        )

        if answer != "y":
            print("Exiting.")
            exit(0)

        for split_idx, (train_val_idx, _) in enumerate(self.kfold.split(self.data)):
            print(f"Starting trial #{split_idx}...")
            train_val_data = self.data.iloc[train_val_idx]
            # test data not needed here, metrics are calculated in .load()

            sa_model = self.model(split_idx=split_idx, train_val_data=train_val_data)

            sa_model.run_optuna(n_trials=10)

    def load(self) -> tuple[list, list, list]:
        """Loads and refits best model for each split and evaluates it on outer test data

        Returns:
            tuple[list, list, list]: val_scores, test_scores, best_params
        """
        val_scores = []
        test_scores = []
        best_params = []

        for split_idx, (train_idx, test_idx) in enumerate(self.kfold.split(self.data)):
            print(f"Re-fitting trial #{split_idx}...")
            train_val_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            sa_model = self.model(split_idx=split_idx, train_val_data=train_val_data)

            best_params.append(sa_model.study.best_params)
            val_scores.append(sa_model.study.best_value)
            sa_model.refit_best_model(train_val_data["text"], train_val_data["label"])
            preds = sa_model.model.predict_proba(test_data["text"])
            test_scores.append(
                roc_auc_score(test_data["label"], preds, multi_class="ovr")
            )

        return val_scores, test_scores, best_params

    def fit_final_best_model(self, all_data: pd.DataFrame):
        """Fits the final best model to all data.

        Args:
            all_data (pd.DataFrame): data frame of all data
        """
        model = self.model(split_idx=None, train_val_data=None).get_pipeline()
        model.set_params(**self.model.FINAL_BEST_PARAMS)
        model.fit(all_data["text"], all_data["label"])
        return model


class OffTheShelfModelBenchmark(ABC):
    """A benchmark of the off the shelf models."""

    def __init__(self, all_data: pd.DataFrame):
        """Init.

        Args:
            all_data (pd.DataFrame): data frame of train and val data for inner CV
        """
        self.data = all_data
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    @abstractmethod
    def load(self) -> list:
        """Loads out-of-sample test scores for all 5 splits for this model.

        Returns:
            list: out-of-sample test scores for all 5 splits
        """
        pass


class VaderBenchmark(OffTheShelfModelBenchmark):
    def __init__(self, all_data: pd.DataFrame):
        super().__init__(all_data)
        self.analyzer = SentimentIntensityAnalyzer()

    def load(self) -> list:
        all_scores = []
        for _, (_, test_idx) in enumerate(self.kfold.split(self.data)):
            test_data = self.data.iloc[test_idx]
            texts = test_data["text"].to_list()

            preds = []
            for tt in texts:
                pred = self.analyzer.polarity_scores(test_data)
                preds.append((pred["neu"], pred["pos"], pred["neg"]))
