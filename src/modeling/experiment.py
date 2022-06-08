import time
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from rich.prompt import Prompt
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TFAutoModelForSequenceClassification)
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

    def run(self, n_trials: int = 100) -> None:
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

            sa_model.run_optuna(n_trials=n_trials)

    def load(self) -> tuple[list, list, list]:
        """Loads and refits best model for each split and evaluates it on outer test data

        Returns:
            tuple[list, list, list]: val_scores, test_scores, best_params
        """
        val_scores = []
        test_scores = []
        best_params = []
        times_taken = []
        for split_idx, (train_idx, test_idx) in enumerate(self.kfold.split(self.data)):
            print(f"Re-fitting trial #{split_idx}...")
            train_val_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]

            sa_model = self.model(split_idx=split_idx, train_val_data=train_val_data)

            best_params.append(sa_model.study.best_params)
            val_scores.append(sa_model.study.best_value)
            sa_model.refit_best_model(train_val_data["text"], train_val_data["label"])

            tic = time.perf_counter()
            preds = sa_model.model.predict_proba(test_data["text"])
            tac = time.perf_counter()
            times_taken.append(tac - tic)
            test_scores.append(
                roc_auc_score(test_data["label"], preds, multi_class="ovr")
            )

        return val_scores, test_scores, best_params, times_taken

    def apply_to_other_data(self, other_data: pd.DataFrame):
        """Applies the final best model to other data.

        Args:
            other_data (pd.DataFrame): data frame of other data, needs columns "text" and "label" where 1=pos, 2=neu, 3=neg
        """

        model = joblib.load(f"outputs/models/final_{self.model.__name__}.gz")
        preds = model.predict_proba(other_data["text"])
        return roc_auc_score(other_data["label"], preds, multi_class="ovr")

    def fit_final_best_model(self, all_data: pd.DataFrame):
        """Fits the final best model to all data.

        Args:
            all_data (pd.DataFrame): data frame of all data
        """
        model = self.model(split_idx=None, train_val_data=None).get_pipeline()
        model.set_params(**self.model.FINAL_BEST_PARAMS)
        model.fit(all_data["text"], all_data["label"])
        joblib.dump(model, f"outputs/models/final_{self.model.__name__}.gz")
        print("Saved final best model.")
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
        test_scores = []
        times_taken = []
        for _, (_, test_idx) in enumerate(self.kfold.split(self.data)):
            test_data = self.data.iloc[test_idx]
            texts = test_data["text"].to_list()

            preds = []
            tic = time.perf_counter()
            for tt in texts:
                pred = self.analyzer.polarity_scores(tt)
                probas = (pred["pos"], pred["neu"], pred["neg"])
                if probas[0] == probas[1] == probas[2] == 0:
                    probas = (0, 1, 0)  # no signal = neutral
                preds.append(probas)

            # prevent some float issues where the sum of probas is 0.9999 or 1.0001
            preds = np.array(preds)
            preds = preds / preds.sum(axis=1, keepdims=True)
            tac = time.perf_counter()

            times_taken.append(tac - tic)
            test_scores.append(
                roc_auc_score(test_data["label"], preds, multi_class="ovr")
            )

        return test_scores, times_taken


class FinBERTBenchmark(OffTheShelfModelBenchmark):
    def __init__(self, all_data: pd.DataFrame):
        super().__init__(all_data)
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )
        self.model.eval()

    def load(self) -> list:
        test_scores = []
        times_taken = []
        for split_idx, (_, test_idx) in enumerate(self.kfold.split(self.data)):
            print(f"At split #{split_idx}")
            print(f"{test_scores = }")
            test_data = self.data.iloc[test_idx]
            texts = test_data["text"].to_list()

            BATCHSIZE = (
                512  # we need to batch inference, runs OOM at CPU inference w/ 64GB RAM
            )
            batched_scores = []
            tic = time.perf_counter()
            for idx in range(0, len(texts), BATCHSIZE):
                print(f"{idx = }")
                tokens = self.tokenizer(
                    texts[idx : idx + BATCHSIZE],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                output = self.model(**tokens).logits.detach().numpy()
                scores = softmax(output, axis=1)
                batched_scores.append(scores)

            scores = np.vstack(batched_scores)
            tac = time.perf_counter()
            times_taken.append(tac - tic)

            test_scores.append(
                roc_auc_score(
                    test_data["label"], scores[:, [0, 2, 1]], multi_class="ovr"
                )
            )

        return test_scores, times_taken


class TwitterRoBERTaBenchmark(OffTheShelfModelBenchmark):
    def __init__(self, all_data: pd.DataFrame):
        super().__init__(all_data)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.model.eval()

    def load(self) -> list:
        test_scores = []
        times_taken = []
        for split_idx, (_, test_idx) in enumerate(self.kfold.split(self.data)):
            print(f"At split #{split_idx}")
            print(f"{test_scores = }")
            test_data = self.data.iloc[test_idx]
            texts = test_data["text"].to_list()

            BATCHSIZE = (
                512  # we need to batch inference, runs OOM at CPU inference w/ 64GB RAM
            )
            batched_scores = []
            tic = time.perf_counter()
            for idx in range(0, len(texts), BATCHSIZE):
                print(f"{idx = }")
                tokens = self.tokenizer(
                    texts[idx : idx + BATCHSIZE],
                    return_tensors="pt",
                    padding=True,
                )  # this is a dict w/ attention masks, not only tokens!
                output = self.model(**tokens).logits.detach().numpy()
                scores = softmax(output, axis=1)
                batched_scores.append(scores)

            scores = np.vstack(batched_scores)
            tac = time.perf_counter()
            times_taken.append(tac - tic)

            test_scores.append(
                roc_auc_score(
                    test_data["label"], scores[:, [2, 1, 0]], multi_class="ovr"
                )
            )

        return test_scores, times_taken


class NTUSDMeBenchmark(OffTheShelfModelBenchmark):
    def __init__(self, all_data: pd.DataFrame, path_to_ntusd_folder: str):
        super().__init__(all_data)
        dfs = [
            pd.read_json(f"{path_to_ntusd_folder}/{f}")[["token", "market_sentiment"]]
            for f in [
                "words.json",
                "NTUSD_Fin_emoji_v1.0.json",
                "NTUSD_Fin_hashtag_v1.0.json",
            ]
        ]
        self.ntusd = pd.concat(dfs).reset_index(drop=True)
        self.ntusd = {
            k: v for k, v in [r.values() for r in self.ntusd.to_dict(orient="records")]
        }  # dict: token -> sentiment

    def predict_one(self, text: str) -> float:
        # text_sentiment = 0.0
        text_sentiment = {"pos": 0.0, "neu": 0.0, "neg": 0.0}

        for token, token_sentiment in self.ntusd.items():
            if token in text:
                if token_sentiment > 0.0:
                    text_sentiment["pos"] += np.abs(token_sentiment)
                elif token_sentiment < 0.0:
                    text_sentiment["neg"] += np.abs(token_sentiment)
                else:
                    text_sentiment["neu"] += np.abs(token_sentiment)

        # normalize to resemble probas
        factor = sum(text_sentiment.values())
        if factor == 0.0:
            return {"pos": 0.0, "neu": 1.0, "neg": 0.0}
        else:
            return {k: v / factor for k, v in text_sentiment.items()}

    def load(self):
        test_scores = []
        times_taken = []
        for _, (_, test_idx) in enumerate(self.kfold.split(self.data)):
            test_data = self.data.iloc[test_idx]
            texts = test_data["text"].to_list()

            preds = []
            tic = time.perf_counter()
            for tt in texts:
                pred = self.predict_one(tt)
                probas = (pred["pos"], pred["neu"], pred["neg"])
                preds.append(probas)

            # prevent some float issues where the sum of probas is 0.9999 or 1.0001
            tac = time.perf_counter()
            times_taken.append(tac - tic)

            test_scores.append(
                roc_auc_score(test_data["label"], preds, multi_class="ovr")
            )

        return test_scores, times_taken


# if __name__ == "__main__":
#     ntusd = NTUSDMeBenchmark(None, "data/NTUSD-Fin")
#     print(ntusd.predict_one(""))
