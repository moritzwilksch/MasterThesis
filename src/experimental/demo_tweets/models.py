import csv
import urllib.request
from abc import ABC

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class ModelWrapper(ABC):
    def predict(self, text: str) -> str:
        pass

    def _preprocess(self, text: str) -> str:
        pass

    def prettify(self, sentiment: dict) -> dict:
        N_DECIMALS = 3
        return {
            "neg": round(sentiment.get("neg"), N_DECIMALS),
            "neu": round(sentiment.get("neu"), N_DECIMALS),
            "pos": round(sentiment.get("pos"), N_DECIMALS),
        }


# -----------------------------------------------------------------------------


class TwitterRoberta(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        task = "sentiment"
        self.MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL)

        # download label mapping
        self.labels = []
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode("utf-8").split("\n")
            csvreader = csv.reader(html, delimiter="\t")
        self.labels = [row[1] for row in csvreader if len(row) > 1]

        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.MODEL)

    def _preprocess(self, text: str) -> str:
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def predict(self, text: str) -> str:
        text = self._preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors="tf")
        output = self.model(encoded_input)
        scores = output[0][0].numpy()
        scores = softmax(scores)
        sentiment = dict(zip(["neg", "neu", "pos"], scores))
        return self.prettify(sentiment)


# -----------------------------------------------------------------------------


class FinBERT(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        )

    def predict(self, text: str) -> str:

        tokens = self.tokenizer(text, return_tensors="pt")
        output = self.model(**tokens).logits.detach().numpy()[0]
        scores = softmax(output)
        sentiment = dict(zip(["pos", "neg", "neu"], scores))
        return self.prettify(sentiment)


# -----------------------------------------------------------------------------


class FinancialBERT(ModelWrapper):  # is this a knock-off model?
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ahmedrachid/FinancialBERT-Sentiment-Analysis"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "ahmedrachid/FinancialBERT-Sentiment-Analysis"
        )

    def predict(self, text: str) -> str:

        tokens = self.tokenizer(text, return_tensors="pt").get("input_ids")
        output = self.model(tokens).logits.detach().numpy()[0]
        scores = softmax(output)
        sentiment = dict(zip(["neg", "neu", "pos"], scores))
        return self.prettify(sentiment)


# -----------------------------------------------------------------------------


class Vader(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> str:
        scores = self.analyzer.polarity_scores(text)
        return self.prettify(scores)  # will drop the "compound" field


class PyFinLogReg(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.model = joblib.load("outputs/models/final_LogisticRegressionModel.gz")

    def predict(self, text: str) -> str:
        preds = self.model.predict_proba([text]).ravel()
        return self.prettify({"pos": preds[0], "neu": preds[1], "neg": preds[2]})


# -----------------------------------------------------------------------------
class NTUSDFin(ModelWrapper):
    def __init__(self, path_to_ntusd_folder: str = "data/NTUSD-Fin/") -> None:
        super().__init__()
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

    def predict(self, text: str) -> float:
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
            prediction = {"pos": 0.0, "neu": 1.0, "neg": 0.0}
        else:
            prediction = {k: v / factor for k, v in text_sentiment.items()}

        return self.prettify(prediction)
