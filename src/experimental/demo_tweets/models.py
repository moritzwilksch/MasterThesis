import csv
import urllib.request
from abc import ABC

from scipy.special import softmax
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          TFAutoModelForSequenceClassification)
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

        tokens = self.tokenizer(text, return_tensors="pt").get("input_ids")
        output = self.model(tokens).logits.detach().numpy()[0]
        scores = softmax(output)
        sentiment = dict(zip(["pos", "neg", "neu"], scores))
        return self.prettify(sentiment)


# -----------------------------------------------------------------------------


class FinancialBERT(ModelWrapper):
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
