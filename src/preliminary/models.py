from abc import ABC
from tkinter import N


class ModelWrapper(ABC):
    def predict(text: str) -> str:
        pass

    def _preprocess(text: str) -> str:
        pass


# -----------------------------------------------------------------------------


class TwitterRoberta(ModelWrapper):
    def __init__(self) -> None:
        super().__init__()
