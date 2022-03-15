from rich.console import Console
from rich.panel import Panel

from src.preliminary.demo_tweets.models import (FinBERT, ModelWrapper,
                                                TwitterRoberta, Vader)

c = Console()
c.clear()


class Experiment:
    def __init__(self, models: list[ModelWrapper], texts: list[str]):
        self.models = models
        self.texts = texts

    def run(self) -> None:
        for text in self.texts:
            c.print(Panel(text))
            for model in self.models:
                pred = model.predict(text)
                c.print(f"{f'{model.__class__.__name__}:':<25} {pred}")


experiment = Experiment(
    models=[TwitterRoberta(), FinBERT(), Vader()],
    texts=[
        "Today is tuesday.",
        "Had a great day today! #happy #good #great #greatday #goodday #gooddaytoday #gooddaytoday",
        "What a miserable shitty day! #sad #bad #badday #baddaytoday #baddaytoday",
        "Stocks rallied and the British pound gained.",
        "Stocks deep red and the British pound lost 5%...",
    ],
)
experiment.run()
