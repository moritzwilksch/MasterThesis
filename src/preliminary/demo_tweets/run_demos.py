import io

import yaml
from rich.console import Console
from rich.panel import Panel

from src.preliminary.demo_tweets.models import (FinancialBERT, FinBERT,
                                                ModelWrapper, TwitterRoberta,
                                                Vader)
from src.utils.db_logging import logger
from src.utils.storage import bucket

logger.info("Starting demo run")
c = Console(record=True)
c.clear()


class Experiment:
    def __init__(self, models: list[ModelWrapper], examples_filename: str):
        self.models = models

        with open(examples_filename) as f:
            self.examples = yaml.safe_load(f)

    def run(self, manual_input: str = None) -> None:
        if manual_input is not None:
            c.print(manual_input, style="bold")
            for model in self.models:
                pred = model.predict(manual_input)
                c.print(f"\t{f'{model.__class__.__name__}:':<25} {pred}")
            c.print()
            return None

        for category in self.examples:
            c.print(Panel(category.upper()), style="bold")
            texts = self.examples.get(category)
            for text in texts:
                c.print(text, style="bold")
                for model in self.models:
                    pred = model.predict(text)
                    c.print(f"\t{f'{model.__class__.__name__}:':<25} {pred}")
                c.print()


experiment = Experiment(
    models=[TwitterRoberta(), FinBERT(), FinancialBERT(), Vader()],
    examples_filename="data/examples/examples.yaml",
)
experiment.run()

with io.BytesIO() as f:
    f.write(c.export_text().encode("utf-8"))
    f.seek(0)
    bucket.upload_fileobj(f, "outputs/examples/examples.txt")

# with open("outputs/examples/examples.txt", "w") as f:
#     f.writelines(c.export_text())

logger.info("Ended demo run")
