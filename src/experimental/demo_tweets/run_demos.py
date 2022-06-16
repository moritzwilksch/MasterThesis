import io

import toml
from rich.console import Console
from rich.panel import Panel

from src.experimental.demo_tweets.models import (FinancialBERT, FinBERT,
                                                 ModelWrapper, NTUSDFin,
                                                 PyFinLogReg, TwitterRoberta,
                                                 Vader)
from src.utils.storage import bucket

c = Console(record=True)
c.clear()


class Experiment:
    def __init__(self, models: list[ModelWrapper], examples_filename: str):
        self.models = models

        # self.examples = toml.load("data/examples/examples.toml")
        self.examples = toml.load(examples_filename)

    def run(self, manual_input: str = None) -> None:
        latex_lines = []

        if manual_input is not None:
            c.print(manual_input, style="bold")
            for model in self.models:
                pred = model.predict(manual_input)
                c.print(f"\t{f'{model.__class__.__name__}:':<25} {pred}")
            c.print()
            return None

        for category in self.examples:
            latex_lines.append(r"\cmidrule(l){1-1}")
            latex_lines.append(f"\\emph{{{category.replace('_', ' ').title()}}} \\\\")
            latex_lines.append(r"\cmidrule(l){1-7}")

            c.print(Panel(category.upper()), style="bold")
            texts = self.examples.get(category)
            for text in texts:
                y_true, text = text.split("|#|")
                latex_ordered_model_preds = []

                c.print(text, style="bold")
                for model in self.models:
                    pred = model.predict(text)
                    prediction_correct = max(pred, key=pred.get) == y_true

                    c.print(f"\t{f'{model.__class__.__name__}:':<25} {pred}")
                    latex_ordered_model_preds.append(
                        f"\\emph{{\\textbf{{{pred[y_true]:.2f}*}}}}"
                        if prediction_correct
                        else f"{pred[y_true]:.2f}"
                    )

                latex_lines.append(
                    f"{text} & {y_true.upper()} & {' & '.join(latex_ordered_model_preds)} \\\\"
                )
                c.print()

        c.print(
            "\n".join(
                [s.replace("$", "\$").replace("%", "\%") for s in latex_lines][1:]
            )
        )


experiment = Experiment(
    models=[Vader(), NTUSDFin(), FinBERT(), TwitterRoberta(), PyFinLogReg()],
    # models=[Vader(), NTUSDFin(), PyFinLogReg()],
    examples_filename="data/examples/demo_pieces.toml",
)
experiment.run()

# with io.BytesIO() as f:
#     f.write(c.export_text().encode("utf-8"))
#     f.seek(0)
#     bucket.upload_fileobj(f, "outputs/examples/examples.txt")

# with open("outputs/examples/examples.txt", "w") as f:
#     f.writelines(c.export_text())

# logger.info("Ended demo run")
