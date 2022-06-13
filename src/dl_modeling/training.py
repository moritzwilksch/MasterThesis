from posixpath import split

import numpy as np
import optuna
import pytorch_lightning as ptl
import torch
from lightgbm import early_stopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score

from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import BATCH_SIZE, RecurrentSAModel

if __name__ == "__main__":

    def objective(trial):
        aucs_per_split = []

        for split_idx in range(5):
            print(f"Starting split {split_idx}")
            tb_logger = TensorBoardLogger(
                "lightning_logs", name=f"recurrent-split-{split_idx}"
            )
            data = TweetDataModule(split_idx=split_idx, batch_size=BATCH_SIZE)

            if trial is None:
                model = RecurrentSAModel(
                    vocab_size=3_000,
                    token_dropout=0.2,
                    embedding_dim=64,
                    gru_hidden_dim=64,
                    hidden_dim=64,
                    dropout=0.5,
                    lr=1e-3,
                )

            else:
                model = RecurrentSAModel(
                    vocab_size=3_000,
                    token_dropout=trial.suggest_float("token_dropout", 0.0, 0.5),
                    embedding_dim=trial.suggest_int("embedding_dim", 4, 128),
                    gru_hidden_dim=trial.suggest_int("gru_hidden_dim", 4, 256),
                    hidden_dim=trial.suggest_int("hidden_dim", 8, 256),
                    dropout=trial.suggest_float("dropout", 0.0, 0.5),
                    lr=1e-3,
                )

            # callbacks
            checkpoint_callback = ptl.callbacks.ModelCheckpoint(
                save_top_k=1,
                monitor="val_acc",
                mode="max",
                dirpath=f"lightning_logs/recurrent-split-{split_idx}",
                filename="{epoch:02d}-{val_acc:.2f}",
            )

            early_stopping_callback = ptl.callbacks.EarlyStopping(
                monitor="val_acc", mode="max", patience=10
            )

            # trainer
            trainer = ptl.Trainer(
                logger=tb_logger,
                max_epochs=50,
                log_every_n_steps=50,
                auto_lr_find=False,
                callbacks=[checkpoint_callback, early_stopping_callback],
            )

            trainer.fit(
                model,
                train_dataloaders=data.train_dataloader(),
                val_dataloaders=data.val_dataloader(),
            )

            # load best and calculate test AUC
            best_model_path = trainer.checkpoint_callback.best_model_path
            print(best_model_path)

            model = RecurrentSAModel.load_from_checkpoint(best_model_path)
            model.eval()

            test = data.test_dataloader()
            batched_preds = trainer.predict(model, test)

            preds = torch.vstack(batched_preds)  # .argmax(dim=1)
            aucs_per_split.append(
                roc_auc_score(data.ytest, preds.numpy(), multi_class="ovr")
            )

            if trial is None:
                break

        return np.mean(aucs_per_split)

    study = optuna.create_study(
        storage="sqlite:///tuning/dl_optuna.db",
        study_name=f"GRU",
        direction="maximize",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)
    # objective(trial=None)  # one manual run for testing a model
