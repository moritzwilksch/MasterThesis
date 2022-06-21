import numpy as np
import optuna
import pytorch_lightning as ptl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score

from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import BATCH_SIZE, TransformerSAModel

torch.set_num_threads(8)
if __name__ == "__main__":

    def objective(trial):
        aucs_per_split = []

        for split_idx in range(5):
            print(f"Starting split {split_idx}")
            tb_logger = TensorBoardLogger(
                "lightning_logs", name=f"transformer-split-{split_idx}"
            )
            data = TweetDataModule(
                split_idx=split_idx, batch_size=BATCH_SIZE, model_type="transformer"
            )

            if trial is None:
                model = TransformerSAModel(
                    vocab_size=3_000,
                    token_dropout=0.2,
                    embedding_dim=50,
                    nhead=1,
                    dim_ff=128,
                    hidden_dim=64,
                    dropout=0.2,
                    lr=1e-3,
                )

            else:
                model = TransformerSAModel(
                    vocab_size=3_000,
                    nhead=4,
                    # token_dropout=trial.suggest_float("token_dropout", 0.0, 0.5),
                    # embedding_dim=trial.suggest_int("embedding_dim", 4, 128),
                    # dim_ff=trial.suggest_int("dim_ff", 4, 256),
                    # hidden_dim=trial.suggest_int("hidden_dim", 8, 256),
                    # dropout=trial.suggest_float("dropout", 0.0, 0.5),
                    # lr=1e-3,
                    **TransformerSAModel.BEST_PARAMS,
                )

            # callbacks
            checkpoint_callback = ptl.callbacks.ModelCheckpoint(
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                dirpath=f"lightning_logs/transformer-split-{split_idx}",
                filename="{epoch:02d}-{val_acc:.2f}",
            )

            early_stopping_callback = ptl.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=10
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

            model = TransformerSAModel.load_from_checkpoint(best_model_path)
            model.eval()

            val = data.val_dataloader()

            # create val-set predictions
            batched_preds = trainer.predict(model, val)
            preds = torch.vstack(batched_preds)

            # we need to extract the val-set labels from the dataloader
            yval_true = []
            for _, _, y in data.val_dataloader():
                yval_true.append(y.numpy())
            yval_true = np.concatenate(yval_true)

            aucs_per_split.append(
                roc_auc_score(yval_true, preds.numpy(), multi_class="ovr")
            )
            print(f"Split {split_idx} AUCs: {aucs_per_split}")

            if trial is None:
                break

        return np.mean(aucs_per_split)

    # study = optuna.create_study(
    #     storage="sqlite:///tuning/dl_optuna_transformer.db",
    #     study_name=f"Transformer",
    #     direction="maximize",
    #     load_if_exists=True,
    # )

    # study.optimize(objective, n_trials=50)

    objective(trial=None)  # one manual run


def retrain_best_model():
    data = TweetDataModule(
        split_idx="retrain", batch_size=BATCH_SIZE, model_type="transformer"
    )

    train_dataloader, mini_val_dataloader = data.trainval_dataloader_for_retraining()

    model = TransformerSAModel(
        vocab_size=3_000, nhead=1, **TransformerSAModel.BEST_PARAMS
    )

    tb_logger = TensorBoardLogger("lightning_logs", name=f"transformer_final")

    # callbacks
    checkpoint_callback = ptl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val_auc",
        mode="max",
        dirpath=f"lightning_logs/transformer_final",
        filename="final_{epoch:02d}-{val_acc:.2f}",
    )

    early_stopping_callback = ptl.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=10
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
        train_dataloaders=train_dataloader,
        val_dataloaders=mini_val_dataloader,
    )

    # load best and calculate test AUC
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)

    model = TransformerSAModel.load_from_checkpoint(best_model_path)
    model.eval()

    test = data.test_dataloader()

    # create test-set predictions
    batched_preds = trainer.predict(model, test)
    preds = torch.vstack(batched_preds)

    # we need to extract the test-set labels from the dataloader
    ytest_true = []
    for _, _, y in data.test_dataloader():
        ytest_true.append(y.numpy())
    ytest_true = np.concatenate(ytest_true)

    print(roc_auc_score(ytest_true, preds.numpy(), multi_class="ovr"))


# retrain_best_model()
