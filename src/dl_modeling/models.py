#%%
from abc import ABC

import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from positional_encodings import PositionalEncoding1D, Summer
from pytorch_lightning.loggers import TensorBoardLogger
from regex import R

from src.dl_modeling.data import TweetDataModule

tb_logger = TensorBoardLogger("lightning_logs", name="recurrent")
BATCH_SIZE = 64


class BaseDLModel(ABC, ptl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()


class RecurrentSAModel(BaseDLModel):
    def __init__(
        self,
        vocab_size: int,
        token_dropout: float,
        embedding_dim: int,
        gru_hidden_dim: int,
        hidden_dim: int,
        dropout: float,
        lr: float = 1e-3,
    ):
        super().__init__()

        # layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.token_dropout = nn.Dropout(token_dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            bidirectional=True,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.hidden1 = nn.Linear(gru_hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, x, seq_lens):
        x = self.embedding(x)
        x = self.token_dropout(x)
        x = nn.utils.rnn.pack_padded_sequence(x, seq_lens, enforce_sorted=False)
        out, hidden = self.gru(x)
        hidden = self.dropout1(hidden)
        x = self.hidden1(hidden[-1, :, :])
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        x = x.squeeze()

        return x

    def training_step(self, batch, batch_idx):
        x, seq_lens, y = batch
        y_hat = self.forward(x, seq_lens)
        loss = F.cross_entropy(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log("loss", loss, batch_size=BATCH_SIZE)
        self.log("train_acc", self.train_accuracy, prog_bar=True, batch_size=BATCH_SIZE)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, seq_lens, y = batch
        y_hat = self.forward(x, seq_lens)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log("val_loss", loss, batch_size=BATCH_SIZE)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=BATCH_SIZE)

    def predict_step(self, batch, batch_idx):
        x, seq_lens, y = batch
        y_hat = self.forward(x, seq_lens)
        return F.softmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class TransformerSAModel(BaseDLModel):
    BEST_PARAMS = {
        "dim_ff": 241,
        "dropout": 0.18409452359591996,
        "embedding_dim": 105,
        "hidden_dim": 249,
        "token_dropout": 0.463518938835938,
    }  # Val AUC = 0.81899, Test AUC = 0.8005475221900585

    def __init__(
        self,
        vocab_size: int,
        token_dropout: float,
        embedding_dim: int,
        nhead: int,
        dim_ff: int,
        hidden_dim: int,
        dropout: float,
        lr: float = 1e-3,
    ):
        super().__init__()

        # layers
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0
        )
        self.pos_encodings = Summer(PositionalEncoding1D(embedding_dim))
        self.token_dropout = nn.Dropout(token_dropout)

        self.transformer = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.hidden1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, x, masks):
        x = self.embedding(x)
        # x = x + self.pos_encodings(x)
        x = torch.swapaxes(x, 0, 1)
        x = self.pos_encodings(
            x
        )  # this library needs (batch_size, x, emb_dim) tensors!!
        x = torch.swapaxes(x, 0, 1)

        x = self.token_dropout(x)
        x = self.transformer(x, src_key_padding_mask=masks)
        x = self.dropout1(x)
        x = torch.mean(x, dim=0)

        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.output_layer(x)

        return x

    def training_step(self, batch, batch_idx):
        x, masks, y = batch
        y_hat = self.forward(x, masks)
        loss = F.cross_entropy(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log("loss", loss, batch_size=BATCH_SIZE)
        self.log("train_acc", self.train_accuracy, prog_bar=True, batch_size=BATCH_SIZE)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, masks, y = batch
        y_hat = self.forward(x, masks)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.log("val_loss", loss, batch_size=BATCH_SIZE)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=BATCH_SIZE)

    def predict_step(self, batch, batch_idx):
        x, masks, y = batch
        y_hat = self.forward(x, masks)
        return F.softmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


#%%
if __name__ == "__main__":
    tb_logger = TensorBoardLogger("lightning_logs", name=f"transformer-split-{0}")
    data = TweetDataModule(split_idx=0, batch_size=BATCH_SIZE, model_type="transformer")

    model = TransformerSAModel(
        vocab_size=3_000,
        token_dropout=0.2,
        embedding_dim=128,
        nhead=1,
        dim_ff=128,
        hidden_dim=64,
        dropout=0.2,
        lr=1e-3,
    )

    # trainer
    trainer = ptl.Trainer(
        logger=tb_logger,
        max_epochs=10,
        log_every_n_steps=50,
        # auto_lr_find=True,
    )

    # res = trainer.tuner.lr_find(
    #     model,
    #     data.train_dataloader(),
    #     data.val_dataloader(),
    #     num_training=100,
    #     min_lr=1e-6,
    #     max_lr=0.1,
    # )
    # res.plot()

    trainer.fit(
        model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )
