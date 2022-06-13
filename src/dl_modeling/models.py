#%%
from abc import ABC

import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
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
