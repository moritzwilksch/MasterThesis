#%%
from abc import ABC

import gensim
import numpy as np
import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from positional_encodings import PositionalEncoding1D, Summer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer, pipeline

from src.dl_modeling.data import (BERTTensorDataModule, BertTensorDataSet,
                                  TweetDataModule)

tb_logger = TensorBoardLogger("lightning_logs", name="recurrent")
BATCH_SIZE = 64


class BaseDLModel(ABC, ptl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_auc = torchmetrics.AUROC(num_classes=3)


###############################################################################
class RecurrentSAModel(BaseDLModel):
    BEST_PARAMS = {
        "dropout": 0.4137700949108063,
        "embedding_dim": 54,
        "gru_hidden_dim": 44,
        "hidden_dim": 189,
        "token_dropout": 0.49831147449844915,
    }  # val-auc: 0.7950535820672638,

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
        self.val_auc(y_hat, y)
        self.log("val_loss", loss, batch_size=BATCH_SIZE)
        self.log("val_auc", self.val_auc, prog_bar=True, batch_size=BATCH_SIZE)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=BATCH_SIZE)

    def predict_step(self, batch, batch_idx):
        x, seq_lens, y = batch
        y_hat = self.forward(x, seq_lens)
        return F.softmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


###############################################################################
class TransformerSAModel(BaseDLModel):
    BEST_PARAMS = {
        "dim_ff": 140,
        "dropout": 0.28603031004494467,
        "embedding_dim": 72,
        "hidden_dim": 237,
        "token_dropout": 0.4387140736539719,
    }  # Val AUC = 0.81488 Test AUC = 0.80826

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

        # print("Loading embeddings...")
        # model = gensim.models.KeyedVectors.load_word2vec_format(
        #     "~/Downloads/glove.6B.50d.txt", binary=False, no_header=True
        # )
        # weights = torch.FloatTensor(model.vectors)
        # self.embedding = nn.Embedding.from_pretrained(weights)
        # print("Done.")

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
        self.val_auc(y_hat, y)
        self.log("val_loss", loss, batch_size=BATCH_SIZE)
        self.log("val_auc", self.val_auc, prog_bar=True, batch_size=BATCH_SIZE)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=BATCH_SIZE)

    def predict_step(self, batch, batch_idx):
        x, masks, y = batch
        y_hat = self.forward(x, masks)
        return F.softmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


###############################################################################
class BERTSAModel(BaseDLModel):
    BEST_PARAMS = {
        "dropout": 0.24879384650702396,
        "hidden_dim": 238,
    }  # val-auc = 0.8016462095744542

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.dropout1 = nn.Dropout(dropout)
        self.hidden1 = nn.Linear(768, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.output_layer(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_accuracy(y_hat, y)
        self.log("loss", loss, batch_size=BATCH_SIZE)
        self.log("train_acc", self.train_accuracy, prog_bar=True, batch_size=BATCH_SIZE)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_accuracy(y_hat, y)
        self.val_auc(y_hat, y)
        self.log("val_loss", loss, batch_size=BATCH_SIZE)
        self.log("val_auc", self.val_auc, prog_bar=True, batch_size=BATCH_SIZE)
        self.log("val_acc", self.val_accuracy, prog_bar=True, batch_size=BATCH_SIZE)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        return F.softmax(y_hat, dim=1)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
