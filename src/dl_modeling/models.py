import pytorch_lightning as ptl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl_modeling.data import TweetDataModule


class RecurrentSAModel(ptl.LightningModule):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters({"lr": lr})
        self.dense1 = nn.Linear(1, 1)

    def forward(self, x):
        print(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


model = RecurrentSAModel()

data = TweetDataModule(split_idx=0)
trainer = ptl.Trainer(max_epochs=10)
trainer.fit(
    model,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader(),
)
