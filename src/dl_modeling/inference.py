#%%
import polars as pl
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import RecurrentSAModel
from src.utils.preprocessing import Preprocessor

#%%
data = TweetDataModule(split_idx=0, batch_size=64)
prepper = Preprocessor()
model = RecurrentSAModel.load_from_checkpoint(
    "lightning_logs/recurrent/best_checkpoints/recurrent-epoch=46-val_acc=0.62.ckpt"
)
model.eval()
#%%
s = "buy this great stock!!! long $AAPL"
df = prepper.process(pl.DataFrame({"text": [s]}))
x = torch.Tensor(data.vocab(data.tokenizer(df["text"][0]))).long().reshape(-1, 1)
# F.softmax(model(x, seq_lens=[len(x)]), dim=0)
model(x, seq_lens=[len(x)])
#%%
test = data.test_dataloader()
trainer = ptl.Trainer()
batched_preds = trainer.predict(model, test)

#%%
preds = torch.vstack(batched_preds)  # .argmax(dim=1)
print(roc_auc_score(data.ytest, preds.numpy(), multi_class="ovr"))

#%%
