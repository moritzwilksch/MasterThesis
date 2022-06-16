#%%
import polars as pl
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix

from src.dl_modeling.data import TweetDataModule
from src.dl_modeling.models import RecurrentSAModel, TransformerSAModel
from src.utils.preprocessing import Preprocessor

#%%
data = TweetDataModule(split_idx=0, batch_size=64, model_type="transformer")
prepper = Preprocessor()
# model = RecurrentSAModel.load_from_checkpoint(
#     "lightning_logs/recurrent-split-0/epoch=22-val_acc=0.63.ckpt"
# )
model = TransformerSAModel.load_from_checkpoint(
    "lightning_logs/transformer_final/final_epoch=24-val_acc=0.64.ckpt"
)
model.eval()


#%%
s = "short $TSLA, buy puts"
# s = "long $TSLA, buy calls"
s = "$OXY just laid off 40% of staff"

df = prepper.process(pl.DataFrame({"text": [s]}))
# x = torch.Tensor(data.vocab(data.tokenizer(df["text"][0]))).long().reshape(-1, 1)
x = torch.Tensor(list(data.tokenizer(df["text"]))[0]).long().reshape(-1, 1)

# F.softmax(model(x, seq_lens=[len(x)]), dim=0)
out = model(x, masks=torch.full((len(x),), False).unsqueeze(0))
F.softmax(out, dim=1)

#%%
test = data.test_dataloader()
trainer = ptl.Trainer()
batched_preds = trainer.predict(model, test)

#%%
preds = torch.vstack(batched_preds)  # .argmax(dim=1)
print(roc_auc_score(data.ytest, preds.numpy(), multi_class="ovr"))
print(confusion_matrix(data.ytest, preds.numpy().argmax(axis=1), normalize="true"))
