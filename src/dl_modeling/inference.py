#%%
import time

import polars as pl
import pytorch_lightning as ptl
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

from src.dl_modeling.data import (BERTTensorDataModule, BertTensorDataSet,
                                  TweetDataModule)
from src.dl_modeling.models import (BERTSAModel, RecurrentSAModel,
                                    TransformerSAModel)
from src.utils.preprocessing import Preprocessor

#%%
USE_MODEL = "bert"

prepper = Preprocessor()
if USE_MODEL == "transformer":
    data = TweetDataModule(split_idx="retrain", batch_size=64, model_type="transformer")
    model = TransformerSAModel.load_from_checkpoint(
        "outputs/models/transformer_final.ckpt"
    )
    model.eval()
elif USE_MODEL == "recurrent":
    data = TweetDataModule(split_idx="retrain", batch_size=64, model_type="recurrent")
    model = RecurrentSAModel.load_from_checkpoint("outputs/models/gru_final.ckpt")
elif USE_MODEL == "bert":
    data = BERTTensorDataModule(
        split_idx="retrain", data_prefix="prep_pyfin", batch_size=64
    )
    model = BERTSAModel.load_from_checkpoint("outputs/models/bert_final.ckpt")
model.eval()
#%%
# s = "short $TSLA, buy puts"
# # s = "long $TSLA, buy calls"
# s = "$OXY just laid off 40% of staff"

# df = prepper.process(pl.DataFrame({"text": [s]}))
# # x = torch.Tensor(data.vocab(data.tokenizer(df["text"][0]))).long().reshape(-1, 1)
# x = torch.Tensor(list(data.tokenizer(df["text"]))[0]).long().reshape(-1, 1)

# # F.softmax(model(x, seq_lens=[len(x)]), dim=0)
# out = model(x, masks=torch.full((len(x),), False).unsqueeze(0))
# F.softmax(out, dim=1)

#%%
test = data.test_dataloader()
trainer = ptl.Trainer()

tic = time.perf_counter()
batched_preds = trainer.predict(model, test)
tac = time.perf_counter()

time_taken = (
    tac - tic
)  # to make comparable: this should be inference time for 2_000 samples
data_len = len(data.ytest) if USE_MODEL != "bert" else 2_500
bert_conversion_time = 0 if USE_MODEL != "bert" else 0.0563
time_for_2000 = (time_taken / data_len + bert_conversion_time) * 2_000
print(f"Prediction time for 2000 samples: {time_for_2000}")

#%%
preds = torch.vstack(batched_preds)  # .argmax(dim=1)
print(roc_auc_score(data.ytest, preds.numpy(), multi_class="ovr"))
print(confusion_matrix(data.ytest, preds.numpy().argmax(axis=1), normalize="true"))
print(classification_report(data.ytest, preds.numpy().argmax(axis=1)))
