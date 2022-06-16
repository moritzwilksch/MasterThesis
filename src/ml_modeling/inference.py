#%%
import numpy as np
from pyfin_sentiment.model import SentimentModel

from src.dl_modeling.data import TweetDataModule

data = TweetDataModule(0, batch_size=16)
x = data.xtest
y = data.ytest


#%%
# SentimentModel.download("small")
model = SentimentModel("small")
preds = model.predict_proba(x.to_list())

#%%
from sklearn.metrics import confusion_matrix, roc_auc_score

# print(confusion_matrix(y, preds.astype("int") - 1, normalize="true"))
print(roc_auc_score(y, preds, multi_class="ovr"))
