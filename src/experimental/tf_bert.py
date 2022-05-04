#%%
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = TFBertModel.from_pretrained("bert-base-cased")

#%%
inputs = tokenizer("Hello, my dog is cute", return_tensors="tf")
outputs = model(inputs).last_hidden_state
print(outputs)
