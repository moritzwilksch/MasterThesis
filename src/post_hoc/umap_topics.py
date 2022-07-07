#%%
import pandas as pd
import torch
import umap
import umap.plot
from bokeh.plotting import show

#%%
tensors = []
data_prefix = "prep_pyfin"
for ii in range(20):
    tensors.append(torch.load(f"data/distilbert/{data_prefix}_{ii}.pt"))
all_data = torch.cat(tensors, dim=0).clone().detach().numpy()


df = pd.read_parquet("data/labeled/labeled_tweets.parquet")
#%%

reducer = umap.UMAP()
mapper = reducer.fit(all_data)
# low_dim = reducer.transform(all_data)

#%%

umap.plot.output_notebook()
p = umap.plot.interactive(
    mapper,
    hover_data=pd.DataFrame({"index": range(10_000), "text": df["text"]}),
    point_size=2,
    labels=df["label"],
)
show(p)
