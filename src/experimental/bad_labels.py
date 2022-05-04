#%%
import numpy as np
import polars as pl

#%%
df = pl.read_csv(
    "https://github.com/moritzwilksch/SocialMediaBusinessAnalytics/raw/main/00_source_data/SRS_sentiment_labeled.csv",
    ignore_errors=True,
).select(["tweet", "sentiment"])


#%%
from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer()
data = tf.fit_transform(df.select("tweet").to_series().to_list())

#%%
subset = data[df.select(pl.col("sentiment") == 0).to_series().to_list()]

#%%
import umap

reducer = umap.UMAP()
lower_dim_data = reducer.fit_transform(subset)


#%%
from sklearn.metrics.pairwise import euclidean_distances

avg = lower_dim_data.mean(axis=0)
dists = euclidean_distances(lower_dim_data, [avg])

outlier_threshold = np.quantile(dists, 0.975)
is_outlier = (dists > outlier_threshold).ravel()

#%%
import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Palatino"
plt.rcParams["font.size"] = 16


fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(
    lower_dim_data[~is_outlier, 0],
    lower_dim_data[~is_outlier, 1],
    color="blue",
    alpha=0.2,
)
ax.scatter(lower_dim_data[is_outlier, 0], lower_dim_data[is_outlier, 1], color="red")
ax.set(xlabel="UMAP Component 1", ylabel="UMAP Component 2")
sns.despine()
plt.savefig("outputs/plots/plot.pdf", bbox_inches="tight")
#%%
for o in (
    df.filter(pl.col("sentiment") == 0)[is_outlier]
    .select("tweet")
    .to_series()
    .to_list()
):
    print(o)
    print("-" * 80)
