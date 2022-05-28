#%%
import polars as pl

df = pl.read_json("data/raw/fin-SoMe.json")

#%%
query = "regret"
subset = df.filter(pl.col("tweet").str.contains(query))
for t in subset.to_dicts():
    print(f"|{t.get('market_sentiment').upper()}| {t.get('tweet')}")
    print("-" * 80)
