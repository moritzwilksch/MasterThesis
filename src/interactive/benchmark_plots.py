#%%
import toml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import set_style, Colors
set_style()
#%%
data = toml.load("data/raw/model_test_scores.toml")

scores = pd.DataFrame({k: data[k]["test_scores"] for k in data})
scores_melted = scores.melt(var_name="model", value_name="score")

#%%
fig, ax = plt.subplots(figsize=(12, 6))
sns.pointplot(
    data=scores_melted,
    x="model",
    y="score",
    order=scores.mean().sort_values().index,
    color=Colors.UPBLUE.value,
    linestyles=":",
    join=True,
    # capsize=0.05,
    ci="sd",
    ax=ax
)
ax.grid(axis="y", ls="--", color="black", alpha=0.25)
# sns.lineplot(data=scores_melted.loc[scores.mean().sort_values().index], x="model", y="score", ax=ax)
# ax.axhline(y=0.5, ls="--", color="black")
ax.set_ylim(0.48, 0.85)
sns.despine()