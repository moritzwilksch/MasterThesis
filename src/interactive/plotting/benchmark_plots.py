#%%
import toml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plotting import set_style, Colors
import numpy as np

set_style()
#%%
data = toml.load("outputs/static/model_test_scores.toml")

scores = pd.DataFrame({k: data[k]["test_scores"] for k in data})
scores_melted = scores.melt(var_name="model", value_name="score")

#%%
fig, ax = plt.subplots(figsize=(10, 5))
sns.pointplot(
    data=scores_melted,
    x="model",
    y="score",
    order=scores.mean().sort_values().index,
    color=Colors.UPBLUE.value,
    linestyles=":",
    join=False,
    # capsize=0.05,
    ci="sd",
    ax=ax,
    zorder=10
)

ax.plot(
    scores.mean().sort_values().index,
    scores.mean().sort_values().values,
    color=Colors.UPBLUE.value,
    alpha=0.5,
    ls="-",
    zorder=0
)

ax.grid(axis="y", ls="--", color="black", alpha=0.25)
ax.set_ylim(0.49, 0.85)
ax.set_xlim(-0.25, 3.25)
ax.set_xlabel(None)
ax.set_ylabel("Out-of-sample ROC AUC", labelpad=15)

ax.tick_params(axis="x", length=0)
sns.despine(bottom=True)
plt.tight_layout()

