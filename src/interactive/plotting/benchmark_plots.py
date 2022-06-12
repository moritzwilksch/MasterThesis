#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml

from src.utils.plotting import Colors, set_style

set_style()
#%%
data = toml.load("outputs/static/model_test_scores.toml")

scores = pd.DataFrame({k: data[k]["test_scores"] for k in data})
scores_melted = scores.melt(var_name="model", value_name="score")

times = pd.DataFrame({k: data[k]["times_taken"] for k in data})
times = times / 2000 * 1000  # per-sample in ms!!!
times_melted = times.melt(var_name="model", value_name="time")

#%%
fig, ax = plt.subplots(figsize=(12, 5))
sns.pointplot(
    data=scores_melted,
    x="model",
    y="score",
    order=scores.mean().sort_values().index,
    color=Colors.DARKBLUE.value,
    linestyles=":",
    join=False,
    # capsize=0.05,
    ci="sd",
    ax=ax,
    zorder=10,
)

ax.plot(
    scores.mean().sort_values().index,
    scores.mean().sort_values().values,
    color=Colors.DARKBLUE.value,
    alpha=0.5,
    ls="-",
    zorder=0,
)

ax.grid(axis="y", ls="--", color="black", alpha=0.25)
ax.set_ylim(0.49, 0.85)
ax.set_xlim(-0.25, 5.25)
ax.set_xlabel("Model", weight="bold", labelpad=15)
ax.set_ylabel("Out-of-sample ROC AUC", labelpad=15, weight="bold")

ax.tick_params(axis="x", length=0)
# ax.set_xticklabels(ax.get_xticklabels(), weight="bold")
sns.despine(bottom=True)
plt.tight_layout()
fig.savefig(
    "outputs/plots/model_performance_pyfin.pdf", bbox_inches="tight", facecolor="white"
)

#%%
fig, ax = plt.subplots(figsize=(12, 5))
sns.pointplot(
    data=times_melted,
    x="model",
    y="time",
    order=times.mean().sort_values().index,
    color=Colors.DARKBLUE.value,
    linestyles=":",
    join=False,
    # capsize=0.05,
    ci="sd",
    ax=ax,
    zorder=10,
)

ax.plot(
    times.mean().sort_values().index,
    times.mean().sort_values().values,
    color=Colors.DARKBLUE.value,
    alpha=0.5,
    ls="-",
    zorder=0,
)

ax.grid(axis="y", ls="--", color="black", alpha=0.25)

ax.set_xlabel("Model", labelpad=15, weight="bold")
ax.set_ylabel("Inference time per sample (ms)", labelpad=15, weight="bold")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")
ax.tick_params(axis="x", length=0)
sns.despine(bottom=True)
plt.tight_layout()
fig.savefig(
    "outputs/plots/model_inference_time.pdf", bbox_inches="tight", facecolor="white"
)
