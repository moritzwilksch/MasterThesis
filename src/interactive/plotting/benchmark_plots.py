#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml

from src.utils.plotting import Colors, set_style

set_style()
OURS = ["LogReg", "SVM", "TransformerNN", "RecurrentNN"]
PRETRAINED = ["FinBERT", "TwitterRoBERTa", "VADER", "NTUSD-Fin"]
#%%
data = toml.load("outputs/static/model_test_scores.toml")

scores = pd.DataFrame({k: data[k]["test_scores"] for k in data})

times = pd.DataFrame({k: data[k]["times_taken"] for k in data})
times = times / 2000 * 1000  # per-sample in ms!!!

#%%
def plot_scores(score_df, ax):
    scores_melted = score_df.melt(var_name="model", value_name="score")
    sns.pointplot(
        data=scores_melted,
        x="model",
        y="score",
        order=score_df.mean().sort_values().index,
        palette=[Colors.YELLOW.value] * len(PRETRAINED)
        + [Colors.DARKBLUE.value] * len(OURS),
        linestyles=":",
        join=False,
        # capsize=0.05,
        scale=1.1,
        ci="sd",
        ax=ax,
        zorder=10,
    )

    ax.plot(
        score_df.mean().sort_values().index[: len(PRETRAINED)],
        score_df.mean().sort_values().values[: len(PRETRAINED)],
        color=Colors.YELLOW.value,
        ls="-",
        lw=3,
        zorder=0,
        label="existing models",
    )

    ax.plot(
        score_df.mean().sort_values().index[len(PRETRAINED) :],
        score_df.mean().sort_values().values[len(PRETRAINED) :],
        color=Colors.DARKBLUE.value,
        ls="-",
        lw=3,
        zorder=0,
        label="proposed models",
    )

    # ax.legend(framealpha=1)
    ax.text(
        x=1.5,
        y=0.87,
        s="Existing Models",
        weight="bold",
        color=Colors.YELLOW.value,
        ha="center",
    )
    ax.text(
        x=5.5,
        y=0.87,
        s="Proposed Models",
        weight="bold",
        color=Colors.DARKBLUE.value,
        ha="center",
    )


fig, ax = plt.subplots(figsize=(15, 5))
plot_scores(scores, ax)


ax.set_ylim(0.49, 0.86)
ax.set_ylabel("Out-of-sample ROC AUC", labelpad=15, weight="bold")
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.25, 7.25)
ax.set_xlabel("Model", weight="bold", labelpad=15)
ax.grid(axis="y", ls="--", color="black", alpha=0.25)

sns.despine(bottom=True)
plt.tight_layout()

fig.savefig(
    "outputs/plots/model_performance_pyfin.pdf", bbox_inches="tight", facecolor="white"
)


#%%
def plot_times(times_df, ax):
    times_melted_pretrained = times_df.melt(var_name="model", value_name="time").query(
        "model in @PRETRAINED"
    )
    times_melted_ours = times_df.melt(var_name="model", value_name="time").query(
        "model in @OURS"
    )

    times_melted = times_df.melt(var_name="model", value_name="time")

    sns.pointplot(
        data=times_melted_pretrained,
        x="model",
        y="time",
        order=times[PRETRAINED].mean().sort_values().index.to_list()
        + times[OURS].mean().sort_values().index.to_list(),
        color=Colors.YELLOW.value,
        join=False,
        ci="sd",
        ax=ax,
        zorder=10,
    )

    sns.pointplot(
        data=times_melted_ours,
        x="model",
        y="time",
        order=times[PRETRAINED].mean().sort_values().index.to_list()
        + times[OURS].mean().sort_values().index.to_list(),
        color=Colors.DARKBLUE.value,
        join=False,
        ci="sd",
        ax=ax,
        zorder=10,
    )

    ax.plot(
        times_df[PRETRAINED].mean().sort_values().index,
        times_df[PRETRAINED].mean().sort_values().values,
        color=Colors.YELLOW.value,
        ls="-",
        lw=3,
        zorder=0,
    )

    ax.plot(
        times_df[OURS].mean().sort_values().index,
        times_df[OURS].mean().sort_values().values,
        color=Colors.DARKBLUE.value,
        ls="-",
        lw=3,
        zorder=0,
    )

    ax.legend(framealpha=1)
    ax.text(
        x=1.5,
        y=200,
        s="Existing Models",
        weight="bold",
        color=Colors.YELLOW.value,
        ha="center",
    )
    ax.text(
        x=5.5,
        y=200,
        s="Proposed Models",
        weight="bold",
        color=Colors.DARKBLUE.value,
        ha="center",
    )


fig, ax = plt.subplots(figsize=(15, 5))

plot_times(times, ax)

ax.set_xlabel("Model", labelpad=15, weight="bold")
ax.set_ylabel("Inference time per sample (ms)", labelpad=15, weight="bold")
ax.set_yscale("log")
ax.tick_params(axis="x", length=0)
ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")

ax.grid(axis="y", ls="--", color="black", alpha=0.25)
ax.legend(frameon=False)
sns.despine(bottom=True)
plt.tight_layout()
fig.savefig(
    "outputs/plots/model_inference_time.pdf", bbox_inches="tight", facecolor="white"
)
