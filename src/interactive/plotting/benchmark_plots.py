#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toml

from src.utils.plotting import Colors, set_style

set_style()
OURS = ["LogReg", "SVM", "TransformerNN", "RecurrentNN", "BERTFinetune"]
PRETRAINED = ["FinBERT", "TwitterRoBERTa", "VADER", "NTUSD-Fin"]
DISPLAY_NAMES = {
    "LogReg": "Logistic\nRegression",
    "SVM": "SVM",
    "TransformerNN": "Transformer\nNN",
    "RecurrentNN": "Recurrent\nNN",
    "BERTFinetune": "BERT\nFinetune",
    "FinBERT": "FinBERT",
    "TwitterRoBERTa": "Twitter\nRoBERTa",
    "VADER": "VADER",
    "NTUSD-Fin": "NTUSD-Fin",
}
#%%
data = toml.load("outputs/static/model_test_scores.toml")
scores = pd.DataFrame({k: data[k]["test_scores"] for k in data})

external_data = toml.load("outputs/static/model_scores_external_data.toml")
finsome_scores = pd.DataFrame(
    {k: external_data[k]["finsome_score"] for k in external_data}
)
semeval_scores = pd.DataFrame(
    {k: external_data[k]["semeval_score"] for k in external_data}
)

times = pd.DataFrame({k: data[k]["times_taken"] for k in data})
times = times / 2000 * 1000  # per-sample in ms!!!

#%%
def plot_scores(score_df, ax):
    scores_melted = score_df.melt(var_name="model", value_name="score")
    sns.pointplot(
        data=scores_melted,
        x="model",
        y="score",
        order=score_df[PRETRAINED].mean().sort_values().index.to_list()
        + score_df[OURS].mean().sort_values().index.to_list(),
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
        score_df[PRETRAINED].mean().sort_values().index,
        score_df[PRETRAINED].mean().sort_values().values,
        color=Colors.YELLOW.value,
        ls="-",
        lw=3,
        zorder=5,
        label="existing models",
    )

    ax.plot(
        score_df[OURS].mean().sort_values().index,
        score_df[OURS].mean().sort_values().values,
        color=Colors.DARKBLUE.value,
        ls="-",
        lw=3,
        zorder=5,
        label="proposed models",
    )

    ax.text(
        x=1.5,
        y=score_df.max(axis=1).values.ravel()[0] + 0.02,
        s="Existing Models",
        weight="bold",
        color=Colors.YELLOW.value,
        ha="center",
        zorder=10,
    )

    ax.text(
        x=6,
        y=score_df.max(axis=1).values.ravel()[0] + 0.02,
        s="Proposed Models",
        weight="bold",
        color=Colors.DARKBLUE.value,
        ha="center",
    )


fig, ax = plt.subplots(figsize=(15, 5))
ax.grid(axis="y", ls="--", color="black", alpha=0.25, zorder=0)
plot_scores(scores, ax)


ax.set_ylim(0.49, 0.86)
ax.set_ylabel("Out-of-sample ROC AUC", labelpad=15, weight="bold")
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.25, 8.25)
ax.set_xlabel("Model", weight="bold", labelpad=15)
ax.set_xticklabels([DISPLAY_NAMES.get(name.get_text()) for name in ax.get_xticklabels()])

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
        zorder=5,
    )

    ax.plot(
        times_df[OURS].mean().sort_values().index,
        times_df[OURS].mean().sort_values().values,
        color=Colors.DARKBLUE.value,
        ls="-",
        lw=3,
        zorder=5,
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
        x=6,
        y=200,
        s="Proposed Models",
        weight="bold",
        color=Colors.DARKBLUE.value,
        ha="center",
    )


fig, ax = plt.subplots(figsize=(15, 5))
ax.grid(axis="y", ls="--", color="black", alpha=0.25, zorder=0)

plot_times(times, ax)

ax.set_xlabel("Model", labelpad=15, weight="bold")
ax.set_ylabel("Inference time per sample (ms)", labelpad=15, weight="bold")
ax.set_yscale("log")
ax.tick_params(axis="x", length=0)
ax.set_xticklabels([DISPLAY_NAMES.get(name.get_text()) for name in ax.get_xticklabels()])
ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.1f}")

ax.legend(frameon=False)
sns.despine(bottom=True)
plt.tight_layout()
fig.savefig(
    "outputs/plots/model_inference_time.pdf", bbox_inches="tight", facecolor="white"
)


#%%
fig, axes = plt.subplots(2, 1, figsize=(15, 10))
axes[0].grid(axis="y", ls="--", color="black", alpha=0.25, zorder=0)
axes[1].grid(axis="y", ls="--", color="black", alpha=0.25, zorder=0)
plot_scores(finsome_scores, axes[0])
plot_scores(semeval_scores, axes[1])

axes[0].set_ylim(0.49, 0.76)
axes[0].set_ylabel("ROC AUC on Fin-SoMe", labelpad=15, weight="bold")
axes[0].tick_params(axis="x", length=0)
axes[0].set_xlim(-0.25, 8.25)
axes[0].set_xlabel("Model", weight="bold", labelpad=15)
axes[0].set_xticklabels(
    [DISPLAY_NAMES.get(name.get_text()) for name in axes[0].get_xticklabels()]
)
axes[0].set_title("Fin-SoMe", weight="bold")
axes[1].set_title("SemEval", weight="bold")

axes[1].set_ylim(0.49, 0.77)
axes[1].set_ylabel("ROC AUC on SemEval", labelpad=15, weight="bold")
axes[1].tick_params(axis="x", length=0)
axes[1].set_xlim(-0.25, 8.25)
axes[1].set_xlabel("Model", weight="bold", labelpad=15)
axes[1].set_xticklabels(
    [DISPLAY_NAMES.get(name.get_text()) for name in axes[1].get_xticklabels()]
)


sns.despine(bottom=True)
plt.tight_layout()
plt.subplots_adjust(bottom=0, top=1)


fig.savefig(
    "outputs/plots/model_performance_finsome.pdf",
    bbox_inches="tight",
    facecolor="white",
)
