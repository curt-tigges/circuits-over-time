#%%
import torch
from pathlib import Path 
import json 
import matplotlib.pyplot as plt
from utils import core_model_color_palette, core_models, color_palette, big_core_models
from matplotlib.ticker import FuncFormatter


plt.rcParams["font.family"] = 'DejaVu Serif'

#%%
def first_digit(x, pos):
    return str(x)[0]

core_model_color_palette = color_palette
ioi_gt = torch.load('../results/task_performance_metrics/all_models_task_performance.pt')
fig, axs = plt.subplots(2,2)
fig.set_size_inches(11, 5)
for model in big_core_models:
    try:
        baseline = ioi_gt[model]['ioi']['logit_diff']
    except KeyError:
        continue
    xs = [int(x) for x in baseline.keys()]
    xs.sort()
    xs = [x for x in xs if x > 100]
    axs[0,0].plot(xs, [baseline[x] for x in xs], label=model, c=core_model_color_palette[model])

axs[0,0].set_title('Indirect Object Identification')
axs[0,0].set_ylabel('Logit Difference')

axs[0,1].set_title('Gendered Pronouns')

for model in big_core_models:
    try:
        baseline = ioi_gt[model]['greater_than']['prob_diff']
    except KeyError:
        continue

    xs = list(baseline.keys())
    xs.sort(key = lambda x: int(x))
    xs = [int(x) for x in baseline.keys()]
    xs.sort()
    xs = [x for x in xs if x > 100]
    axs[1,0].plot(xs, [baseline[x] for x in xs], label=model, c=core_model_color_palette[model])


axs[1,0].set_title('Greater-Than')
axs[1,0].set_ylabel('Probability Difference')
axs[1,0].set_xlabel('Step')

for model in core_models:
    try:
        with open(f'../results/baselines/{model}/sva.json') as f:
            baseline = json.load(f)
    except FileNotFoundError:
        try:
            with open(f'/mnt/hdd-0/circuits-over-time/results/baselines/{model}/sva.json') as f:
                baseline = json.load(f)
        except FileNotFoundError:
            continue

    xs = list(baseline.keys())
    xs.sort(key = lambda x: int(x))
    xs = [int(x) for x in baseline.keys()]
    xs.sort()
    xs = [x for x in xs if x > 100]
    axs[1,1].plot(xs, [baseline[str(x)] for x in xs], label=model, c=core_model_color_palette[model])


axs[1,1].set_title('Subject-Verb Agreement')
axs[1,1].set_xlabel('Step')


for ax in axs.flat:
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(which='minor', labelsize=8)
    ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
    ax.xaxis.set_tick_params(which='major', pad=10)

handles, labels = axs[1,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
fig.show()
fig.tight_layout()
fig
# %%
