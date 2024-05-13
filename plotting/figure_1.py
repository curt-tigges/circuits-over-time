#%%
import torch
from pathlib import Path 
import json 
import matplotlib.pyplot as plt
from plotting_utils import core_models, color_palette
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = 'DejaVu Serif'

ioi_gt = torch.load('../results/task_performance_metrics/all_models_task_performance.pt')
def load_results(task: str, model: str):
    try:
        if task == 'ioi':
            return ioi_gt[model][task]['logit_diff']
        elif task == 'greater_than':
            return ioi_gt[model][task]['prob_diff']
    except KeyError:
        pass 
    try:
        with open(f'../results/baselines/{model}/{task}.json') as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            with open(f'/mnt/hdd-0/circuits-over-time/results/baselines/{model}/{task}.json') as f:
                return json.load(f)
        except FileNotFoundError:
            print("found none for", task, model)
            return None


fig, axs = plt.subplots(2,2)
fig.set_size_inches(11, 5)
for model in core_models:
    baseline = load_results('ioi', model)
    if baseline is None:
        continue
    baseline = {int(k):v for k, v in baseline.items()}
    xs = sorted(baseline.keys())
    xs = [x for x in xs if x > 100]
    axs[0,0].plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])

axs[0,0].set_title('Indirect Object Identification')
axs[0,0].set_ylabel('Logit Difference')

for model in core_models:
    baseline = load_results('gender_pronoun', model)
    if baseline is None:
        continue
    baseline = {int(k):v for k, v in baseline.items()}
    xs = sorted(baseline.keys())
    xs = [x for x in xs if x > 100]
    axs[0,1].plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])
axs[0,1].set_title('Gendered Pronouns')

for model in core_models:
    baseline = load_results('greater_than', model)
    if baseline is None:
        continue
    baseline = {int(k):v for k, v in baseline.items()}
    xs = sorted(baseline.keys())
    xs = [x for x in xs if x > 100]
    axs[1,0].plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])


axs[1,0].set_title('Greater-Than')
axs[1,0].set_ylabel('Probability Difference')
axs[1,0].set_xlabel('Step')

for model in core_models:
    baseline = load_results('sva', model)
    if baseline is None:
        continue
    baseline = {int(k):v for k, v in baseline.items()}
    xs = sorted(baseline.keys())
    xs = [x for x in xs if x > 100]
    if '160m' in model:
        print("plotting 160m")
    axs[1,1].plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])
axs[1,1].set_title('Subject-Verb Agreement')
axs[1,1].set_xlabel('Step')


def first_digit(x, pos):
    return str(x)[0]

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
