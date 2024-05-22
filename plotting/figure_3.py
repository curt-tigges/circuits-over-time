#%%
import pandas as pd 
import numpy as np
import torch
from pathlib import Path 
import json 
import matplotlib.pyplot as plt
from plotting_utils import core_models, color_palette, steps2tokens
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = 'DejaVu Serif'

#%%
ioi_gt = torch.load('../results/task_performance_metrics/all_models_task_performance.pt')
def load_results_wrapped(task: str, model: str):
    metric = 'prob_diff' if task=='greater_than' or task == 'sva' else 'logit_diff'
    if '1b' in model:
        return None
    try:
        if task == 'ioi' or task == 'greater_than':
            return ioi_gt[model][task][metric]
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
        
def load_results(task: str, model: str, use_tokens=False):
    baselines = load_results_wrapped(task, model)
    if baselines is not None:
        baselines = {int(k):v for k, v in baselines.items()}
        if use_tokens:
            baselines = {steps2tokens(k): v for k, v in baselines.items()}
    return baselines

thresh = 200000000  # right now measured in tokens; if you set tokens=False, set it as a number of steps
#%%
fig, axs = plt.subplots(1,2)
fig.set_size_inches(11, 3)
def first_digit(x, pos):
    return str(x)[0]


#%%
ax = axs[0]
task = 'ioi'
df = pd.read_csv('/mnt/hdd-0/circuits-over-time/results/comp_swap/pythia-160m/ioi_nmh_logit_diff_results.csv')
model='pythia-160m'
baseline = load_results(task, model)
xs = sorted(baseline.keys())
xs = [x for x in xs if x > 500]
ax.plot(xs, [baseline[x] for x in xs], label='baseline', c=color_palette[model])

for source in range(18000,18001,2000):
    plot_df = df[df['source'] == source]
    ax.plot(plot_df['target'], plot_df['swapped_val'], label=f'source: {source}')


"""d = {target: {source: v for source, v in zip(df[df['target'] == target]['source'], df[df['target'] == target]['swapped_val'])} for target in set(df['target'].tolist())}
for target in d.keys():
    d[target][target] = df[df['target'] == target]['baseline_val'].tolist()[0]

targets = sorted(list(set(df['target'].tolist())))

xs = [target for target in targets if target+3000 in d[target]]
ys = [d[target][target+3000] for target in targets if target+3000 in d[target]]
ax.plot(xs, ys)"""
ax.set_title("IOI Swap")
ax.set_xscale('log')
ax.xaxis.set_tick_params(which='minor', labelsize=8)
ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
ax.xaxis.set_tick_params(which='major', pad=10)
fig.legend()
fig
#%%
ax = axs[1]
task = 'greater_than'
model='pythia-160m'
df = pd.read_csv('/mnt/hdd-0/circuits-over-time/results/comp_swap/pythia-160m/gt_induction_successor_head_prob_diff_results.csv')
baseline = load_results(task, model)
xs = sorted(baseline.keys())
xs = [x for x in xs if x > 500]
ax.plot(xs, [baseline[x] for x in xs], label='baseline', c=color_palette[model])

d = {target: {source: v for source, v in zip(df[df['target'] == target]['source'], df[df['target'] == target]['swapped_val'])} for target in set(df['target'].tolist())}
for target in d.keys():
    d[target][target] = df[df['target'] == target]['baseline_val'].tolist()[0]

targets = sorted(list(set(df['target'].tolist())))

xs = [target for target in targets if target+3000 in d[target]]
ys = [d[target][target+3000] for target in targets if target+3000 in d[target]]
ax.plot(xs, ys)

"""for source in range(2000,8001,2000):
    plot_df = df[df['source'] == source]
    ax.plot(plot_df['target'], plot_df['swapped_val'], label=f'source: {source}')"""

ax.set_title("Greater Than Swap")
ax.set_xscale('log')
ax.xaxis.set_tick_params(which='minor', labelsize=8)
ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
ax.xaxis.set_tick_params(which='major', pad=10)

fig.tight_layout()
lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
fig
#%%
for ax, task, title in zip(axs.flat, ['ioi', 'gender_pronoun', 'greater_than', 'sva'], ['Indirect Object Identification (IOI)', 'Gendered Pronoun', "Greater-Than", 'Subject-Verb Agreement (SVA)']):
    for model in core_models:
        baseline = load_results(task, model)
        if baseline is None:
            continue
        xs = sorted(baseline.keys())
        xs = [x for x in xs if x > thresh]
        ax.plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])

    ax.set_title(title)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(which='minor', labelsize=8)
    ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
    ax.xaxis.set_tick_params(which='major', pad=10)

axs[0,0].set_ylabel('Logit Difference')
axs[1,0].set_ylabel('Probability Difference')
axs[1,0].set_xlabel('# Tokens Seen')
axs[1,1].set_xlabel('# Tokens Seen')    

handles, labels = axs[1,0].get_legend_handles_labels()
fig.tight_layout()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4)
fig.savefig('../results/plots/fig1.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.show()

fig