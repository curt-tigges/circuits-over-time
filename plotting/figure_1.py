#%%
import torch
from pathlib import Path 
import json 
import matplotlib.pyplot as plt
from plotting_utils import core_models, color_palette, steps2tokens
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = 'DejaVu Serif'

def load_results_wrapped(task: str, model: str):
    metric = 'prob_diff' if task=='greater_than' or task == 'sva' else 'logit_diff'
    if '1b' in model:
        return None
    try:
        if task != 'sva':
            data = torch.load(f'/mnt/hdd-0/circuits-over-time/results/task_performance_metrics/{model}-no-dropout/metrics.pt')
            d = data[task][metric]
            d = {k.split('step')[-1]:v.float().cpu() for k,v in d.items()}
            return d
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
        
def load_results(task: str, model: str, use_tokens=True):
    baselines = load_results_wrapped(task, model)
    if baselines is not None:
        baselines = {int(k):v for k, v in baselines.items()}
        if use_tokens:
            baselines = {steps2tokens(k): v for k, v in baselines.items()}
    return baselines

thresh = 200000000  # right now measured in tokens; if you set tokens=False, set it as a number of steps
fig, axs = plt.subplots(2,2)
fig.set_size_inches(11, 5)
def first_digit(x, pos):
    return str(x)[0]

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

#handles, labels = 
hls = axs[1,0].get_legend_handles_labels()
def str_to_params(x):
    x = x.split('-')[-1]
    num = float(x[:-1])
    if x[-1] == 'b':
        num *= 1000
    return num 

hls = sorted(list(zip(*hls)), key=lambda x: str_to_params(x[1]))
handles, labels = zip(*hls)

fig.tight_layout()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=7)
fig.savefig('../results/plots/fig1.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.show()

fig
# %%
"""
    try: 
        if '6.9b' in model or '12b' in model:
            data = torch.load(f'/mnt/hdd-0/circuits-over-time/results/task_performance_metrics/{model}-no-dropout/metrics.pt')
            d =  data[task][metric]
            d = {k.split('step')[-1]:v.cpu() for k,v in d.items()}
            return d
    except KeyError:
        pass
"""