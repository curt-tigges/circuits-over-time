#%%
import torch
from pathlib import Path 
import json 
import matplotlib.pyplot as plt
from plotting_utils import core_models, color_palette, steps2tokens
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = 'DejaVu Serif'

def load_results(task: str, model: str, use_tokens=True):
    p = Path('/mnt/hdd-0/circuits-over-time/results/algorithmic_consistency/ioi')
    model_path = p / f'{model}.pt'
    if not model_path.exists():
        return None
    baselines = torch.load(model_path)[task]
    if baselines is not None:
        baselines = {int(k):v for k, v in baselines.items()}
        if use_tokens:
            baselines = {steps2tokens(k): v for k, v in baselines.items()}
    return baselines

def first_digit(x, pos):
    return str(x)[0]

def str_to_params(x):
    x = x.split('-')[-1]
    num = float(x[:-1])
    if x[-1] == 'b':
        num *= 1000
    return num 

models = [f'pythia-{x}' for x in ['160m', '410m', '1.4b', '2.8b']]

thresh = 200000000  # right now measured in tokens; if you set tokens=False, set it as a number of steps
fig, ax = plt.subplots()
fig.set_size_inches(5,2.5)
task = 'direct_effects'

for model in core_models:
    baseline = load_results(task, model)
    if baseline is None:
        continue
    xs = sorted(baseline.keys())
    #xs = [x for x in xs if x > thresh]
    ax.plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])

    ax.set_title("Copy Suppression / Name Mover (CS/NM) Heads")
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(which='minor', labelsize=8)
    ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
    ax.xaxis.set_tick_params(which='major', pad=10)

ax.set_ylabel('% Effect from CS/NM Heads')
ax.set_xlabel('# Tokens Seen')
ax.set_ylim(0, 100)
ax.text(0.85, 0.1, 'B', transform=ax.transAxes, size=20, weight='bold')

hls = ax.get_legend_handles_labels()

hls = sorted(list(zip(*hls)), key=lambda x: str_to_params(x[1]))
handles, labels = zip(*hls)

fig.tight_layout()
#lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=7)
fig.savefig('../results/plots/fig4_p1.pdf', bbox_inches='tight')
fig
#%%
fig, axs = plt.subplots(1,2)
fig.set_size_inches(11, 2.5)

for ax, task, title, ylabel, letter in zip(axs.flat, ['s2i_effects', 'tertiary_effects'], ['S2 Inhibition (S2I) Heads', ' Tertiary (Induction / Duplicate Token) Heads'], ['% Effect from S2I Heads','% Effect from Tert. Heads'], ['C', 'D']):
    for model in models:
        baseline = load_results(task, model)
        if baseline is None:
            continue
        xs = sorted(baseline.keys())
        #xs = [x for x in xs if x > thresh]
        ax.plot(xs, [baseline[x] for x in xs], label=model, c=color_palette[model])

    ax.set_title(title)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(which='minor', labelsize=8)
    ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
    ax.xaxis.set_tick_params(which='major', pad=10)

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.set_xlabel('# Tokens Seen')
    ax.text(0.85, 0.1, letter, transform=ax.transAxes, size=20, weight='bold')

hls = ax.get_legend_handles_labels()

hls = sorted(list(zip(*hls)), key=lambda x: str_to_params(x[1]))
handles, labels = zip(*hls)

fig.tight_layout()
lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=7)
fig.savefig('../results/plots/fig4_p2.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
fig.show()

fig
# %%
