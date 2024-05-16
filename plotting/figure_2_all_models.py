#%%
from collections import Counter 
from pathlib import Path 
import json 

import pandas as pd
import torch
import matplotlib.pyplot as plt
from plotting_utils import core_models, color_palette, steps2tokens
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = 'DejaVu Serif'

def load_results_wrapped(head: str, model: str):
    p = Path('../results/components')
    model_path = p/model
    if '1b' in model:
        return None
    try:
        if head == 'successor':
            data = torch.load(model_path / 'successor_heads_over_time.pt')
            steps = sorted(list(data.keys()))
            head_scores = torch.stack([data[step].cpu() for step in steps])
        
        elif head == 'induction':
            data = torch.load(model_path / "full_model_components_over_time.pt")
            steps = sorted(list(data.keys()))
            head_scores = torch.stack([data[step]['tertiary_head_scores']['induction_scores'].cpu() for step in steps])

        elif head == 'copy_suppression':
            data = torch.load(f'../results/components/{model}/early_whole_model_copy_scores.pt')
            steps = sorted(list(data.keys()))
            head_scores = torch.stack([data[step].cpu() for step in steps]) * 0.01
            
        elif head == 'name_mover':
            data = torch.load(f'/mnt/hdd-0/circuits-over-time/results/components/{model}/components_over_time.pt')
            steps = sorted([k for k in data.keys() if data[k]['direct_effect_scores'] is not None])
            head_scores = torch.stack([data[step]['direct_effect_scores']['copy_scores'].cpu() for step in steps]) * 0.01
            
        else:
            raise ValueError(f"Got invalid head {head}")

        layers, heads = (x.tolist() for x in torch.where(head_scores.max(dim=0).values >= (head_scores.max() * 0.25)))

        all_heads = set(zip(layers, heads))
        all_heads = sorted(list(all_heads), key=lambda lh: head_scores[:, lh[0], lh[1]].max(), reverse=True)
        return steps, all_heads, head_scores

    except FileNotFoundError:
        print("couldn't find file for", head,  "in", model_path)
        return None

        
def load_results(head_type: str, model: str, use_tokens=True):
    baselines = load_results_wrapped(head_type, model)
    if baselines is not None and use_tokens:
        return [steps2tokens(x) for x in baselines[0]], baselines[1], baselines[2]
    return baselines

thresh = 200000000  # right now measured in tokens; if you set tokens=False, set it as a number of steps
fig, axs = plt.subplots(2,2)
fig.set_size_inches(11, 5)
def first_digit(x, pos):
    return str(x)[0]

def get_candidates(task: str, model:str):
    try: 
        df = pd.read_feather(f'/mnt/hdd-0/circuits-over-time/results/graphs/{model}/{task}/in_circuit_edges_faithful.feather')
    except FileNotFoundError:
        try:
            df = pd.read_feather(f'/mnt/hdd-0/circuits-over-time/results/graphs/{model}/{task}/in_circuit_edges.feather')
        except FileNotFoundError:
            return None
        
    df = df[df['in_circuit']]
    df['in_node'] = [x.split('->')[0] for x in df['edge']]
    df['out_node'] = [x.split('->')[1].split('<')[0] for x in df['edge']]
    candidate_nodes = set(df['in_node']) & set(df['out_node'])
    count_dict = {cd: len(set(df[(df['in_node'] == cd) | (df['out_node'] ==cd)]['checkpoint'])) for cd in candidate_nodes}
    # turns out this filtering isn't very effective
    count_dict = {(int(node.split('.')[0][1:]), int(node.split('.')[1][1:])):v for node,v in count_dict.items() if '.' in node}
    return count_dict

for ax, head_type, title, metric in zip(axs.flat, ['successor', 'induction', 'copy_suppression', 'name_mover'], ['Successor Heads', 'Induction Heads', "Copy Suppression Heads", 'Name Mover Heads'], ['Succession Score', 'Induction Score', 'CSPA Score', 'Copy Score']):
    for model in core_models:
        task = 'greater_than' if head_type in {'successor', 'induction'} else 'ioi'

        baseline = load_results(head_type, model)
        if baseline is None:
            continue

        x_axis, all_heads, scores = baseline

        candidate_nodes = get_candidates(task, model)
        if candidate_nodes is None:
            print("couldn't find candidates for", model, head_type, task)
            layer, head = all_heads[0]
        else:
            all_heads = [x for x in all_heads if x in candidate_nodes]
            candidate_nodes = Counter({k:v for k,v in candidate_nodes.items() if k in all_heads})
            most_common = candidate_nodes.most_common()
            if len(most_common) >= 2 and most_common[0][1] == most_common[1][1]:
                layer, head = most_common[1][0]
            else:
                layer, head = most_common[0][0]
            #layer, head = all_heads[0]

            """if head_type == 'successor' and '410' in model:
            print(candidate_nodes)
            for layer, head in all_heads:
                head_scores = scores[:, layer, head].numpy()
                ax.plot(x_axis, head_scores, label=f'{model} {layer,head}', color=color_palette[model])
            continue"""
        
        head_scores = scores[:, layer, head].numpy()
        ax.plot(x_axis, head_scores, label=model, color=color_palette[model])

    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xscale('log')
    ax.xaxis.set_tick_params(which='minor', labelsize=8)
    ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
    ax.xaxis.set_tick_params(which='major', pad=10)

axs[1,0].set_xlabel('# Tokens Seen')
axs[1,1].set_xlabel('# Tokens Seen')    

hls = axs[0,0].get_legend_handles_labels()

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
fig.savefig('../results/plots/fig2_all_models.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
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

"""
        if head_type == 'successor' and '6.9' in model:
            for layer, head in all_heads:
                head_scores = scores[:, layer, head].numpy()
                ax.plot(x_axis, head_scores, label=f'{model} {layer,head}', color=color_palette[model])
            continue"""