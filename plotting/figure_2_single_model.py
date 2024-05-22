#%%
from collections import Counter 
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path 
from typing import Dict
import json 
import matplotlib.pyplot as plt
from plotting_utils import core_models, color_palette, steps2tokens
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = 'DejaVu Serif'

def clean_outliers(checkpoint_dict: Dict[int, torch.Tensor], min_value: float, max_value: float) -> Dict[int, torch.Tensor]:
    for checkpoint in checkpoint_dict.keys():
        tensor = checkpoint_dict[checkpoint]
        # Set values outside the range to 0.0
        tensor = torch.where(tensor < min_value, torch.tensor(0.0), tensor)
        tensor = torch.where(tensor > max_value, torch.tensor(0.0), tensor)
        checkpoint_dict[checkpoint] = tensor
    return checkpoint_dict


def load_results_wrapped(head: str, model: str):
    p = Path('/mnt/hdd-0/circuits-over-time/results/components')
    model_path = p/model
    try:
        if head == 'successor':
            data = torch.load(model_path / 'successor_heads_over_time.pt')
            data = {data['checkpoints'][i]:data['data'][i] for i in range(len(data['checkpoints']))}
            steps = sorted(list(data.keys()))
            head_scores = torch.stack([data[step].cpu() for step in steps])
            layers, heads = (x.tolist() for x in torch.where(head_scores.max(dim=0).values >= (head_scores.max() * 0.4)))
            all_heads = set(zip(layers, heads))
            

            return steps, all_heads, head_scores
        elif head == 'induction':
            data = torch.load(model_path / "full_model_components_over_time.pt")
            steps = sorted(list(data.keys()))
            head_scores = torch.stack([data[step]['tertiary_head_scores']['induction_scores'].cpu() for step in steps])

            layers, heads = (x.tolist() for x in torch.where(head_scores.max(dim=0).values >= (head_scores.max() * 0.25)))


            all_heads = set(zip(layers, heads))
            return steps, all_heads, head_scores
        elif head == 'copy_suppression':
            data = torch.load(model_path / 'whole_model_cspa.pt')
            data = clean_outliers(data, 0.0, 1.0)
            steps = sorted(list(data.keys()))
            head_scores = torch.stack([data[step].cpu() for step in steps])

            layers, heads = (x.tolist() for x in torch.where(head_scores.max(dim=0).values >= (head_scores.max() * 0.15)))

            all_heads = set(zip(layers, heads))
            return steps, all_heads, head_scores * 0.01
        elif head == 'name_mover':
            data = torch.load(model_path / 'early_whole_model_copy_scores.pt')
            steps = sorted([k for k in data.keys() if data[k] is not None])
            head_scores = torch.stack([data[step].cpu() for step in steps])

            layers, heads = (x.tolist() for x in torch.where(head_scores.max(dim=0).values >= (head_scores.max() * 0.25)))

            all_heads = set(zip(layers, heads)) 
            return steps, all_heads, head_scores * 0.01
        else:
            raise ValueError(f"Got invalid head {head}")

    except FileNotFoundError:
        print("couldn't find file for", head,  "in", model_path)
        return None

def load_results(head_type: str, model: str, use_tokens=True):
    baselines = load_results_wrapped(head_type, model)
    if baselines is not None and use_tokens:
        return [steps2tokens(x) for x in baselines[0]], baselines[1], baselines[2]
    return baselines

def first_digit(x, pos):
    return str(x)[0]

# get all nodes that were at some point in the circuit
def get_candidates(task: str):
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


thresh = 1  # right now measured in tokens; if you set tokens=False, set it as a number of steps

# change core_models if you want to iterate over something else (e.g. 160m model variants)
for model in core_models:
    if '1b' in model:
        continue
    fig, axs = plt.subplots(2,2)
    fig.set_size_inches(11, 5)

    gt_cand = get_candidates('greater_than')
    ioi_cand = get_candidates('ioi')
    both_cand = {**gt_cand, **ioi_cand} if (gt_cand is not None) and (ioi_cand is not None) else None

    style_dict = {}
    colors = list(color_palette.values())
    linestyles = ['solid', 'dashed', 'dotted']
    all_styles = [(c, ls) for ls in linestyles for c in colors]
    info_dict = {task:{} for task in ['successor', 'induction', 'copy_suppression', 'name_mover']}

    for ax, head_type, title, metric, candidate_nodes in zip(axs.flat, ['successor', 'induction', 'copy_suppression', 'name_mover'], ['Successor Heads', 'Induction Heads', "Copy Suppression Heads", 'Name Mover Heads'], ['Succession Score', 'Induction Score', 'CSPA Score', 'Copy Score'], [gt_cand, gt_cand, ioi_cand, ioi_cand]):
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.set_xscale('log')
        ax.xaxis.set_tick_params(which='minor', labelsize=8)
        ax.xaxis.set_minor_formatter(FuncFormatter(first_digit))
        ax.xaxis.set_tick_params(which='major', pad=10)
        
        baseline = load_results(head_type, model)
        if baseline is None:
            continue
        x_axis, all_heads, scores = baseline

        if candidate_nodes is not None:
            candidate_nodes = Counter({k:v for k,v in candidate_nodes.items() if k in all_heads})
            all_heads = [k for k,v in candidate_nodes.most_common(5)]

        for layer, head in all_heads:
            #if (layer, head) not in candidate_nodes:
            #    continue
            if (layer, head) in style_dict:
                c, ls = style_dict[(layer, head)]
            else:
                c, ls = all_styles[0]
                all_styles = all_styles[1:]
                style_dict[layer, head] = (c, ls)

            info_dict[head_type][(layer, head)] = (c, ls)
            head_scores = scores[:, layer, head].numpy()
            ax.plot(x_axis, head_scores, label=f"({layer}, {head})", color=c, linestyle=ls)

    axs[1,0].set_xlabel('# Tokens Seen')
    axs[1,1].set_xlabel('# Tokens Seen')    

    handles_labels = [(handle, label) for ax in axs.flat for handle, label in zip(*ax.get_legend_handles_labels())] 
    labels = [hl[1] for hl in handles_labels]
    handles_labels = [hl for i, hl in enumerate(handles_labels) if hl[1] not in labels[:i]]
    # sort both labels and handles by labels
    handles, labels = zip(*sorted(handles_labels, key=lambda t: eval(t[1])))
    print(model)
    #handles, labels = axs[1,0].get_legend_handles_labels()
    fig.tight_layout()
    lgd = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6)
    os.makedirs(f'/mnt/hdd-0/circuits-over-time/results/plots/fig2_single_model/', exist_ok=True)
    fig.savefig(f'/mnt/hdd-0/circuits-over-time/results/plots/fig2_single_model/{model}.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')
    fig

# %%