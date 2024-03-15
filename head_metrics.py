#%%
from typing import Tuple, List
from functools import partial

import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from einops import rearrange

from utils.data_processing import (
    load_edge_scores_into_dictionary,
    get_ckpts,
    load_metrics,
    get_ckpts
)
from path_patching_cm.ioi_dataset import IOIDataset
from path_patching_cm.path_patching import Node, path_patch

#%%
def convert_head_names_to_tuple(head_name):
    head_name = head_name.replace('a', '')
    head_name = head_name.replace('h', '')
    layer, head = head_name.split('.')
    return (int(layer), int(head))

def collate_fn(ds, device):
    if not ds:
        return {}
    return {k: torch.stack([d[k] for d in ds], dim=0).to(device) for k in ds[0].keys()}

class BatchIOIDataset(IOIDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.N
    
    def __getitem__(self, i):
        return {'toks':self.toks[i], 'io_token_id': torch.tensor(self.io_tokenIDs[i]), 's_token_id': torch.tensor(self.s_tokenIDs[i]), **{f'{k}_pos':v[i] for k, v in self.word_idx.items()}}
    
##%

def make_s2i(layer, head):
    return Node(f'blocks.{layer}.attn.hook_z', layer, head)
def make_nmh(layer, head):
    return Node(f'blocks.{layer}.hook_q_input', layer, head)
def S2I_head_metrics(model: HookedTransformer, ioi_dataset, potential_s2i_list: List[Tuple[int, int]],  NMH_list: List[Tuple[int, int]], batch_size): 
# a head is an S2I head if it meets 2-4 conditions
# 1. ablating the head->NM<q> path hurts logit diff
# - specifically, ablating them should increase NMH attn to S1
# 2. the head attends mainly to S2 from the END position
    
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    abc_dataset.__class__ = BatchIOIDataset
    ioi_dataloader = DataLoader(ioi_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device))
    abc_dataloader = DataLoader(abc_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device))

    potential_s2i_layers, potential_s2i_heads = (torch.tensor(x, device=model.cfg.device) for x in zip(*potential_s2i_list))

    NMH_layers, NMH_heads = (torch.tensor(x, device=model.cfg.device) for x in zip(*NMH_list))

    baseline_logit_diffs = []
    new_logit_diffs = []
    end_s2_attention_values = []
    baseline_nmh_s1_attention_values = []
    new_nmh_s1_attention_values = []
    for batch, abc_batch in zip(ioi_dataloader, abc_dataloader):
        toks = batch['toks']
        end_pos = batch['end_pos']
        s2_pos = batch['S2_pos']
        s1_pos = batch['S1_pos']
        s_token_ids = batch['s_token_id']
        io_token_ids = batch['io_token_id']

        cache, caching_hooks, _ = model.get_caching_hooks(lambda name: 'hook_pattern' in name)
        with model.hooks(caching_hooks):
            logits = model(toks)[torch.arange(len(toks)), end_pos]

        s_logits = logits[torch.arange(len(toks)), s_token_ids]
        io_logits = logits[torch.arange(len(toks)), io_token_ids]
        baseline_logit_diff = io_logits - s_logits 
        baseline_logit_diffs.append(baseline_logit_diff)
        
        attention_patterns = torch.stack([cache[f'blocks.{n}.attn.hook_pattern'] for n in range(model.cfg.n_layers)])  #layer, batch, head, query, key

        end_s2_attention_value = attention_patterns[:, torch.arange(len(toks)), :, end_pos, s2_pos] # batch, layer, head
        end_s2_attention_value = end_s2_attention_value[:, potential_s2i_layers, potential_s2i_heads]
        end_s2_attention_values.append(end_s2_attention_value)

        nmh_s1_attention_value_baseline = attention_patterns[:, torch.arange(len(toks)), :, end_pos, s1_pos] # batch layer head
        nmh_s1_attention_value_baseline = nmh_s1_attention_value_baseline[:, NMH_layers, NMH_heads]
        baseline_nmh_s1_attention_values.append(nmh_s1_attention_value_baseline)

        new_logit_diff = []
        new_nmh_s1_attention_value = []
        for potential_s2i in potential_s2i_list:
            s2i_layer, s2i_head = potential_s2i
            # do interventions          
            new_logits = path_patch(model, toks, abc_batch['toks'], make_s2i(s2i_layer, s2i_head), [make_nmh(nmh_layer, nmh_head) for nmh_layer, nmh_head in NMH_list], lambda x: x, seq_pos=end_pos)[torch.arange(len(toks)), end_pos]

            # there's maybe a way to get both the logits and the cache in one go, but I don't know how to use this path patch fn
            mixed_cache = path_patch(model, toks, abc_batch['toks'], make_s2i(s2i_layer, s2i_head), [make_nmh(nmh_layer, nmh_head) for nmh_layer, nmh_head in NMH_list], lambda x: x, seq_pos=end_pos, apply_metric_to_cache=True, names_filter_for_cache_metric=lambda name: 'hook_pattern' in name)

            # record new logit diff
            new_s_logits = new_logits[torch.arange(len(toks)), s_token_ids]
            new_io_logits = new_logits[torch.arange(len(toks)), io_token_ids]
            ablated_logit_diff = new_io_logits - new_s_logits 
            new_logit_diff.append(ablated_logit_diff)

            # record new attn patterns
            new_attention_patterns = torch.stack([mixed_cache[f'blocks.{n}.attn.hook_pattern'] for n in range(model.cfg.n_layers)])
            new_nmh_s1_attention_val = new_attention_patterns[:, torch.arange(len(toks)), :, end_pos, s1_pos] # batch layer head
            new_nmh_s1_attention_val = new_nmh_s1_attention_val[:, NMH_layers, NMH_heads]
            new_nmh_s1_attention_value.append(new_nmh_s1_attention_val)

        new_logit_diff = torch.stack(new_logit_diff, dim=1).view(len(toks), len(potential_s2i_list))
        new_logit_diffs.append(new_logit_diff)

        new_nmh_s1_attention_value = torch.stack(new_nmh_s1_attention_value, dim=1).view(len(toks), len(potential_s2i_list), len(NMH_list))
        new_nmh_s1_attention_values.append(new_nmh_s1_attention_value)

    baseline_logit_diffs = torch.cat(baseline_logit_diffs, dim=0)
    end_s2_attention_values = torch.cat(end_s2_attention_values, dim=0)
    baseline_nmh_s1_attention_values = torch.cat(baseline_nmh_s1_attention_values, dim=0)
    new_logit_diffs = torch.cat(new_logit_diffs, dim=0)
    new_nmh_s1_attention_values = torch.cat(new_nmh_s1_attention_values, dim=0)

    return baseline_logit_diffs, end_s2_attention_values, baseline_nmh_s1_attention_values, new_logit_diffs, new_nmh_s1_attention_values

#%%
def S2I_token_pos(model: HookedTransformer, ioi_dataset: IOIDataset, S2I_list: List[Tuple[int, int]],  NMH_list: List[Tuple[int, int]], batch_size): 
    patch_dataset_names = ['token_same_pos_same', 'token_diff_pos_same', 'token_oppo_pos_same', 'token_same_pos_oppo', 'token_diff_pos_oppo', 'token_oppo_pos_oppo']
    
    random_name_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYY, BAB->YXY") # token diff, pos same
    io_s1_dataset = ioi_dataset.gen_flipped_prompts("ABB->BAB, BAB->ABB") # token same, pos oppo
    io_s2_dataset = ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA") # token oppo, pos oppo

    random_name_io_s1_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYX, BAB->XYY") # token diff pos oppo
    # we omit this one, since it's actually the same
    # random_name_io_s2_dataset = ioi_dataset.gen_flipped_prompts("ABB->XYX, BAB->XYY") # token diff pos oppo
    io_s1_io_s2_dataset = ioi_dataset.gen_flipped_prompts("ABB->BAA, BAB->ABA") # token oppo pos same

    patch_datasets = [ioi_dataset, random_name_dataset, io_s1_io_s2_dataset, io_s1_dataset,random_name_io_s1_dataset, io_s2_dataset]
    for dataset in patch_datasets:
        dataset.__class__ = BatchIOIDataset
    ioi_dataloader = DataLoader(ioi_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device))
    patch_dataloaders = [DataLoader(dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device)) for dataset in patch_datasets]

    NMH_layers, NMH_heads = (torch.tensor(x, device=model.cfg.device) for x in zip(*NMH_list))

    logit_diffs = {name: [] for name in patch_dataset_names}
    s1_attention_values = {name: [] for name in patch_dataset_names}
    s2_attention_values = {name: [] for name in patch_dataset_names}
    io_attention_values = {name: [] for name in patch_dataset_names}

    S2I_nodes = [make_s2i(layer, head) for layer, head in S2I_list]
    NMH_nodes = [make_nmh(layer, head) for layer, head in NMH_list]

    for batch, *patch_batches in zip(ioi_dataloader, *patch_dataloaders):
        toks = batch['toks']
        io_pos = batch['IO_pos']
        end_pos = batch['end_pos']
        s2_pos = batch['S2_pos']
        s1_pos = batch['S1_pos']
        s_token_ids = batch['s_token_id']
        io_token_ids = batch['io_token_id']

        cache, caching_hooks, _ = model.get_caching_hooks(lambda name: 'hook_pattern' in name)
        with model.hooks(caching_hooks):
            logits = model(toks)[torch.arange(len(toks)), end_pos]

        s_logits = logits[torch.arange(len(toks)), s_token_ids]
        io_logits = logits[torch.arange(len(toks)), io_token_ids]

        baseline_logit_diff = io_logits - s_logits 
        logit_diffs['token_same_pos_same'].append(baseline_logit_diff)
        
        attention_patterns = torch.stack([cache[f'blocks.{n}.attn.hook_pattern'] for n in range(model.cfg.n_layers)])  #layer, batch, head, query, key
        attention_patterns_by_head = attention_patterns[NMH_layers, :, NMH_heads]
        nmh_s1_attention_value_baseline = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, s1_pos] # batch layer head
        nmh_s2_attention_value_baseline = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, s2_pos] # batch layer head
        nmh_io_attention_value_baseline = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, io_pos] # batch layer head
        s1_attention_values['token_same_pos_same'].append(nmh_s1_attention_value_baseline.transpose(0,1))
        s2_attention_values['token_same_pos_same'].append(nmh_s2_attention_value_baseline.transpose(0,1))
        io_attention_values['token_same_pos_same'].append(nmh_io_attention_value_baseline.transpose(0,1))

        for patch_batch, patch_dataset_name in zip(patch_batches, patch_dataset_names): 
            if patch_dataset_name == 'token_same_pos_same':
                continue  

            new_logits = path_patch(model, toks, patch_batch['toks'], S2I_nodes, NMH_nodes, lambda x: x, seq_pos=end_pos)[torch.arange(len(toks)), end_pos]

            # there's maybe a way to get both the logits and the cache in one go, but I don't know how to use this path patch fn
            mixed_cache = path_patch(model, toks, patch_batch['toks'], S2I_nodes, NMH_nodes, lambda x: x, seq_pos=end_pos, apply_metric_to_cache=True, names_filter_for_cache_metric=lambda name: 'hook_pattern' in name)

            # record new logit diff
            new_s_logits = new_logits[torch.arange(len(toks)), s_token_ids]
            new_io_logits = new_logits[torch.arange(len(toks)), io_token_ids]
            ablated_logit_diff = new_io_logits - new_s_logits 
            logit_diffs[patch_dataset_name].append(ablated_logit_diff)

            # record new attn patterns
            new_attention_patterns = torch.stack([mixed_cache[f'blocks.{n}.attn.hook_pattern'] for n in range(model.cfg.n_layers)])

            attention_patterns_by_head = new_attention_patterns[NMH_layers, :, NMH_heads]
            nmh_s1_attention_value_baseline = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, s1_pos] # batch layer head
            nmh_s2_attention_value_baseline = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, s2_pos] # batch layer head
            nmh_io_attention_value_baseline = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, io_pos] # batch layer head
            s1_attention_values[patch_dataset_name].append(nmh_s1_attention_value_baseline.transpose(0,1))
            s2_attention_values[patch_dataset_name].append(nmh_s2_attention_value_baseline.transpose(0,1))
            io_attention_values[patch_dataset_name].append(nmh_io_attention_value_baseline.transpose(0,1))

    for d in [logit_diffs, s1_attention_values, s2_attention_values, io_attention_values]:
        for patch_dataset_name in patch_dataset_names:
            d[patch_dataset_name] = torch.cat(d[patch_dataset_name], dim=0)

    return logit_diffs, s1_attention_values, s2_attention_values, io_attention_values
#%%
TASK = 'ioi'
model_name = 'EleutherAI/pythia-160m'
device = 'cuda'
ckpt = 143000
cache_dir = None
kwargs = {"cache_dir": cache_dir} if cache_dir is not None else {}
size=100
seed = 42
batch_size = 8
torch.set_grad_enabled(False)

model = HookedTransformer.from_pretrained(
            model_name, 
            checkpoint_value=int(ckpt),
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            #dtype=torch.bfloat16,
            **kwargs,
        )

model.cfg.use_attn_result = True
model.cfg.use_attn_in = True 
model.cfg.use_split_qkv_input = True

ioi_dataset = BatchIOIDataset(
        prompt_type="mixed",
        N=size,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=str(device)
    )
# ioi_dataset, abc_dataset = generate_data_and_caches(model, size, verbose=True)
#%%
folder_path = f'results/graphs/pythia-160m/{TASK}'
df = load_edge_scores_into_dictionary(folder_path)

directory_path = 'results'
perf_metrics = load_metrics(directory_path)

ckpts = get_ckpts(schedule="exp_plus_detail")

# filter everything before 1000 steps
df = df[df['checkpoint'] >= 1000]

df[['source', 'target']] = df['edge'].str.split('->', expand=True)
len(df['target'].unique())
#%%
# you need to fill in the name mover heads here!
# Either manually, or via some earlier automatic finding process
name_mover_heads = [(8, 1), (8, 2), (8, 10), (9, 4), (9, 6), (10, 7)] #[(8, 10), (8, 2)]
targeting_nmh = np.logical_or.reduce(np.array([df['target'] == f'a{layer}.h{head}<q>' for layer, head in name_mover_heads]))
candidate_s2i = df[targeting_nmh]
candidate_s2i = candidate_s2i[candidate_s2i['in_circuit'] == True]

candidate_list = candidate_s2i[candidate_s2i['checkpoint']==ckpt]['source'].unique().tolist()
candidate_list = [convert_head_names_to_tuple(c) for c in candidate_list if (c[0] != 'm' and c != 'input')]
#%%
baseline_logit_diffs, end_s2_attention_values, baseline_nmh_s1_attention_values, new_logit_diffs, new_nmh_s1_attention_values = S2I_head_metrics(model, ioi_dataset, candidate_list, name_mover_heads, batch_size)

# %%
# our three measures are thus:

# attention (higher is better)
s2i_s2_attention = end_s2_attention_values.mean(0)

# logit diff change (lower is better)
logit_diff_change = (new_logit_diffs - baseline_logit_diffs.unsqueeze(1)).mean(0)

# NMH s1 attention change (higher is better)
nmh_s1_attention_change = (new_nmh_s1_attention_values - baseline_nmh_s1_attention_values.unsqueeze(1)).mean(0).mean(-1)
# %%
print(candidate_list)
print(s2i_s2_attention)
print(logit_diff_change)
print(nmh_s1_attention_change)
# %%
s2i_heads = [(7,9), (7,2), (6,6), (6,5),]
logit_diffs, s1_attention_values, s2_attention_values, io_attention_values = S2I_token_pos(model, ioi_dataset, s2i_heads, name_mover_heads, batch_size)
# %%
patch_dataset_names = ['token_same_pos_same', 'token_diff_pos_same', 'token_oppo_pos_same', 'token_same_pos_oppo', 'token_diff_pos_oppo', 'token_oppo_pos_oppo']
for dataset_name in patch_dataset_names:
    print(dataset_name)
    print(logit_diffs[dataset_name].mean())
# %%
for dataset_name in patch_dataset_names:
    print(dataset_name)
    # should be high with pos = same, low with pos = diff
    print(io_attention_values[dataset_name].mean())
    # should be low with pos = same, high with pos = diff
    print(s1_attention_values[dataset_name].mean())
    # shouldn't change much
    print(s2_attention_values[dataset_name].mean())
# %%
