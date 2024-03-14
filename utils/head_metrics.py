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
def S2I_head_metrics(model: HookedTransformer, ioi_dataset, potential_s2i_list: List[Tuple[int, int]],  NMH_list: List[Tuple[int, int]], batch_size): 
# a head is an S2I head if it meets 2-4 conditions
# 1. ablating the head->NM<q> path hurts logit diff
# - specifically, ablating them should increase NMH attn to S1
# 2. the head attends mainly to S2 from the END position
# 3.
    
    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA")
    abc_dataset.__class__ = BatchIOIDataset
    ioi_dataloader = DataLoader(ioi_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device))
    abc_dataloader = DataLoader(abc_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device))

    potential_s2i_layers, potential_s2i_heads = (torch.tensor(x, device=model.cfg.device) for x in zip(*potential_s2i_list))

    s2i_hook_set = set([f'blocks.{layer}.attn.hook_result' for layer, _ in potential_s2i_list])

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
            def make_s2i(layer, head):
                return Node(f'blocks.{layer}.attn.hook_z', layer, head)
            def make_nmh(layer, head):
                print(f"Making NMH node for layer {layer} and head {head}")
                return Node(f'blocks.{layer}.hook_q_input', layer, head)
            
            temp_NMH_list = [make_nmh(nmh_layer, nmh_head) for nmh_layer, nmh_head in NMH_list]
            print(temp_NMH_list)
            
            new_logits = path_patch(model, toks, abc_batch['toks'], make_s2i(s2i_layer, s2i_head), temp_NMH_list, lambda x: x, seq_pos=end_pos)[torch.arange(len(toks)), end_pos]

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

    return {
        'baseline_logit_diffs': baseline_logit_diffs,
        'end_s2_attention_values': end_s2_attention_values,
        'baseline_nmh_s1_attention_values': baseline_nmh_s1_attention_values,
        'new_logit_diffs': new_logit_diffs,
        'new_nmh_s1_attention_values': new_nmh_s1_attention_values
    }