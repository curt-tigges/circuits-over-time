from typing import Tuple, List
from functools import partial

import numpy as np 
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

import einops

from utils.backup_analysis import compute_copy_score

from utils.cspa_main import prepare_data
from path_patching_cm.ioi_dataset import IOIDataset

from utils.cspa_main import get_result_mean, get_cspa_results_batched, get_performance_recovered
from utils.head_metrics import (
    convert_head_names_to_tuple,
    collate_fn,
    BatchIOIDataset,
    S2I_head_metrics, 
    S2I_token_pos
)

def get_cspa_for_head(model, data_toks, cspa_semantic_dict, layer, head, verbose=False):

    current_batch_size = 17 # Smaller values so we can check more checkpoints in a reasonable amount of time
    current_seq_len = 61

    result_mean = get_result_mean([(layer, head)], data_toks[:100, :], model, verbose=True)
    cspa_results_qk_ov = get_cspa_results_batched(
        model=model,
        toks=data_toks[:current_batch_size, :current_seq_len],
        max_batch_size=1,  # 50,
        negative_head=(layer, head),
        interventions=["ov", "qk"],
        only_keep_negative_components=True,
        K_unembeddings=0.05,  # most interesting in range 3-8 (out of 80)
        K_semantic=1,  # either 1 or up to 8 to capture all sem similar
        semantic_dict=cspa_semantic_dict,
        result_mean=result_mean,
        use_cuda=True,
        verbose=True,
        compute_s_sstar_dict=False,
        computation_device="cpu",  # device
    )
    head_results = get_performance_recovered(cspa_results_qk_ov)

    if verbose:
        print(f"Layer {layer}, head {head} done. Performance: {head_results:.2f}")

    return head_results


def get_attention_to_ioi_token(
        model: HookedTransformer, 
        ioi_dataset: IOIDataset,  
        head_list: List[Tuple[int, int]], 
        batch_size
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 

    ioi_dataset.__class__ = BatchIOIDataset
    ioi_dataloader = DataLoader(ioi_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, device=model.cfg.device))
    
    NMH_layers, NMH_heads = zip(*head_list)
    NMH_layers = torch.tensor(NMH_layers, device=model.cfg.device)
    NMH_heads = torch.tensor(NMH_heads, device=model.cfg.device)

    # Initialize tensors to accumulate attention values
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    s1_attention_accum = torch.zeros((n_layers, n_heads), device=model.cfg.device)
    s2_attention_accum = torch.zeros((n_layers, n_heads), device=model.cfg.device)
    io_attention_accum = torch.zeros((n_layers, n_heads), device=model.cfg.device)
    batch_count = 0

    for batch in ioi_dataloader:
        batch_count += 1
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

        attention_patterns = torch.stack([cache[f'blocks.{n}.attn.hook_pattern'] for n in range(n_layers)])  #layer, batch, head, query, key
        attention_patterns_by_head = attention_patterns[NMH_layers, :, NMH_heads]

        nmh_s1_attention_values = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, s1_pos]  # batch, layer, head
        nmh_s2_attention_values = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, s2_pos]  # batch, layer, head
        nmh_io_attention_values = attention_patterns_by_head[:, torch.arange(len(toks)), end_pos, io_pos]  # batch, layer, head

        # Accumulate attention values
        for i, (layer, head) in enumerate(head_list):
            s1_attention_accum[layer, head] += nmh_s1_attention_values[:, i].mean()
            s2_attention_accum[layer, head] += nmh_s2_attention_values[:, i].mean()
            io_attention_accum[layer, head] += nmh_io_attention_values[:, i].mean()

    # Calculate mean attention values
    s1_attention_means = s1_attention_accum / batch_count
    s2_attention_means = s2_attention_accum / batch_count
    io_attention_means = io_attention_accum / batch_count

    return s1_attention_means, s2_attention_means, io_attention_means


# get NMH candidates
def evaluate_direct_effect_heads(model, edge_df, dataset, verbose=False):
    direct_effect_heads = edge_df[edge_df['target']=='logits']
    direct_effect_heads = direct_effect_heads[direct_effect_heads['in_circuit'] == True]

    head_list = direct_effect_heads['source'].unique().tolist()
    head_list = [convert_head_names_to_tuple(c) for c in head_list if (c[0] != 'm' and c != 'input')]

    head_data = dict()


    # Test for NMH behavior
    head_data['copy_scores'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer, head in head_list:
        head_data['copy_scores'][layer, head] = compute_copy_score(model, layer, head, dataset, verbose=False, neg=False)

    # Test for attention to IOI tokens
    s1_attn_scores, s2_attn_scores, io_attn_scores = get_attention_to_ioi_token(model, dataset, head_list, batch_size=70)
    head_data['s1_attn_scores'], head_data['s2_attn_scores'], head_data['io_attn_scores'] = s1_attn_scores, s2_attn_scores, io_attn_scores
    
    # Test for copy suppression behavior
    model.cfg.use_split_qkv_input = False
    DATA_TOKS, DATA_STR_TOKS_PARSED, cspa_semantic_dict, indices = prepare_data(model)

    head_data['copy_suppression_scores'] = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer, head in head_list:
        if layer > 1:
            head_data['copy_suppression_scores'][layer, head] = get_cspa_for_head(model, DATA_TOKS, cspa_semantic_dict, layer, head, verbose=verbose)

    model.cfg.use_split_qkv_input = True
    
    return head_data


def filter_name_movers(direct_effect_scores, copy_score_threshold):
    direct_effect_scores['filtered_copy_scores'] = direct_effect_scores['copy_scores'].clone()

    nmh_list = []

    for layer in range(direct_effect_scores['copy_scores'].shape[0]):
        for head in range(direct_effect_scores['copy_scores'].shape[1]):
            if direct_effect_scores['copy_scores'][layer, head] < copy_score_threshold:
                direct_effect_scores['filtered_copy_scores'][layer, head] = 0

            if direct_effect_scores['copy_scores'][layer, head] > copy_score_threshold \
                and direct_effect_scores['io_attn_scores'][layer, head] > direct_effect_scores['s1_attn_scores'][layer, head] \
                     and direct_effect_scores['io_attn_scores'][layer, head] > direct_effect_scores['s2_attn_scores'][layer, head]:
                nmh_list.append((layer, head))

    return nmh_list


def evaluate_s2i_candidates(model, checkpoint_df, ioi_dataset, name_mover_heads, batch_size, verbose=False):

    patch_dataset_names = ['token_same_pos_oppo', 'token_oppo_pos_same', 'token_oppo_pos_oppo']
    targeting_nmh = np.logical_or.reduce(np.array([checkpoint_df['target'] == f'a{layer}.h{head}<q>' for layer, head in name_mover_heads]))
    candidate_s2i = checkpoint_df[targeting_nmh]
    candidate_s2i = candidate_s2i[candidate_s2i['in_circuit'] == True]

    candidate_list = candidate_s2i['source'].unique().tolist()
    candidate_list = [convert_head_names_to_tuple(c) for c in candidate_list if (c[0] != 'm' and c != 'input')]


    s2i_heads = candidate_list # [(7,9), (7,2), (6,6), (6,5),]


    s2i_ablated_logit_diff_deltas = {patch_dataset_name: torch.zeros((model.cfg.n_layers, model.cfg.n_heads)) for patch_dataset_name in patch_dataset_names}
    s2i_io_attention_deltas = {patch_dataset_name: torch.zeros((model.cfg.n_layers, model.cfg.n_heads)) for patch_dataset_name in patch_dataset_names}
    s2i_s1_attention_deltas = {patch_dataset_name: torch.zeros((model.cfg.n_layers, model.cfg.n_heads)) for patch_dataset_name in patch_dataset_names}
    s2i_s2_attention_deltas = {patch_dataset_name: torch.zeros((model.cfg.n_layers, model.cfg.n_heads)) for patch_dataset_name in patch_dataset_names}
    true_s2i_mask = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    true_s2i_heads = []

    for head in s2i_heads:
        s2i_token_pos_results = S2I_token_pos(model, ioi_dataset, [head], name_mover_heads, batch_size)
        if verbose:
            print(f'Head {head}')
        mean_original_logit_diff = s2i_token_pos_results['ablated_logit_diffs']['token_same_pos_same'].mean()
        mean_original_io_attention = s2i_token_pos_results['io_attention_values']['token_same_pos_same'].mean()
        mean_original_s1_attention = s2i_token_pos_results['s1_attention_values']['token_same_pos_same'].mean()
        mean_original_s2_attention = s2i_token_pos_results['s2_attention_values']['token_same_pos_same'].mean()

        for dataset_name in patch_dataset_names:

            mean_ablated_logit_diff = s2i_token_pos_results['ablated_logit_diffs'][dataset_name].mean()
            mean_ablated_io_attention = s2i_token_pos_results['io_attention_values'][dataset_name].mean()
            mean_ablated_s1_attention = s2i_token_pos_results['s1_attention_values'][dataset_name].mean()
            mean_ablated_s2_attention = s2i_token_pos_results['s2_attention_values'][dataset_name].mean()

            logit_diff_delta = (mean_ablated_logit_diff - mean_original_logit_diff) / mean_original_logit_diff
            io_attention_delta = (mean_ablated_io_attention - mean_original_io_attention) / mean_original_io_attention
            s1_attention_delta = (mean_ablated_s1_attention - mean_original_s1_attention) / mean_original_s1_attention
            s2_attention_delta = (mean_ablated_s2_attention - mean_original_s2_attention) / mean_original_s2_attention

            s2i_ablated_logit_diff_deltas[dataset_name][head] = logit_diff_delta
            s2i_io_attention_deltas[dataset_name][head] = io_attention_delta
            s2i_s1_attention_deltas[dataset_name][head] = s1_attention_delta
            s2i_s2_attention_deltas[dataset_name][head] = s2_attention_delta
            
            if verbose:
                print(dataset_name)
                print(f"Logit diff after patching: {100 * logit_diff_delta:.2f}%")
                # should be high with pos = same, low with pos = diff
                print(f"NMH IO Attention Change: {100 * io_attention_delta:.2f}%")
                # should be low with pos = same, high with pos = diff
                print(f"NMH S1 Attention Change: {100 * s1_attention_delta:.2f}%")
                # shouldn't change much
                print(f"NMH S2 Attention Change: {100 * s2_attention_delta:.2f}%")
                print('\n')
        
        layer, head_idx = head
        if s2i_ablated_logit_diff_deltas['token_same_pos_oppo'][layer, head_idx] < 0 \
            and s2i_io_attention_deltas['token_same_pos_oppo'][layer, head_idx] < 0 \
            and s2i_s1_attention_deltas['token_same_pos_oppo'][layer, head_idx] > 0:
            true_s2i_mask[layer, head_idx] = 1
            true_s2i_heads.append(head)

    # mask the deltas
    s2i_ablated_logit_diff_deltas = {k: v * true_s2i_mask for k, v in s2i_ablated_logit_diff_deltas.items()}
    s2i_io_attention_deltas = {k: v * true_s2i_mask for k, v in s2i_io_attention_deltas.items()}
    s2i_s1_attention_deltas = {k: v * true_s2i_mask for k, v in s2i_s1_attention_deltas.items()}
    s2i_s2_attention_deltas = {k: v * true_s2i_mask for k, v in s2i_s2_attention_deltas.items()}

    return {
        's2i_ablated_logit_diff_deltas': s2i_ablated_logit_diff_deltas, 
        's2i_io_attention_deltas': s2i_io_attention_deltas, 
        's2i_s1_attention_deltas': s2i_s1_attention_deltas, 
        's2i_s2_attention_deltas': s2i_s2_attention_deltas
    }, true_s2i_heads
        
        
def get_induction_scores(model):
    seq_len = 100
    batch_size = 2

    prev_token_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")

    def prev_token_hook(pattern, hook):
        layer = hook.layer()
        diagonal = pattern.diagonal(offset=1, dim1=-1, dim2=-2)
        prev_token_scores[layer] = einops.reduce(diagonal, "batch head_index diagonal -> head_index", "mean")

    duplicate_token_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")

    def duplicate_token_hook(pattern, hook):
        layer = hook.layer()
        diagonal = pattern.diagonal(offset=seq_len, dim1=-1, dim2=-2)
        duplicate_token_scores[layer] = einops.reduce(diagonal, "batch head_index diagonal -> head_index", "mean")

    induction_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device="cuda")

    def induction_hook(pattern, hook):
        layer = hook.layer()
        diagonal = pattern.diagonal(offset=seq_len-1, dim1=-1, dim2=-2)
        induction_scores[layer] = einops.reduce(diagonal, "batch head_index diagonal -> head_index", "mean")

    original_tokens = torch.randint(100, 20000, size=(batch_size, seq_len))
    repeated_tokens = einops.repeat(original_tokens, "batch seq_len -> batch (2 seq_len)").cuda()

    pattern_filter = lambda act_name: act_name.endswith("hook_pattern")
    loss = model.run_with_hooks(repeated_tokens, return_type="loss", fwd_hooks=[(pattern_filter, prev_token_hook), (pattern_filter, duplicate_token_hook), (pattern_filter, induction_hook)])

    return induction_scores, prev_token_scores, duplicate_token_scores


def evaluate_induction_scores(model, checkpoint_df):
    
    circuit_heads = checkpoint_df[checkpoint_df['in_circuit'] == True]['source'].unique().tolist()
    circuit_heads = [convert_head_names_to_tuple(c) for c in circuit_heads if (c[0] != 'm' and c != 'input')]
    
    circuit_mask = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device)
    for layer, head in circuit_heads:
        circuit_mask[layer, head] = 1
    
    induction_scores, prev_token_scores, duplicate_token_scores = get_induction_scores(model)
    induction_scores = induction_scores * circuit_mask
    prev_token_scores = prev_token_scores * circuit_mask
    duplicate_token_scores = duplicate_token_scores * circuit_mask

    return {
        'induction_scores': induction_scores, 
        'prev_token_scores': prev_token_scores, 
        'duplicate_token_scores': duplicate_token_scores
    }