# %%
import os
import sys
import torch
from torch import Tensor
import numpy as np
import einops
from fancy_einsum import einsum
import circuitsvis as cv
from IPython.display import display, clear_output, HTML

import transformer_lens.utils as utils

from transformer_lens import HookedTransformer
import transformer_lens.patching as patching

from torch import Tensor
from jaxtyping import Float
import plotly.express as px

from functools import partial

from torchtyping import TensorType as TT

from utils.path_patching import Node, IterNode, path_patch, act_patch
from utils.head_metrics import S2I_head_metrics, BatchIOIDataset

from utils.visualization import imshow_p, plot_attention_heads

from utils.data_processing import load_edge_scores_into_dictionary

from utils.visualization import (
    plot_attention_heads,
    scatter_attention_and_contribution,
    get_attn_head_patterns
)
from utils.circuit_analysis import get_pct_effect, extract_source_nodes, check_source_nodes
from utils.backup_analysis import (
    load_model, 
    compute_copy_score,
    setup,
    get_metrics_and_attributions,
    run_ablated_model,
    convert_head_names_to_tuple
)
from utils.cspa_main import (
    get_cspa_for_model,
    load_model_for_cspa
)
from utils.data_processing import get_ckpts
from utils.metrics import logit_diff_denoising, logit_diff_noising
from utils.data_utils import generate_data_and_caches, _logits_to_mean_logit_diff
from utils.component_evaluation import evaluate_s2i_candidates
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

torch.set_grad_enabled(False)

# %%
TASK = 'ioi'
PERFORMANCE_METRIC = 'logit_diff'
BASE_MODEL = "pythia-1.4b"
VARIANT = None#"EleutherAI/pythia-160m-attndropout"
MODEL_SHORTNAME = BASE_MODEL if not VARIANT else VARIANT[11:]
CACHE = "/mnt/hdd-0/circuits-over-time/model_cache/ct"
DATASET_SIZE = 30

CSPA_THRESHOLD = 0.10
COPY_SCORE_THRESHOLD = 10

checkpoints = get_ckpts("sparse")[12:]
algorithm_scores = dict()
algorithm_scores['direct_effects'] = dict()
algorithm_scores['s2i_effects'] = dict()
algorithm_scores['tertiary_effects'] = dict()

RESULT_STORE = f'/mnt/hdd-0/circuits-over-time/results/algorithmic_consistency/ioi'
os.makedirs(RESULT_STORE, exist_ok=True)

# %%
print(checkpoints)

# %%
for ckpt in checkpoints:

    print(f"Processing checkpoint {ckpt}")
    # load saved results if file exists
    if os.path.exists(f'{RESULT_STORE}/{MODEL_SHORTNAME}.pt'):
        algorithm_scores = torch.load(f'{RESULT_STORE}/{MODEL_SHORTNAME}.pt')
        if ckpt in algorithm_scores['direct_effects'] and ckpt in algorithm_scores['s2i_effects'] and ckpt in algorithm_scores['tertiary_effects']:
            print(f"Checkpoint {ckpt} already processed")
            continue

    model = load_model(BASE_MODEL, VARIANT, ckpt, CACHE, device, large_model=False)
    model.tokenizer.add_bos_token = False
    model.set_use_attn_in(True)

    N_LAYERS = model.cfg.n_layers
    N_HEADS = model.cfg.n_heads

    # load graph from EAP-IG
    folder_path = f'/mnt/hdd-0/circuits-over-time/results/graphs/{MODEL_SHORTNAME}/{TASK}/raw'
    edge_df = load_edge_scores_into_dictionary(folder_path, ckpt)
    edge_df[['source', 'target']] = edge_df['edge'].str.split('->', expand=True)
    
    nodes = extract_source_nodes(edge_df)

    # data setup
    ioi_dataset, abc_dataset = generate_data_and_caches(model, N=DATASET_SIZE, verbose=True)
    clean_logits, clean_cache = model.run_with_cache(ioi_dataset.toks)
    corrupted_logits, corrupted_cache = model.run_with_cache(abc_dataset.toks)

    clean_logit_diff = _logits_to_mean_logit_diff(clean_logits, ioi_dataset)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = _logits_to_mean_logit_diff(corrupted_logits, ioi_dataset)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

    corrupted_cache.to("cpu")
    clean_cache.to("cpu")
    CLEAN_BASELINE = clean_logit_diff
    CORRUPTED_BASELINE = corrupted_logit_diff

    logit_diff_denoising_ioi = partial(
        logit_diff_denoising, 
        flipped_logit_diff=corrupted_logit_diff,
        clean_logit_diff=clean_logit_diff, 
        dataset=ioi_dataset
    )
    logit_diff_noising_ioi = partial(
        logit_diff_noising,
        clean_logit_diff=clean_logit_diff,
        flipped_logit_diff=corrupted_logit_diff,
        dataset=ioi_dataset)


    # path patching for direct heads
    path_patch_resid_post = path_patch(
        model,
        orig_input=ioi_dataset.toks,
        new_input=abc_dataset.toks,
        sender_nodes=IterNode('z'),
        receiver_nodes=Node('resid_post', int(N_LAYERS - 1)),
        patching_metric=logit_diff_noising_ioi,
        verbose=True
    )

    # get direct-effects from stored graph for comparison
    direct_effect_head_df = edge_df[edge_df['target']=='logits']
    direct_effect_head_df = direct_effect_head_df[direct_effect_head_df['in_circuit'] == True]

    # create new column with absolute value of score
    direct_effect_head_df['abs_score'] = direct_effect_head_df['score'].abs()

    # sort by direct effect
    direct_effect_head_df = direct_effect_head_df.sort_values(by='edge', ascending=False)


    # positive name movers
    copy_scores = torch.zeros((N_LAYERS, N_HEADS))
    copy_score_masks = torch.zeros((N_LAYERS, N_HEADS))

    copy_scores = compute_copy_score(model, [(layer, head) for layer in range(N_LAYERS) for head in range (N_HEADS)], ioi_dataset)
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            if copy_scores[layer, head] > COPY_SCORE_THRESHOLD:
                copy_score_masks[layer, head] = 1

    nmh_scores = path_patch_resid_post['z'] * 100 * copy_score_masks
    DE_NMH = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if nmh_scores[l, h] < 0]

    # negative name movers
    neg_copy_scores = torch.zeros((N_LAYERS, N_HEADS))
    neg_copy_score_masks = torch.zeros((N_LAYERS, N_HEADS))
    neg_copy_scores = compute_copy_score(model, [(layer, head) for layer in range(N_LAYERS) for head in range (N_HEADS)], ioi_dataset, neg=True)
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            if neg_copy_scores[layer, head] > COPY_SCORE_THRESHOLD:
                neg_copy_score_masks[layer, head] = 1


    neg_nmh_scores = path_patch_resid_post['z'] * 100 * neg_copy_score_masks
    DE_NEG_NMH = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if neg_nmh_scores[l, h] > 0 or neg_nmh_scores[l, h] < 0]

    # copy suppression heads
    head_targets = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if path_patch_resid_post['z'][l, h] < -0.01]

    file_path = f'results/cspa/{MODEL_SHORTNAME}/all_checkpoints.pt'

    data_found = False
    if os.path.exists(file_path):
        cspa_dict = torch.load(f'/mnt/hdd-0/circuits-over-time/results/components/{MODEL_SHORTNAME}/whole_model_cspa.pt')
        if ckpt in cspa_dict:
            cspa_results = cspa_dict[ckpt]
            data_found = True
            print("data found")

    if not data_found:
        cspa_device = 6
        cspa_model = load_model_for_cspa(BASE_MODEL, VARIANT, ckpt, CACHE, f"cuda:{cspa_device}")

        # note that if you want to save these results, you must do so separately
        cspa_results = get_cspa_for_model(cspa_model, start_layer=2, cuda_device=cspa_device, head_targets=head_targets)

    

    cspa_scores = torch.zeros((N_LAYERS, N_HEADS))
    cspa_score_masks = torch.zeros((N_LAYERS, N_HEADS))
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            cspa_scores[layer, head] = cspa_results[layer, head]
            if cspa_scores[layer, head] > CSPA_THRESHOLD:
                cspa_score_masks[layer, head] = 1
    
    copy_suppression_scores = path_patch_resid_post['z'] * 100 * cspa_score_masks
    DE_CSH = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if copy_suppression_scores[l, h] < 0]
    DE_NEG_CSH = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if copy_suppression_scores[l, h] > 0]

    print(f"NMH heads: {DE_NMH}")
    print(f"Negative NMH heads: {DE_NEG_NMH}")
    print(f"Copy suppression heads: {DE_CSH}")
    print(f"Negative copy suppression heads: {DE_NEG_CSH}")

    identified_DE_heads = list(set(DE_NMH + DE_CSH + DE_NEG_CSH + DE_NEG_NMH))
    direct_effect_result = get_pct_effect(identified_DE_heads, path_patch_resid_post['z'], nodes)[0]
    print(f"Percent of total: {direct_effect_result:.2f}%")

    algorithm_scores['direct_effects'][ckpt] = direct_effect_result


    # s2i heads
    results = path_patch(
        model,
        orig_input=ioi_dataset.toks,
        new_input=abc_dataset.toks,
        sender_nodes=IterNode("z"),
        receiver_nodes=[Node("q", layer, head=head) for layer, head in DE_NMH],
        patching_metric=logit_diff_noising_ioi,
        verbose=True,
    )
    S2I_CANDIDATES = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads) if results['z'][l, h] < -0.001]
    s2i_evals = evaluate_s2i_candidates(model, edge_df, ioi_dataset, DE_NMH, batch_size = DATASET_SIZE)
    S2I = []
    for head in S2I_CANDIDATES:
        if (
            s2i_evals[0]['s2i_ablated_logit_diff_deltas']['token_same_pos_oppo'][head[0], head[1]].item() < -0.0
            and s2i_evals[0]['s2i_io_attention_deltas']['token_same_pos_oppo'][head[0], head[1]].item() < 0.0
            and s2i_evals[0]['s2i_s1_attention_deltas']['token_same_pos_oppo'][head[0], head[1]].item() > 0.0
        ):
            S2I.append(head)

        # s2i_results = S2I_head_metrics(model, s2i_ioi_dataset, potential_s2i_list=[head], NMH_list=DE_NMH, batch_size=32)

        # s2i_s2_attention = s2i_results['end_s2_attention_values'].mean(0)

        # # logit diff change (lower is better)
        # logit_diff_change = (s2i_results['new_logit_diffs'] - s2i_results['baseline_logit_diffs'].unsqueeze(1)).mean(0)

        # # NMH s1 attention change (higher is better)
        # nmh_s1_attention_change = (s2i_results['new_nmh_s1_attention_values'] - s2i_results['baseline_nmh_s1_attention_values'].unsqueeze(1)).mean(0).mean(-1)

        print(f"S2I Candidate: {head}:")
        print(f"S2I Positional Signal Ablation Logit Diff Delta:        {s2i_evals[0]['s2i_ablated_logit_diff_deltas']['token_same_pos_oppo'][head[0], head[1]].item():.3f}")
        print(f"S2I Token Signal Ablation Logit Diff Delta:             {s2i_evals[0]['s2i_ablated_logit_diff_deltas']['token_oppo_pos_same'][head[0], head[1]].item():.3f}")
        print(f"NMH IO Attention Delta(after S2I pos signal ablation):  {s2i_evals[0]['s2i_io_attention_deltas']['token_same_pos_oppo'][head[0], head[1]].item():.3f}")
        print(f"NMH S1 Attention Delta (after S2I pos signal ablation): {s2i_evals[0]['s2i_s1_attention_deltas']['token_same_pos_oppo'][head[0], head[1]].item():.3f}")
        print(f"NMH S2 Attention Delta (after S2I pos signal ablation): {s2i_evals[0]['s2i_s2_attention_deltas']['token_same_pos_oppo'][head[0], head[1]].item():.3f}")
        print("\n")

    print(f"S2I heads: {S2I}")
    pure_S2I = [h for h in S2I if h not in DE_CSH]
    pct_effect, total_effect = get_pct_effect(S2I, results['z'], nodes)
    print(f"Percent of total: {pct_effect:.2f}%")
    print(f"Actual total: {total_effect:.2f}")

    algorithm_scores['s2i_effects'][ckpt] = pct_effect

    # tertiary effect heads
    results = path_patch(
        model,
        orig_input=ioi_dataset.toks,
        new_input=abc_dataset.toks,
        sender_nodes=IterNode("z"),
        receiver_nodes=[Node("v", layer, head=head) for layer, head in pure_S2I],
        patching_metric=logit_diff_noising_ioi,
        verbose=True,
    )
    scores_by_checkpoint = torch.load(f'results/components/{MODEL_SHORTNAME}/full_model_components_over_time.pt')
    scores_by_type = dict()
    for type in scores_by_checkpoint[ckpt]['tertiary_head_scores'].keys():
        
        scores_by_type[type] = dict()
        scores_by_type[type] = {checkpoint: v['tertiary_head_scores'][type] for checkpoint, v in scores_by_checkpoint.items()}

    IDHs = [
        (l, h) for l in range(model.cfg.n_layers) 
        for h in range(model.cfg.n_heads) 
        if results['z'][l, h] < -0.001
        and scores_by_type['induction_scores'][ckpt][l, h] > scores_by_type['induction_scores'][ckpt].mean()
    ]
    DTHs = [
        (l, h) for l in range(model.cfg.n_layers) 
        for h in range(model.cfg.n_heads) 
        if results['z'][l, h] < -0.001
        and scores_by_type['duplicate_token_scores'][ckpt][l, h] > scores_by_type['duplicate_token_scores'][ckpt].mean()
    ]
    print(f"Induction heads: {IDHs}")
    print(f"Duplicate token heads: {DTHs}")

    pct_effect, total_effect = get_pct_effect(list(set(DTHs+IDHs)), results['z'], nodes)
    print(f"Percent of total: {pct_effect:.2f}%")
    print(f"Actual total: {total_effect:.2f}")

    algorithm_scores['tertiary_effects'][ckpt] = pct_effect

    torch.save(algorithm_scores, f'{RESULT_STORE}/{MODEL_SHORTNAME}.pt')
# %%

print("Direct effects:")
print([f"{algorithm_scores['direct_effects'][ckpt].item():.2f}" for ckpt in checkpoints])

print("S2I effects:")
print([f"{algorithm_scores['s2i_effects'][ckpt].item():.2f}" for ckpt in checkpoints])

print("Tertiary effects:")
print([f"{algorithm_scores['tertiary_effects'][ckpt].item():.2f}" for ckpt in checkpoints])

# %%
