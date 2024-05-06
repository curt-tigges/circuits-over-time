import os
import json
import glob
import torch
import re
import einops
import pandas as pd
from functools import partial
from torch import Tensor
from torchtyping import TensorType as TT
from jaxtyping import Float
from typing import List, Tuple, Dict, Any, Set, Optional, Callable
from data.ioi_dataset import IOIDataset

import numpy as np
import pandas as pd
import plotly.express as px

from transformers import AutoModelForCausalLM

import transformer_lens
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer, ActivationCache
import transformer_lens.patching as patching

import plotly.express as px

from utils.metrics import compute_logit_diff, _logits_to_mean_logit_diff
from utils.component_evaluation import compute_copy_score

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_model(
        base_model: str = "pythia-160m", 
        variant: str = None, 
        checkpoint: int = 143000, 
        cache: str = "model_cache", 
        device: torch.device = torch.device("cuda"),
        large_model: bool = False
    ) -> HookedTransformer:
    """
    Load a transformer model from a pretrained base model or variant.

    Args:
        BASE_MODEL (str): The name of the base model.
        VARIANT (str): The name of the model variant (if applicable).
        CHECKPOINT (int): The checkpoint value for the model.
        CACHE (str): The directory to cache the model.
        device (torch.device): The device to load the model onto.

    Returns:
        HookedTransformer: The loaded transformer model.
    """
    if not variant:

        if large_model:
            model_type = torch.bfloat16
        else:
            model_type = None

        model = HookedTransformer.from_pretrained(
            base_model,
            checkpoint_value=checkpoint,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
            #refactor_factored_attn_matrices=False,
            dtype=model_type,
            **{"cache_dir": cache},
        )
    elif not variant and large_model:
        if large_model:
            model_type = torch.bfloat16
        else:
            model_type = None
        revision = f"step{checkpoint}"
        source_model = AutoModelForCausalLM.from_pretrained(
           f"EleutherAI/{base_model}", revision=revision, cache_dir=cache
        ).to(model_type).to("cpu")
        print(f"Loaded model {variant} at {revision}; now loading into HookedTransformer")
        model = HookedTransformer.from_pretrained(
            base_model,
            hf_model=source_model,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
            dtype=model_type,
            **{"cache_dir": cache},
        )
    else:
        if large_model:
            model_type = torch.bfloat16
        else:
            model_type = None

        revision = f"step{checkpoint}"
        source_model = AutoModelForCausalLM.from_pretrained(
           variant, revision=revision, cache_dir=cache
        ).to("cpu") #.to(torch.bfloat16)
        print(f"Loaded model {variant} at {revision}; now loading into HookedTransformer")
        model = HookedTransformer.from_pretrained(
            base_model,
            hf_model=source_model,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            device=device,
            dtype=model_type,
            **{"cache_dir": cache},
        )

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model

def convert_head_names_to_tuple(head_name: str) -> Tuple[int, int]:
    """
    Convert a head name string to a tuple of (layer, head) indices.

    Args:
        head_name (str): The head name string in the format "a{layer}h{head}".

    Returns:
        Tuple[int, int]: A tuple containing the layer and head indices.
    """
    head_name = head_name.replace('a', '')
    head_name = head_name.replace('h', '')
    layer, head = head_name.split('.')
    return (int(layer), int(head))


def ablate_top_head_hook(z: TT["batch", "pos", "head_index", "d_head"], hook: Callable, head_idx: int = 0) -> TT["batch", "pos", "head_index", "d_head"]:
    """
    Hook function to ablate the top head in the attention mechanism.

    Args:
        z (TT["batch", "pos", "head_index", "d_head"]): The input tensor.
        hook (Callable): The hook function.
        head_idx (int, optional): The index of the head to ablate. Defaults to 0.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: The modified tensor with the top head ablated.
    """
    z[:, :, head_idx, :] = 0
    return z





def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    """
    Calculate the average logit difference between the correct and incorrect answer
    for a given stack of components in the residual stream.

    Args:
        residual_stack (Float[Tensor, "... batch d_model"]): The residual stack tensor.
        cache (ActivationCache): The activation cache.
        logit_diff_directions (Float[Tensor, "batch d_model"]): The logit difference directions.

    Returns:
        Float[Tensor, "..."]: The average logit difference.
    """
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size


# Prepare model
def setup(base_model: str, variant: str, dataset: IOIDataset, checkpoint: int = 143000) -> Tuple[HookedTransformer, Float[Tensor, "batch d_model"]]:
    """
    Set up the model and prepare the logit difference directions.

    Args:
        base_model (str): The name of the base model.
        variant (str): The name of the model variant (if applicable).
        dataset (IOIDataset): The IOI dataset.
        checkpoint (int, optional): The checkpoint value for the model. Defaults to 143000.

    Returns:
        Tuple[HookedTransformer, Float[Tensor, "batch d_model"]]: A tuple containing the loaded model and the logit difference directions.
    """
    model = load_model(base_model, variant, checkpoint, cache="model_cache", device=device)

    answer_tokens = torch.cat((torch.Tensor(dataset.io_tokenIDs).unsqueeze(1), torch.Tensor(dataset.s_tokenIDs).unsqueeze(1)), dim=1).to(device)
    answer_tokens = answer_tokens.long()
    answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
    logit_diff_directions = answer_residual_directions[:, 0] - answer_residual_directions[:, 1]

    # Test logit_diff_directions with logit diff calculation
    # final_residual_stream: Float[Tensor, "batch seq d_model"] = orig_cache["resid_post", -1]

    # scaled_residual_stream = orig_cache.apply_ln_to_stack(final_residual_stream, layer=-1)
    # scaled_final_token_residual_stream: Float[Tensor, "batch d_model"] = scaled_residual_stream[torch.arange(scaled_residual_stream.size(0)), ioi_dataset.word_idx["end"]]
    
    # batch_size = ioi_dataset.toks.shape[0]

    # average_logit_diff = einops.einsum(
    #     scaled_final_token_residual_stream, logit_diff_directions,
    #     "batch d_model, batch d_model ->"
    # ) / batch_size

    # print(f"Calculated logit diff: {average_logit_diff:.10f}")

    return model, logit_diff_directions

# Get metrics & attribution scores
def get_metrics_and_attributions(model: HookedTransformer, logits: Float[Tensor, "batch seq d_vocab"], cache: ActivationCache, dataset: IOIDataset, logit_diff_directions: Float[Tensor, "batch d_model"]) -> Tuple[float, Float[Tensor, "layer head"]]:
    """
    Calculate the logit difference and per-head logit differences.

    Args:
        model (HookedTransformer): The transformer model.
        logits (Float[Tensor, "batch seq d_vocab"]): The output logits from the model.
        cache (ActivationCache): The activation cache.
        dataset (IOIDataset): The IOI dataset.
        logit_diff_directions (Float[Tensor, "batch d_model"]): The logit difference directions.

    Returns:
        Tuple[float, Float[Tensor, "layer head"]]: A tuple containing the logit difference and per-head logit differences.
    """

    logit_diff = _logits_to_mean_logit_diff(logits, dataset).item()

    per_head_residual, labels = cache.stack_head_results(layer=-1, return_labels=True)
    per_head_residual_final_token = per_head_residual[:, torch.arange(per_head_residual.size(1)), dataset.word_idx["end"]]
    per_head_residual_final_token = einops.rearrange(
        per_head_residual_final_token,
        "(layer head) ... -> layer head ...",
        layer=model.cfg.n_layers
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual_final_token, cache, logit_diff_directions)

    return logit_diff, per_head_logit_diffs

# Get copy scores from circuit members
def get_ablation_targets(model: HookedTransformer, checkpoint: int, edge_df: pd.DataFrame, dataset: IOIDataset, threshold: float = 75.0) -> Tuple[List[Tuple[int, int, float]], List[Tuple[int, int]]]:
    """
    Get the ablation targets based on copy scores and a threshold.

    Args:
        model (HookedTransformer): The transformer model.
        checkpoint (int): The checkpoint value.
        edge_df (pd.DataFrame): The DataFrame containing edge information.
        dataset (IOIDataset): The IOI dataset.
        threshold (float, optional): The threshold for selecting ablation targets. Defaults to 75.0.

    Returns:
        Tuple[List[Tuple[int, int, float]], List[Tuple[int, int]]]: A tuple containing the NMHs (no meaning heads) and the heads to ablate.
    """
    candidate_nmh = edge_df[edge_df['target']=='logits']
    candidate_nmh = candidate_nmh[candidate_nmh['in_circuit'] == True]

    candidate_list = candidate_nmh[candidate_nmh['checkpoint']==checkpoint]['source'].unique().tolist()
    candidate_list = [convert_head_names_to_tuple(c) for c in candidate_list if (c[0] != 'm' and c != 'input')]

    NMHs = []

    for layer, head in candidate_list:
        copy_score = compute_copy_score(model, layer, head, dataset, verbose=False, neg=False)
        NMHs.append((layer, head, copy_score))

    heads_to_ablate = [x[:2] for x in NMHs if x[2] >= threshold]

    return NMHs, heads_to_ablate

# Run ablation experiment
def run_ablated_model(model: HookedTransformer, dataset: IOIDataset, ablation_targets: Optional[List[Tuple[int, int]]] = None) -> Tuple[Float[Tensor, "batch seq d_vocab"], ActivationCache]:
    """
    Run the model with ablated heads and return the logits and activation cache.

    Args:
        model (HookedTransformer): The transformer model.
        dataset (IOIDataset): The IOI dataset.
        ablation_targets (Optional[List[Tuple[int, int]]], optional): The ablation targets. Defaults to None.

    Returns:
        Tuple[Float[Tensor, "batch seq d_vocab"], ActivationCache]: A tuple containing the logits and activation cache from the ablated model.
    """
    if ablation_targets is None:
        ablation_targets = get_ablation_targets(model, dataset)

    for layer, head in ablation_targets:
        ablate_head_hook = partial(ablate_top_head_hook, head_idx=head)
        model.blocks[layer].attn.hook_z.add_hook(ablate_head_hook)

    ablated_logits, ablated_cache = model.run_with_cache(dataset.toks)
    
    model.reset_hooks()

    return ablated_logits, ablated_cache

# Run experiment
def run_iteration(base_model: str, variant: str, edge_df: pd.DataFrame, checkpoint: int, dataset: IOIDataset, experiment_metrics: Dict[int, Dict[str, Any]], threshold: float) -> Dict[int, Dict[str, Any]]:
    """
    Run a single iteration of the experiment.

    Args:
        base_model (str): The name of the base model.
        variant (str): The name of the model variant (if applicable).
        edge_df (pd.DataFrame): The DataFrame containing edge information.
        checkpoint (int): The checkpoint value.
        dataset (IOIDataset): The IOI dataset.
        experiment_metrics (Dict[int, Dict[str, Any]]): The dictionary to store experiment metrics.
        threshold (float): The threshold for selecting ablation targets.

    Returns:
        Dict[int, Dict[str, Any]]: The updated experiment metrics dictionary.
    """

    model, logit_diff_directions = setup(base_model, variant, dataset=dataset, checkpoint=checkpoint)
    orig_logits, orig_cache = model.run_with_cache(dataset.toks.long())
    logit_diff, per_head_logit_diffs = get_metrics_and_attributions(model, orig_logits, orig_cache, dataset, logit_diff_directions=logit_diff_directions)

    NMHs, ablation_targets = get_ablation_targets(model, checkpoint=checkpoint, edge_df=edge_df, dataset=dataset, threshold=threshold)
    ablated_logits, ablated_cache = run_ablated_model(model, dataset, ablation_targets)
    ablated_logit_diff, per_head_ablated_logit_diffs = get_metrics_and_attributions(model, ablated_logits, ablated_cache, dataset, logit_diff_directions=logit_diff_directions)
    
    print(f"Checkpoint {checkpoint}:")
    print(f"Heads ablated:            {ablation_targets}")
    print(f"Original logit diff:      {logit_diff:.10f}")
    print(f"Post ablation logit diff: {ablated_logit_diff:.10f}")
    print(f"Logit diff % change:      {((ablated_logit_diff - logit_diff) / logit_diff) * 100:.2f}%")

    experiment_metrics[checkpoint] = {
        "logit_diff": logit_diff,
        "per_head_logit_diffs": per_head_logit_diffs,
        "ablation_targets": ablation_targets,
        "ablated_logit_diff": ablated_logit_diff,
        "per_head_ablated_logit_diffs": per_head_ablated_logit_diffs,
        "per_head_logit_diff_delta": per_head_ablated_logit_diffs - per_head_logit_diffs
    }

    return experiment_metrics

def get_past_nmhs_for_checkpoints(
        experiment_metrics: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[int, Set[Tuple[int, int]]], Dict[int, Set[Tuple[int, int]]]]:
    """
    Get the cumulative and individual NMHs (no meaning heads) for each checkpoint.

    Args:
        experiment_metrics (Dict[int, Dict[str, Any]]): The experiment metrics dictionary.

    Returns:
        Tuple[Dict[int, Set[Tuple[int, int]]], Dict[int, Set[Tuple[int, int]]]]: A tuple containing the cumulative and individual NMHs for each checkpoint.
    """
    checkpoint_nmhs =  {checkpoint: set(experiment_metrics[checkpoint]['ablation_targets']) for checkpoint in experiment_metrics.keys()}
    #print(checkpoint_nmhs[20000])

    checkpoint_list = list(checkpoint_nmhs.keys())
    checkpoint_list.sort()
    
    cumulative_nmhs = dict()
    previous_set = set()
    previous_ckpt = 0

    for checkpoint in checkpoint_list:
        cumulative_nmhs[checkpoint] = set()
        if previous_ckpt == 0:
            cumulative_nmhs[checkpoint] = checkpoint_nmhs[checkpoint]
        else:
            cumulative_nmhs[checkpoint] = checkpoint_nmhs[checkpoint].union(previous_set)

        previous_set = cumulative_nmhs[checkpoint]
        previous_ckpt = checkpoint

    return cumulative_nmhs, checkpoint_nmhs




def process_backup_results(edge_df: pd.DataFrame, checkpoint: int, experiment_metrics: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Process the backup results and update the experiment metrics.

    Args:
        edge_df (pd.DataFrame): The DataFrame containing edge information.
        checkpoint (int): The checkpoint value.
        experiment_metrics (Dict[int, Dict[str, Any]]): The experiment metrics dictionary.

    Returns:
        Dict[int, Dict[str, Any]]: The updated experiment metrics dictionary.
    """

    # exclude the delta of the ablated heads
    for layer, head in experiment_metrics[checkpoint]["ablation_targets"]:
        experiment_metrics[checkpoint]["per_head_logit_diff_delta"][layer, head] = 0

    # get the list of heads in the circuit
    circuit_heads = edge_df[edge_df['in_circuit'] == True]
    circuit_heads = circuit_heads[circuit_heads['checkpoint']==checkpoint]['source'].unique().tolist()
    circuit_heads = [convert_head_names_to_tuple(c) for c in circuit_heads if (c[0] != 'm' and c != 'input')]    

    in_circuit_head_delta = torch.zeros_like(experiment_metrics[checkpoint]["per_head_logit_diffs"])
    outside_circuit_head_delta = torch.zeros_like(experiment_metrics[checkpoint]["per_head_logit_diffs"])

    for layer in range(in_circuit_head_delta.shape[0]):
        for head in range(in_circuit_head_delta.shape[1]):
            if (layer, head) in circuit_heads:
                in_circuit_head_delta[layer, head] = experiment_metrics[checkpoint]["per_head_logit_diff_delta"][layer, head]
            else:
                outside_circuit_head_delta[layer, head] = experiment_metrics[checkpoint]["per_head_logit_diff_delta"][layer, head]

    experiment_metrics[checkpoint]["in_circuit_head_delta"] = in_circuit_head_delta
    experiment_metrics[checkpoint]["outside_circuit_head_delta"] = outside_circuit_head_delta

    experiment_metrics[checkpoint]["summed_in_circuit_head_delta"] = in_circuit_head_delta.sum().item()
    experiment_metrics[checkpoint]["summed_outside_circuit_head_delta"] = outside_circuit_head_delta.sum().item()
    experiment_metrics[checkpoint]["summed_total_head_delta"] = experiment_metrics[checkpoint]["per_head_logit_diff_delta"].sum().item()

    # convert tensors to numpy cpu arrays
    experiment_metrics[checkpoint]["per_head_logit_diffs"] = experiment_metrics[checkpoint]["per_head_logit_diffs"].cpu().numpy()
    experiment_metrics[checkpoint]["per_head_ablated_logit_diffs"] = experiment_metrics[checkpoint]["per_head_ablated_logit_diffs"].cpu().numpy()
    experiment_metrics[checkpoint]["per_head_logit_diff_delta"] = experiment_metrics[checkpoint]["per_head_logit_diff_delta"].cpu().numpy()
    experiment_metrics[checkpoint]["in_circuit_head_delta"] = experiment_metrics[checkpoint]["in_circuit_head_delta"].cpu().numpy()
    experiment_metrics[checkpoint]["outside_circuit_head_delta"] = experiment_metrics[checkpoint]["outside_circuit_head_delta"].cpu().numpy()
    

    return experiment_metrics



def plot_top_heads(
        model_name: str,
        checkpoint_dict: Dict[int, np.ndarray], 
        cumulative_nmhs: Dict[int, Set[Tuple[int, int]]], 
        top_k_per_checkpoint: int = 5, 
        top_k: int = 5
    ) -> pd.DataFrame:
    """
    Plot the top backup heads attribution across checkpoints.

    Args:
        checkpoint_dict (Dict[int, np.ndarray]): A dictionary mapping checkpoints to numpy arrays of head attributions.
        cumulative_nmhs (Dict[int, Set[Tuple[int, int]]]): A dictionary mapping checkpoints to sets of cumulative NMHs.
        top_k_per_checkpoint (int, optional): The number of top heads to consider per checkpoint. Defaults to 5.
        top_k (int, optional): The number of overall top heads to plot. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing the plot data.
    """
    # Step 1: Identify the top heads for each checkpoint
    top_heads = {}
    for checkpoint, array in checkpoint_dict.items():
        # Use argpartition to get the indices of the top 5 heads in the entire array
        flat_indices = np.argpartition(array.flatten(), -top_k_per_checkpoint)[-top_k_per_checkpoint:]
        # Convert flat indices to 2D indices
        indices = np.unravel_index(flat_indices, array.shape)
        # Store the top heads for this checkpoint
        top_heads[checkpoint] = [(layer, head) for layer, head in zip(indices[0], indices[1])]

    # Step 2: Prepare the data for plotting
    plot_data = []
    
    for checkpoint, heads in top_heads.items():
        array = checkpoint_dict[checkpoint]
        for layer, head in heads:
            in_previous_nmh_list = (layer, head) in cumulative_nmhs[checkpoint]
            plot_data.append(
                {
                    'Checkpoint': checkpoint, 
                    'Layer-Head': f'Layer {layer}-Head {head}', 
                    'Layer': layer, 
                    'Head': head, 
                    'Value': array[layer, head],
                    'Previous NMH': in_previous_nmh_list
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    df = df[df['Layer-Head']!='Layer 3-Head 11']
    df = df[df['Layer-Head']!='Layer 3-Head 10']

    # Calculate sum of values over all checkpoints for each head
    summary_df = df.groupby(['Layer-Head', 'Layer', 'Head']).sum().reset_index()

    # label the top 5 items in summary_df
    summary_df['Top K'] = summary_df['Layer-Head'].isin(df.groupby('Layer-Head').mean().nlargest(top_k, 'Value').index)

    # Join to main df
    df = df.merge(summary_df, on=['Layer-Head', 'Layer', 'Head'], suffixes=('', '_sum'))

    plot_df = df[df['Top K'] == True]

    # Step 3: Plot the data
    fig = px.line(
        plot_df, 
        x='Checkpoint', 
        y='Value', 
        color='Layer-Head', 
        title=f'Top Backup Heads Attribution Across Checkpoints ({model_name})', 
        height=500,
        labels={'x': 'Checkpoint', 'y': 'Change as % of original logit diff'}
    )
    fig.show()

    return df