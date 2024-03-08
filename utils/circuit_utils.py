import os
from functools import partial
from collections import namedtuple

import torch
from torch import Tensor
from typing import List, Optional, Union, Dict, Tuple
from path_patching_cm.ioi_dataset import IOIDataset
from torchtyping import TensorType as TT
#from utils.path_patching import path_patch, get_path_patching_results

from path_patching_cm import path_patching
import transformer_lens.patching as patching
from transformer_lens import HookedTransformer
import plotly.graph_objs as go
import torch
import ipywidgets as widgets
from IPython.display import display

from utils.model_utils import load_model, clear_gpu_memory
from utils.data_utils import UniversalPatchingDataset
from utils.metrics import (
    CircuitMetric, 
    compute_logit_diff, 
    compute_probability_diff, 
    compute_probability_mass, 
    compute_rank_0_rate, 
    compute_accuracy,
    compute_mean_reciprocal_rank,
    compute_max_group_rank_reciprocal
)
from utils.metrics import _logits_to_mean_logit_diff, _logits_to_mean_accuracy, _logits_to_rank_0_rate, CircuitMetric, get_logit_diff, ioi_metric

from ACDCPP.acdcpp import get_acdcpp_results


if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"

# =============== CIRCUIT ===============
CircuitComponent = namedtuple(
    "CircuitComponent", ["heads", "position", "receiver_type"]
)

# =============== INFERENCE BATCHING UTILS ===============
def make_shapes_uniform(
        batch_tokens: Tensor, 
        max_seq_len: int
):
    """
    Makes the shape of the batch token tensor conform to max length by padding with zeros.
    """
    batch_size, seq_len = batch_tokens.shape
    if seq_len < max_seq_len:
        #print(f"Padding batch of shape {batch_tokens.shape} to {max_seq_len}...")
        batch_tokens = torch.cat([batch_tokens, torch.zeros((batch_size, max_seq_len - seq_len), dtype=torch.long).to(device)], dim=1)

    return batch_tokens


def process_in_batches(
        model: HookedTransformer, 
        token_tensor: Tensor, 
        batch_size: int,
        max_seq_len: int
)-> List[Tensor]:
    """ Processes a tensor of tokens in batches.

    Args:
        model (HookedTransformer): Model to run.
        token_tensor (Tensor): Tensor of tokens to run inference on.
        batch_size (int): Batch size to use for inference.

    Returns:
        List[Tensor]: List of logits for each batch.
    """
    dataset_len = token_tensor.shape[0]
    num_batches = dataset_len // batch_size + (1 if dataset_len % batch_size > 0 else 0)
    results = []
    for i in range(num_batches):
        batch = token_tensor[i * batch_size:(i + 1) * batch_size]
        resized_batch = make_shapes_uniform(batch, max_seq_len=max_seq_len)
        batch_logits = model(resized_batch)
        results.append(batch_logits)
    return results


def run_with_batches(
        model: HookedTransformer, 
        token_tensor: Tensor, 
        batch_size: int,
        max_seq_len: int
)-> Tensor:
    """ Performs inference with a HookedTransformer model in batches. Resulting logits are concatenated.

    Args:
        model (HookedTransformer): Model to run.
        token_tensor (Tensor): Tensor of tokens to run inference on.
        batch_size (int): Batch size to use for inference.

    Returns:
        Tensor: Concatenated logits.
    """
    logits = process_in_batches(model, token_tensor, batch_size, max_seq_len)
    logit_tensor = torch.cat(logits, dim=0)
    return logit_tensor


# =============== VISUALIZATION UTILS ===============
def visualize_tensor(tensor, labels, zmin=-1.0, zmax=1.0):
    """Visualizes a 3D tensor as a series of heatmaps.

    Args:
        tensor (torch.Tensor): Tensor to visualize.
        labels (List[str]): List of labels for each slice in the tensor.
        zmin (float, optional): Minimum value for the color scale. Defaults to -1.0.
        zmax (float, optional): Maximum value for the color scale. Defaults to 1.0.

    Raises:
        AssertionError: If the number of labels does not match the number of slices in the tensor.
    """
    assert (
        len(labels) == tensor.shape[-1]
    ), "The number of labels should match the number of slices in the tensor."

    def plot_slice(selected_slice):
        """Plots a single slice of the tensor."""
        fig = go.FigureWidget(
            data=go.Heatmap(
                z=tensor[:, :, selected_slice].numpy(),
                zmin=zmin,
                zmax=zmax,
                colorscale="RdBu",
            ),
            layout=go.Layout(
                title=f"Slice: {selected_slice} - Step: {labels[selected_slice]}",
                yaxis=dict(autorange="reversed"),
            ),
        )
        return fig

    def on_slider_change(change):
        """Updates the plot when the slider is moved."""
        selected_slice = change["new"]
        fig = plot_slice(selected_slice)
        output.clear_output(wait=True)
        with output:
            display(fig)

    slider = widgets.IntSlider(
        min=0, max=tensor.shape[2] - 1, step=1, value=0, description="Slice:"
    )
    slider.observe(on_slider_change, names="value")
    display(slider)

    output = widgets.Output()
    display(output)

    with output:
        display(plot_slice(0))


# =============== PATCHING & KNOCKOUT UTILS ===============
def patch_pos_head_vector(
    orig_head_vector: TT["batch", "pos", "head_index", "d_head"],
    hook,
    pos,
    head_index,
    patch_cache,
):
    """Patches a head vector at a given position and head index.

    Args:
        orig_head_vector (TT["batch", "pos", "head_index", "d_head"]): Original head activation vector.
        hook (Hook): Hook to patch.
        pos (int): Position to patch.
        head_index (int): Head index to patch.
        patch_cache (Dict[str, torch.Tensor]): Patch cache.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: Patched head vector.
    """
    orig_head_vector[:, pos, head_index, :] = patch_cache[hook.name][
        :, pos, head_index, :
    ]
    return orig_head_vector


def patch_head_vector(
    orig_head_vector: TT["batch", "pos", "head_index", "d_head"],
    hook,
    head_index,
    patch_cache,
):
    """Patches a head vector at a given head index.

    Args:
        orig_head_vector (TT["batch", "pos", "head_index", "d_head"]): Original head activation vector.
        hook (Hook): Hook to patch.
        head_index (int): Head index to patch.
        patch_cache (Dict[str, torch.Tensor]): Patch cache.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: Patched head vector.
    """
    orig_head_vector[:, :, head_index, :] = patch_cache[hook.name][:, :, head_index, :]
    return orig_head_vector


def get_path_patching_results(
    model,
    clean_tokens,
    patch_tokens,
    metric,
    step_metric,
    receiver_heads,
    receiver_type="hook_q",
    sender_heads=None,
    position=-1,
):
    """Gets the path patching results for a given model.

    Args:
        model (nn.Module): Model to patch.
        step_logit_diff (Tensor): Logit difference for the particular step/revision.
        receiver_heads (List[Tuple[int, int]]): List of tuples of layer and head indices to patch.
        receiver_type (str, optional): Type of receiver. Defaults to "hook_q".
        sender_heads (List[Tuple[int, int]], optional): List of tuples of layer and head indices to patch. Defaults to None.
        position (int, optional): Positions to patch. Defaults to -1.

    Returns:
        Tensor: Path patching results.
    """
    metric_delta_results = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device="cuda:0"
    )

    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            pass_d_hooks = path_patching(
                model=model,
                patch_tokens=patch_tokens,
                orig_tokens=clean_tokens,
                sender_heads=[(layer, head_idx)],
                receiver_hooks=[
                    (f"blocks.{layer_idx}.attn.{receiver_type}", head_idx)
                    for layer_idx, head_idx in receiver_heads
                ],
                positions=position,
            )

            

            path_patched_logits = model.run_with_hooks(
                clean_tokens, fwd_hooks=pass_d_hooks
            )
            patched_metric = metric(path_patched_logits)
            metric_delta_results[layer, head_idx] = (
                -(step_metric - patched_metric) / step_metric
            )
    return metric_delta_results


def ablate_top_head_hook(
    z: TT["batch", "pos", "head_index", "d_head"], hook, head_idx=0
):
    """Hook to ablate the top head of a given layer.

    Args:
        z (TT["batch", "pos", "head_index", "d_head"]): Attention weights.
        hook ([type]): Hook.
        head_idx (int, optional): Head index to ablate. Defaults to 0.

    Returns:
        TT["batch", "pos", "head_index", "d_head"]: Attention weights.
    """
    z[:, -1, head_idx, :] = 0
    return z


def get_knockout_perf_drop(model, heads_to_ablate, clean_tokens, metric):
    """Gets the performance drop for a given model and heads to ablate.

    Args:
        model (nn.Module): Model to knockout.
        heads_to_ablate (List[Tuple[int, int]]): List of tuples of layer and head indices to knockout.
        clean_tokens (Tensor): Clean tokens.
        answer_token_indices (Tensor): Answer token indices.

    Returns:
        Tensor: Performance drop.
    """
    # Adds a hook into global model state
    for layer, head in heads_to_ablate:
        ablate_head_hook = partial(ablate_top_head_hook, head_idx=head)
        model.blocks[layer].attn.hook_z.add_hook(ablate_head_hook)

    ablated_logits, ablated_cache = model.run_with_cache(clean_tokens)
    ablated_logit_diff = metric(ablated_logits)

    return ablated_logit_diff

# =========================== COMPONENT SWAPPING ===========================
# def get_components_to_swap(
#     model_name: str,
#     revision: str,
#     components: ComponentDict,
#     cache_dir: str,
# ) -> Dict[str, Tensor]:
#     """Gets the weights of the specified transformer components.

#     Args:
#         model_name (str): Model name in HuggingFace.
#         revision (str): Revision to load.
#         components (CircuitComponent): NamedTuple specifying the circuit components to collect.
#         cache_dir (str): Model cache directory.

#     Returns:
#         Dict[str, Tensor]: Dictionary of component parameters.
#     """

# def load_swapped_params(
#     model,
#     component_spec: ComponentDict,,
#     component_params: Dict[str, Tensor]
# ):
#     """Loads the specified component parameters into the model.

#     Args:
#         model: Model to load parameters into.
#         component_spec (CircuitComponent): NamedTuple specifying the circuit components to load.
#         component_params (Dict[str, Tensor]): Dictionary of component parameters.
#     """



# =========================== CIRCUITS OVER TIME ===========================

def get_data_and_metrics(
        model: HookedTransformer,
        task_name: str,
    ):
    assert task_name in ["ioi", "greater_than", "sentiment_cont", "sentiment_class", "mood_sentiment", "sst"]

    if task_name == "ioi":
        ds = UniversalPatchingDataset.from_ioi(model, 70)
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, positions=ds.positions)
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)
        accuracy_metric = partial(compute_accuracy, answer_token_indices=ds.answer_toks, positions=ds.positions)
        accuracy = CircuitMetric("accuracy", accuracy_metric)
        rank_0_metric = partial(compute_rank_0_rate, answer_token_indices=ds.answer_toks, positions=ds.positions)
        rank_0 = CircuitMetric("rank_0", rank_0_metric)
        probability_diff_metric = partial(compute_probability_diff, answer_token_indices=ds.answer_toks, positions=ds.positions)
        probability_diff = CircuitMetric("probability_diff", probability_diff_metric)
        probability_mass_metric = partial(compute_probability_mass, answer_token_indices=ds.answer_toks, positions=ds.positions)
        probability_mass = CircuitMetric("probability_mass", probability_mass_metric)
        mrr_metric = partial(compute_mean_reciprocal_rank, answer_token_indices=ds.answer_toks, positions=ds.positions)
        mrr = CircuitMetric("mrr", mrr_metric)
        metrics = [logit_diff, accuracy, rank_0, probability_diff, probability_mass, mrr]

    elif task_name == "greater_than":
        # Get data
        ds = UniversalPatchingDataset.from_greater_than(model, 1000)
        logit_diff_metric = partial(
            compute_logit_diff, 
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags, 
            mode="groups"
        )
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)
        prob_diff_metric = partial(
            compute_probability_diff, 
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="group_sum"
        )
        probability_diff = CircuitMetric("prob_diff", prob_diff_metric)
        probability_mass_metric = partial(
            compute_probability_mass,
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="group_sum"
        )
        probability_mass = CircuitMetric("prob_mass", probability_mass_metric)
        accuracy_metric = partial(
            compute_accuracy,
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="groups"
        )
        accuracy = CircuitMetric("accuracy", accuracy_metric)
        mrr_metric = partial(
            compute_mean_reciprocal_rank,
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="groups"
        )
        mrr = CircuitMetric("mrr", mrr_metric)
        max_group_mrr_metric = partial(
            compute_max_group_rank_reciprocal,
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="groups"
        )
        max_group_mrr = CircuitMetric("max_group_mrr", max_group_mrr_metric)
        metrics = [logit_diff, probability_diff, probability_mass, accuracy, mrr, max_group_mrr]

    elif task_name == "sentiment_cont":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "cont")
        
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, mode="pairs")
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)

        accuracy_metric = partial(compute_accuracy, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        accuracy = CircuitMetric("accuracy", accuracy_metric)
        
        rank_0_metric = partial(compute_rank_0_rate, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        rank_0 = CircuitMetric("rank_0", rank_0_metric)
        
        probability_diff_metric = partial(compute_probability_diff, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        probability_diff = CircuitMetric("probability_diff", probability_diff_metric)
        
        probability_mass_metric = partial(compute_probability_mass, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        probability_mass = CircuitMetric("probability_mass", probability_mass_metric)

        mrr_metric = partial(compute_mean_reciprocal_rank, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        mrr = CircuitMetric("mrr", mrr_metric)

        max_group_mrr_metric = partial(compute_max_group_rank_reciprocal, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        max_group_mrr = CircuitMetric("max_group_mrr", max_group_mrr_metric)
        
        metrics = [logit_diff, accuracy, rank_0, probability_diff, probability_mass, mrr, max_group_mrr]

    elif task_name == "sentiment_class":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "class")
        
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, mode="pairs")
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)

        accuracy_metric = partial(compute_accuracy, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        accuracy = CircuitMetric("accuracy", accuracy_metric)
        
        rank_0_metric = partial(compute_rank_0_rate, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        rank_0 = CircuitMetric("rank_0", rank_0_metric)
        
        probability_diff_metric = partial(compute_probability_diff, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        probability_diff = CircuitMetric("probability_diff", probability_diff_metric)
        
        probability_mass_metric = partial(compute_probability_mass, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        probability_mass = CircuitMetric("probability_mass", probability_mass_metric)

        mrr_metric = partial(compute_mean_reciprocal_rank, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        mrr = CircuitMetric("mrr", mrr_metric)

        max_group_mrr_metric = partial(compute_max_group_rank_reciprocal, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="pairs")
        max_group_mrr = CircuitMetric("max_group_mrr", max_group_mrr_metric)
        
        metrics = [logit_diff, accuracy, rank_0, probability_diff, probability_mass, mrr, max_group_mrr]

    elif task_name == "sst":
        # Get data
        ds = UniversalPatchingDataset.from_sst(model, 1000)
        
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, mode="simple")
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)

        accuracy_metric = partial(compute_accuracy, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="simple")
        accuracy = CircuitMetric("accuracy", accuracy_metric)
        
        rank_0_metric = partial(compute_rank_0_rate, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="simple")
        rank_0 = CircuitMetric("rank_0", rank_0_metric)
        
        probability_diff_metric = partial(compute_probability_diff, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="simple")
        probability_diff = CircuitMetric("probability_diff", probability_diff_metric)
        
        probability_mass_metric = partial(compute_probability_mass, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="simple")
        probability_mass = CircuitMetric("probability_mass", probability_mass_metric)

        mrr_metric = partial(compute_mean_reciprocal_rank, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="simple")
        mrr = CircuitMetric("mrr", mrr_metric)

        max_group_mrr_metric = partial(compute_max_group_rank_reciprocal, answer_token_indices=ds.answer_toks, positions=ds.positions, mode="simple")
        max_group_mrr = CircuitMetric("max_group_mrr", max_group_mrr_metric)
        
        metrics = [logit_diff, accuracy, rank_0, probability_diff, probability_mass, mrr, max_group_mrr]

    elif task_name == "mood_sentiment":
        raise ValueError("Not yet implemented")
    
    return ds, metrics
# DEPRECATED
# def get_chronological_circuit_performance(
#     model_hf_name: str,
#     model_tl_name: str,
#     cache_dir: str,
#     ckpts: List[int],
#     clean_tokens: Tensor,
#     corrupted_tokens: Tensor,
#     dataset: IOIDataset,
#     max_seq_len: int,
#     batch_size: int = None,
# ):
#     """Gets the performance of a model over time.

#     Args:
#         model_hf_name (str): Model name in HuggingFace.
#         model_tl_name (str): Model name in TorchLayers.
#         cache_dir (str): Cache directory.
#         ckpts (List[int]): Checkpoints to evaluate.
#         clean_tokens (Tensor): Clean tokens.
#         corrupted_tokens (Tensor): Corrupted tokens.
#         answer_token_indices (Tensor): Answer token indices.

#     Returns:
#         dict: Dictionary of performance over time.
#     """
#     logit_diff_vals = []
#     clean_ld_baselines = []
#     corrupted_ld_baselines = []

#     accuracy_vals = []
#     clean_accuracy_baselines = []
#     corrupted_accuracy_baselines = []

#     rank_0_rate_vals = []
#     clean_rank_0_rate_baselines = []
#     corrupted_rank_0_rate_baselines = []

#     get_logit_diff = partial(_logits_to_mean_logit_diff, ioi_dataset=dataset)
#     get_accuracy = partial(_logits_to_mean_accuracy, ioi_dataset=dataset)
#     get_rank_0_rate = partial(_logits_to_rank_0_rate, ioi_dataset=dataset)

#     previous_model = None

#     for ckpt in ckpts:

#         # Get model
#         if previous_model is not None:
#             clear_gpu_memory(previous_model)

#         print(f"Loading model for step {ckpt}...")
#         model = load_model(model_hf_name, model_tl_name, f"step{ckpt}", cache_dir)

#         # Get metric values
#         print("Getting metric values...")
#         if batch_size is None:
#             clean_logits = model(clean_tokens)
#             corrupted_logits = model(corrupted_tokens)
#         else:
#             clean_logits = run_with_batches(model, clean_tokens, batch_size, max_seq_len)
#             corrupted_logits = run_with_batches(model, corrupted_tokens, batch_size, max_seq_len)

#         clean_logit_diff = get_logit_diff(clean_logits)
#         corrupted_logit_diff = get_logit_diff(corrupted_logits)
#         clean_ld_baselines.append(clean_logit_diff)
#         corrupted_ld_baselines.append(corrupted_logit_diff)
#         print(f"Logit diff: {clean_logit_diff}")
#         logit_diff_vals.append(clean_logit_diff)

#         clean_accuracy = get_accuracy(clean_logits)
#         corrupted_accuracy = get_accuracy(corrupted_logits)
#         clean_accuracy_baselines.append(clean_accuracy)
#         corrupted_accuracy_baselines.append(corrupted_accuracy)
#         print(f"Accuracy: {clean_accuracy}")
#         accuracy_vals.append(clean_accuracy)

#         clean_rank_0_rate = get_rank_0_rate(clean_logits)
#         corrupted_rank_0_rate = get_rank_0_rate(corrupted_logits)
#         clean_rank_0_rate_baselines.append(clean_rank_0_rate)
#         corrupted_rank_0_rate_baselines.append(corrupted_rank_0_rate)
#         print(f"Rank 0 rate: {clean_rank_0_rate}")
#         rank_0_rate_vals.append(clean_rank_0_rate)

#         previous_model = model

#     return {
#         "logit_diffs": torch.tensor(logit_diff_vals),
#         "ld_clean_baselines": torch.tensor(clean_ld_baselines),
#         "ld_corrupted_baselines": torch.tensor(corrupted_ld_baselines),
#         "accuracy_vals": torch.tensor(accuracy_vals),
#         "accuracy_clean_baselines": torch.tensor(clean_accuracy_baselines),
#         "accuracy_corrupted_baselines": torch.tensor(corrupted_accuracy_baselines),
#         "rank_0_rate_vals": torch.tensor(rank_0_rate_vals),
#         "rank_0_rate_clean_baselines": torch.tensor(clean_rank_0_rate_baselines),
#         "rank_0_rate_corrupted_baselines": torch.tensor(corrupted_rank_0_rate_baselines),
#     }


def get_chronological_task_performance(
    model_hf_name: str,
    model_tl_name: str,
    config,
    cache_dir: str,
    ckpts: List[int],
    task: str = "ioi",
    batch_size: int = None,
    large_model=False,
):
    """Gets the performance of a model over time.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        task (str): The task for evaluation.
        batch_size (int, optional): Batch size to use for inference. Defaults to None.
        large_model (bool): Flag for loading large models.

    Returns:
        dict: Dictionary of performance over time.
    """

    metric_return = {}
    ds = None
    metrics = None

    results_dir = f"results/{config['model_name']}-no-dropout/{task}"
    os.makedirs(results_dir, exist_ok=True)

    for ckpt in ckpts:
        ckpt_key = f"step{ckpt}"
        # Check if this checkpoint is already processed
        if metric_return and all(ckpt_key in metric_return.get(metric.name, {}) for metric in metrics):
            print(f"Checkpoint {ckpt} already processed. Skipping.")
            continue

        print(f"Loading model for step {ckpt}...")
        if large_model:
            print("Loading large model...")
            # Assuming HookedTransformer is defined elsewhere
            model = HookedTransformer.from_pretrained(
                model_tl_name, 
                checkpoint_value=ckpt,
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
                dtype=torch.bfloat16,
                **{"cache_dir": cache_dir},
            )
        else:
            # Assuming load_model function is defined elsewhere
            model = load_model(model_hf_name, model_tl_name, ckpt_key, cache_dir)

        # Load data and metrics if this is the first iteration
        if not metrics:
            ds, metrics = get_data_and_metrics(model, task)
            metric_return = {metric.name: {} for metric in metrics}  # Initialize dict for each metric

        # Get metric values
        print("Getting metric values...")
        if batch_size is None:
            clean_logits = model(ds.toks)
        else:
            # Assuming run_with_batches is defined elsewhere
            clean_logits = run_with_batches(model, ds.toks, batch_size, ds.max_seq_len)

        # Update results in the dictionary
        for metric in metrics:
            new_result = metric(clean_logits)
            metric_return[metric.name][ckpt_key] = new_result

        # Save results after processing each checkpoint
        torch.save(metric_return, os.path.join(results_dir, "metrics.pt"))

        # Save results after processing each checkpoint
        torch.save(metric_return, os.path.join(results_dir, "metrics.pt"))

    return metric_return


def get_chronological_multi_task_performance(
    model_hf_name: str,
    model_tl_name: str,
    config,
    cache_dir: str,
    ckpts: List[int],
    batch_size: int = None,
    large_model=False,
):
    """Gets the performance of a model over time for multiple tasks.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        config (dict): Configuration dictionary containing tasks.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        batch_size (int, optional): Batch size to use for inference. Defaults to None.
        large_model (bool, optional): Flag to indicate if the model is large. Defaults to False.

    Returns:
        dict: Dictionary of performance over time for each task.
    """
    global_results_dir = f"results/{config['model_name']}-no-dropout"
    os.makedirs(global_results_dir, exist_ok=True)
    metrics_path = os.path.join(global_results_dir, "metrics.pt")

    # Load existing metrics dictionary or initialize it
    if os.path.isfile(metrics_path):
        metric_return = torch.load(metrics_path)
    else:
        metric_return = {task: {} for task in config["tasks"]}

    for ckpt in ckpts:
        ckpt_key = f"step{ckpt}"
        process_this_checkpoint = False

        # Check if this checkpoint needs processing
        for task in config["tasks"]:
            if metric_return[task] and all(ckpt_key in metric_return[task].get(metric, {}) for metric in metric_return[task]):
                continue
            else:
                process_this_checkpoint = True
                break

        if not process_this_checkpoint:
            print(f"Checkpoint {ckpt} has all tasks processed. Skipping.")
            continue

        print(f"Loading model for step {ckpt}...")
        # Load the model
        if large_model:
            print("Loading large model...")
            model = HookedTransformer.from_pretrained(
                model_tl_name, 
                checkpoint_value=ckpt,
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
                dtype=torch.bfloat16,
                **{"cache_dir": cache_dir},
            )
        else:
            model = load_model(model_hf_name, model_tl_name, ckpt_key, cache_dir)

        for task in config["tasks"]:
            ds, metrics = get_data_and_metrics(model, task)
            if batch_size is None:
                clean_logits = model(ds.toks)
            else:
                clean_logits = run_with_batches(model, ds.toks, batch_size, ds.max_seq_len)

            for metric in metrics:
                if ckpt_key not in metric_return[task].get(metric.name, {}):
                    new_result = metric(clean_logits)
                    if metric.name not in metric_return[task]:
                        metric_return[task][metric.name] = {}
                    metric_return[task][metric.name][ckpt_key] = new_result

        # Save the metrics dictionary after processing the checkpoint
        torch.save(metric_return, metrics_path)
    return metric_return

def get_acdcpp_circuits(
    model_name: str,
    cache_dir: str,
    clean_logit_diff,
    corrupt_logit_diff,
    ckpts,
    clean_data,
    corrupted_data,
    threshold,
    batch_size,
):
    previous_model = None

    for ckpt in ckpts:
        # Get model
        if previous_model is not None:
            clear_gpu_memory(previous_model)

        print(f"Loading model for step {ckpt}...")
        model = load_model(model_name,model_name, f"step{ckpt}", cache_dir = cache_dir)
        metric = partial(ioi_metric, clean_logit_diff = clean_logit_diff, corrupt_logit_diff = corrupt_logit_diff)
        return get_acdcpp_results(model, clean_data, corrupted_data, batch_size, threshold, metric)


def get_chronological_multi_task_performance(
    model_hf_name: str,
    model_tl_name: str,
    config,
    cache_dir: str,
    ckpts: List[int],
    batch_size: int = None,
    large_model=False,
):
    """Gets the performance of a model over time for multiple tasks.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        config (dict): Configuration dictionary containing tasks.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        batch_size (int, optional): Batch size to use for inference. Defaults to None.
        large_model (bool, optional): Flag to indicate if the model is large. Defaults to False.

    Returns:
        dict: Dictionary of performance over time for each task.
    """
    global_results_dir = f"results/{config['model_name']}-no-dropout"
    os.makedirs(global_results_dir, exist_ok=True)
    metrics_path = os.path.join(global_results_dir, "metrics.pt")

    # Load existing metrics dictionary or initialize it
    if os.path.isfile(metrics_path):
        metric_return = torch.load(metrics_path)
    else:
        metric_return = {task: {} for task in config["tasks"]}

    for ckpt in ckpts:
        ckpt_key = f"step{ckpt}"
        process_this_checkpoint = False

        # Check if this checkpoint needs processing
        for task in config["tasks"]:
            if metric_return[task] and all(ckpt_key in metric_return[task].get(metric, {}) for metric in metric_return[task]):
                continue
            else:
                process_this_checkpoint = True
                break

        if not process_this_checkpoint:
            print(f"Checkpoint {ckpt} has all tasks processed. Skipping.")
            continue

        print(f"Loading model for step {ckpt}...")
        # Load the model
        if large_model:
            print("Loading large model...")
            model = HookedTransformer.from_pretrained(
                model_tl_name, 
                checkpoint_value=ckpt,
                center_unembed=True,
                center_writing_weights=True,
                fold_ln=True,
                dtype=torch.bfloat16,
                **{"cache_dir": cache_dir},
            )
        else:
            model = load_model(model_hf_name, model_tl_name, ckpt_key, cache_dir)

        for task in config["tasks"]:
            ds, metrics = get_data_and_metrics(model, task)
            if batch_size is None:
                clean_logits = model(ds.toks)
            else:
                clean_logits = run_with_batches(model, ds.toks, batch_size, ds.max_seq_len)

            for metric in metrics:
                if ckpt_key not in metric_return[task].get(metric.name, {}):
                    new_result = metric(clean_logits)
                    if metric.name not in metric_return[task]:
                        metric_return[task][metric.name] = {}
                    metric_return[task][metric.name][ckpt_key] = new_result

        # Save the metrics dictionary after processing the checkpoint
        torch.save(metric_return, metrics_path)

    return metric_return



# =========================== COMPONENT SWAPPING ===========================
# Used for swapping components (mostly, attention heads) between different model checkpoints.
# Only works for Pythia models, and should be used for models with the same architecture.

class ComponentDict:
    """A dictionary to manage the components of a transformer model for parameter swapping.

    This class is used to specify which components (like attention heads, LayerNorm, MLP) of 
    a transformer model should be swapped. It handles the mapping of layers to specific heads 
    and the inclusion of other components like LayerNorm and MLP.

    Attributes:
        components (dict): A dictionary where keys are component names and values are 
                           either slice indices for attention heads or None for other components.

    Args:
        layer_heads (list of tuple): Each tuple contains a layer index and a head index (or indices) within that layer.
        include_ln (bool): If True, includes LayerNorm components for swapping.
        include_mlps (bool): If True, includes MLP components for swapping.
    """
    def __init__(
            self, 
            layer_heads: List[Tuple[int, int]], # Should be [(layer, head), ...)]
            include_ln: bool = False, # Probably shouldn't be used unless most of a layer is replaced
            include_mlps: bool = False 
        ):
        self.components = {}
        hidden_size = 768  # Assuming a hidden size of 768
        num_heads = 12     # Assuming 12 heads per layer
        head_size = hidden_size // num_heads

        # Create a dictionary to store head indices for each layer
        layer_to_heads = {}
        for layer, head in layer_heads:
            if layer not in layer_to_heads:
                layer_to_heads[layer] = []
            layer_to_heads[layer].append(head)

        for layer, heads in layer_to_heads.items():
            for head in heads:
                # Calculate start and end indices for each specified head
                start_idx = head * head_size
                end_idx = start_idx + head_size

                # Store the slice information for each head's weights
                component_weight_key = f'gpt_neox.layers.{layer}.attention.query_key_value.weight'
                if component_weight_key not in self.components:
                    self.components[component_weight_key] = []
                self.components[component_weight_key].append((start_idx, end_idx))

                # Store the slice information for each head's biases
                component_bias_key = f'gpt_neox.layers.{layer}.attention.query_key_value.bias'
                if component_bias_key not in self.components:
                    self.components[component_bias_key] = []
                self.components[component_bias_key].append((start_idx, end_idx))

            # Add LayerNorm components if specified
            if include_ln:
                self.components[f'gpt_neox.layers.{layer}.input_layernorm'] = None
                self.components[f'gpt_neox.layers.{layer}.post_attention_layernorm'] = None

            # Add MLP components if specified
            if include_mlps:
                self.components[f'gpt_neox.layers.{layer}.mlp'] = None

    def get_component_specs(self):
        """Retrieves the component specifications.

        Returns:
            dict: The dictionary containing component specifications.
        """
        return self.components


def get_components_to_swap(source_model, component_dict, cache_dir):
    """Extracts the specified components from a source transformer model.

    This function extracts the components (like specific attention head weights and biases, 
    LayerNorm, and MLP components) specified in the ComponentDict from the source model.

    Args:
        source_model (transformers.PreTrainedModel): The model from which components are to be extracted.
        component_dict (ComponentDict): The ComponentDict specifying which components to extract.
        cache_dir (str): Directory for caching the model.

    Returns:
        dict: A dictionary with component names as keys and tuples (extracted parameters, slice info) as values.
    """
    component_params = {}
    for name, param in source_model.named_parameters():
        comp_specs = component_dict.get_component_specs().get(name)
        if comp_specs is not None:
            # Handle multiple slices for both weights and biases
            if "bias" in name:
                # Bias is a 1D tensor
                slices = [param.detach().clone()[start:end] for start, end in comp_specs]
            else:
                # Weights are a 2D tensor
                slices = [param.detach().clone()[:, start:end] for start, end in comp_specs]
            concatenated_slices = torch.cat(slices, dim=-1)  # Concatenate on the last dimension
            component_params[name] = (concatenated_slices, comp_specs)
        elif comp_specs is None and name in component_dict.get_component_specs():
            # Handle non-sliced components
            component_params[name] = param.detach().clone()
    return component_params


def load_swapped_params(target_model, component_params):
    """Loads the specified components into a target transformer model.

    This function takes the components extracted from a source model and loads them into
    the corresponding components of the target model. It handles both sliced components 
    (like specific attention heads) and whole components (like LayerNorm and MLP).

    Args:
        target_model (transformers.PreTrainedModel): The model into which the components are to be loaded.
        component_params (dict): A dictionary with component names as keys and tuples (parameters to load, slice info) as values.

    Raises:
        ValueError: If there's a mismatch in the shape of the parameters being loaded.
    """
    for name, param in target_model.named_parameters():
        if name in component_params:
            new_param_data, slice_info = component_params[name]
            if slice_info is not None:
                head_size = new_param_data.shape[-1] // len(slice_info)  # Adjust head size calculation
                for i, (start_idx, end_idx) in enumerate(slice_info):
                    if param.data.ndim == 2:
                        param.data[:, start_idx:end_idx] = new_param_data[:, i*head_size:(i+1)*head_size]
                    elif param.data.ndim == 1:
                        param.data[start_idx:end_idx] = new_param_data[i*head_size:(i+1)*head_size]
            elif slice_info is None:
                # For non-sliced components, replace the entire parameter
                param.data = new_param_data


# def run_chronological_swapping_experiment(
#     model_hf_name: str,
#     model_tl_name: str,
#     cache_dir: str,
#     ckpts: List[int],
#     inbound_swap_intervals: List[int],
#     clean_tokens: Tensor,
#     corrupted_tokens: Tensor,
#     dataset: IOIDataset,
#     component_dict: ComponentDict,
#     include_ln: bool = False,
#     include_mlps: bool = False,
# ):
#     """Runs a chronological swapping experiment for a given model and components.

#     This function loads a target model at each checkpoint. For each inbound swap interval,
#     it loads a source model from an alternate checkpoint and swaps the specified components
#     from the source model into the target model. It then evaluates the performance of the
#     target model on clean and corrupted tokens, including logit diff, accuracy, and rank 0 rate.

#     Args:
#         model_hf_name (str): Model name in HuggingFace.
#         model_tl_name (str): Model name in TorchLayers.
#         cache_dir (str): Cache directory.
#         ckpts (List[int]): Checkpoints to evaluate.
#         inbound_swap_intervals (List[int]): Intervals at which to swap components from source models.
#         clean_tokens (Tensor): Clean tokens.
#         corrupted_tokens (Tensor): Corrupted tokens.
#         dataset (IOIDataset): IOIDataset object.
#         component_dict (ComponentDict): The ComponentDict specifying which components to extract.
#         include_ln (bool): If True, includes LayerNorm components for swapping.
#         include_mlps (bool): If True, includes MLP components for swapping.

#     Returns:
#         dict: Dictionary of performance over time.
#     """
#     logit_diff_vals = []
#     clean_ld_baselines = []
#     corrupted_ld_baselines = []

#     accuracy_vals = []
#     clean_accuracy_baselines = []
#     corrupted_accuracy_baselines = []

#     rank_0_rate_vals = []
#     clean_rank_0_rate_baselines = []
#     corrupted_rank_0_rate_baselines = []

#     get_logit_diff = partial(_logits_to_mean_logit_diff, ioi_dataset=dataset)
#     get_accuracy = partial(_logits_to_mean_accuracy, ioi_dataset=dataset)
#     get_rank_0_rate = partial(_logits_to_rank_0_rate, ioi_dataset=dataset)

#     previous_model = None

#     for ckpt in ckpts:

#         # Get model
#         if previous_model is not None:
#             clear_gpu_memory(previous_model)

#         print(f"Loading model for step {ckpt}...")
#         model = load_model(model_hf_name, model_tl_name, f"step{ckpt}", cache_dir)

#         # Get metric values
#         print("Getting metric values...")
#         clean_logits, clean_cache = model.run_with_cache(clean_tokens)
#         corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

#         clean_logit_diff = get_logit_diff(clean_logits)
#         corrupted_logit_diff = get_logit_diff(corrupted_logits)
#         clean_ld_baselines.append(clean_logit_diff)
#         corrupted_ld_baselines.append(corrupted_logit_diff)
#         print(f"Logit diff: {clean_logit_diff}")
#         logit_diff_vals.append(clean_logit_diff)

#         clean_accuracy = get_accuracy(clean_logits)
#         corrupted_accuracy = get_accuracy(corrupted_logits)
#         clean_accuracy_baselines.append(clean_accuracy)
#         corrupted_accuracy_baselines.append(corrupted_accuracy)
#         print(f"Accuracy: {clean_accuracy}")
#         accuracy_vals.append(clean_accuracy)

#         clean_rank_0_rate = get_rank_0_rate(clean_logits)
#         corrupted_rank_0_rate = get_rank_0_rate(corrupted_logits)
#         clean_rank_0_rate_baselines.append(clean_rank_0_rate)
#         corrupted_rank_0_rate_baselines.append(corrupted_rank_0_rate)
#         print(f"Rank 0 rate: {clean_rank_0_rate}")
#         rank_0_rate_vals.append(clean_rank_0_rate)

#         # Swap components from source models
#         for swap_interval in inbound_swap_intervals:
#             if ckpt >= swap_interval
#                 source_ckpt = ckpt - swap_interval
#                 print(f"Loading source model for step {source_ckpt}...")
#                 source_model = load_model(model_hf_name, model_tl_name, f"step{source_ckpt}", cache_dir)

#                 print(f"Swapping components from step {source_ckpt} into step {ckpt}...")
#                 component_params = get_components_to_swap(source_model, component_dict, cache_dir)
#                 load_swapped_params(model, component_params)
