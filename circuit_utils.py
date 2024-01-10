import os
from functools import partial
from collections import namedtuple

import torch
from torch import Tensor
from typing import List, Optional, Union, Dict
from path_patching_cm.ioi_dataset import IOIDataset
from torchtyping import TensorType as TT


import transformer_lens.patching as patching
from transformer_lens import HookedTransformer

import plotly.graph_objs as go
import torch
import ipywidgets as widgets
from IPython.display import display

from utils.model_utils import load_model, clear_gpu_memory
from utils.metrics import _logits_to_mean_logit_diff, _logits_to_mean_accuracy, _logits_to_rank_0_rate

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"

# =============== CIRCUIT ===============
CircuitComponent = namedtuple(
    "CircuitComponent", ["heads", "position", "receiver_type"]
)


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
def get_components_to_swap(
    model_name: str,
    revision: str,
    components: ComponentDict,
    cache_dir: str,
) -> Dict[str, Tensor]:
    """Gets the weights of the specified transformer components.

    Args:
        model_name (str): Model name in HuggingFace.
        revision (str): Revision to load.
        components (CircuitComponent): NamedTuple specifying the circuit components to collect.
        cache_dir (str): Model cache directory.

    Returns:
        Dict[str, Tensor]: Dictionary of component parameters.
    """

def load_swapped_params(
    model,
    component_spec: ComponentDict,,
    component_params: Dict[str, Tensor]
):
    """Loads the specified component parameters into the model.

    Args:
        model: Model to load parameters into.
        component_spec (CircuitComponent): NamedTuple specifying the circuit components to load.
        component_params (Dict[str, Tensor]): Dictionary of component parameters.
    """



# =========================== CIRCUITS OVER TIME ===========================
def get_chronological_circuit_performance(
    model_hf_name: str,
    model_tl_name: str,
    cache_dir: str,
    ckpts: List[int],
    clean_tokens: Tensor,
    corrupted_tokens: Tensor,
    dataset: IOIDataset,
):
    """Gets the performance of a model over time.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        clean_tokens (Tensor): Clean tokens.
        corrupted_tokens (Tensor): Corrupted tokens.
        answer_token_indices (Tensor): Answer token indices.

    Returns:
        dict: Dictionary of performance over time.
    """
    logit_diff_vals = []
    clean_ld_baselines = []
    corrupted_ld_baselines = []

    accuracy_vals = []
    clean_accuracy_baselines = []
    corrupted_accuracy_baselines = []

    rank_0_rate_vals = []
    clean_rank_0_rate_baselines = []
    corrupted_rank_0_rate_baselines = []

    get_logit_diff = partial(_logits_to_mean_logit_diff, ioi_dataset=dataset)
    get_accuracy = partial(_logits_to_mean_accuracy, ioi_dataset=dataset)
    get_rank_0_rate = partial(_logits_to_rank_0_rate, ioi_dataset=dataset)

    previous_model = None

    for ckpt in ckpts:

        # Get model
        if previous_model is not None:
            clear_gpu_memory(previous_model)

        print(f"Loading model for step {ckpt}...")
        model = load_model(model_hf_name, model_tl_name, f"step{ckpt}", cache_dir)

        # Get metric values
        print("Getting metric values...")
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = get_logit_diff(clean_logits)
        corrupted_logit_diff = get_logit_diff(corrupted_logits)
        clean_ld_baselines.append(clean_logit_diff)
        corrupted_ld_baselines.append(corrupted_logit_diff)
        print(f"Logit diff: {clean_logit_diff}")
        logit_diff_vals.append(clean_logit_diff)

        clean_accuracy = get_accuracy(clean_logits)
        corrupted_accuracy = get_accuracy(corrupted_logits)
        clean_accuracy_baselines.append(clean_accuracy)
        corrupted_accuracy_baselines.append(corrupted_accuracy)
        print(f"Accuracy: {clean_accuracy}")
        accuracy_vals.append(clean_accuracy)

        clean_rank_0_rate = get_rank_0_rate(clean_logits)
        corrupted_rank_0_rate = get_rank_0_rate(corrupted_logits)
        clean_rank_0_rate_baselines.append(clean_rank_0_rate)
        corrupted_rank_0_rate_baselines.append(corrupted_rank_0_rate)
        print(f"Rank 0 rate: {clean_rank_0_rate}")
        rank_0_rate_vals.append(clean_rank_0_rate)

        previous_model = model

    return {
        "logit_diffs": torch.tensor(logit_diff_vals),
        "ld_clean_baselines": torch.tensor(clean_ld_baselines),
        "ld_corrupted_baselines": torch.tensor(corrupted_ld_baselines),
        "accuracy_vals": torch.tensor(accuracy_vals),
        "accuracy_clean_baselines": torch.tensor(clean_accuracy_baselines),
        "accuracy_corrupted_baselines": torch.tensor(corrupted_accuracy_baselines),
        "rank_0_rate_vals": torch.tensor(rank_0_rate_vals),
        "rank_0_rate_clean_baselines": torch.tensor(clean_rank_0_rate_baselines),
        "rank_0_rate_corrupted_baselines": torch.tensor(corrupted_rank_0_rate_baselines),
    }


def get_chronological_circuit_data(
    model_name: str,
    cache_dir: str,
    ckpts,
    circuit,
    clean_tokens,
    corrupted_tokens,
    answer_token_indices,
):
    """Extracts data from different circuit components over time.

    Args:
        model_hf_name (str): Model name in HuggingFace.
        model_tl_name (str): Model name in TorchLayers.
        cache_dir (str): Cache directory.
        ckpts (List[int]): Checkpoints to evaluate.
        circuit (dict): Circuit dictionary.
        clean_tokens (Tensor): Clean tokens.
        corrupted_tokens (Tensor): Corrupted tokens.
        answer_token_indices (Tensor): Answer token indices.

    Returns:
        dict: Dictionary of data over time.
    """
    logit_diff_vals = []
    clean_ld_baselines = []
    corrupted_ld_baselines = []
    attn_head_vals = []
    value_patch_vals = []
    circuit_vals = {key: [] for key in circuit.keys()}
    knockout_drops = {key: [] for key in circuit.keys()}

    metric = partial(get_logit_diff, answer_token_indices=answer_token_indices)

    previous_model = None

    for ckpt in ckpts:

        # Get model
        if previous_model is not None:
            clear_gpu_memory(previous_model)

        print(f"Loading model for step {ckpt}...")
        model = load_model(model_name, f"step{ckpt}", cache_dir)

        # Get metric values (relative to final performance)
        print("Getting metric values...")
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = metric(clean_logits).item()
        corrupted_logit_diff = metric(corrupted_logits).item()

        clean_ld_baselines.append(clean_logit_diff)
        corrupted_ld_baselines.append(corrupted_logit_diff)

        logit_diff_vals.append(clean_logit_diff)

        # Get attention pattern patching metrics
        print("Getting attention pattern patching metrics...")
        attn_head_out_all_pos_act_patch_results = (
            patching.get_act_patch_attn_head_pattern_all_pos(
                model, corrupted_tokens, clean_cache, metric
            )
        )
        attn_head_vals.append(attn_head_out_all_pos_act_patch_results)

        # Get value patching metrics
        print("Getting value patching metrics...")
        value_patch_results = patching.get_act_patch_attn_head_v_all_pos(
            model, corrupted_tokens, clean_cache, metric
        )
        value_patch_vals.append(value_patch_results)

        # Get path patching metrics for specific circuit parts
        for key in circuit.keys():
            # Get path patching results
            print(f"Getting path patching metrics for {key}...")
            # TODO: Replace with Callum's patch patching code
            path_patching_results = get_path_patching_results(
                model,
                clean_tokens,
                corrupted_tokens,
                metric,
                clean_logit_diff,
                circuit[key].heads,
                receiver_type=circuit[key].receiver_type,
                position=circuit[key].position,
            )
            circuit_vals[key].append(path_patching_results)

            # Get knockout performance drop
            print(f"Getting knockout performance drop for {key}...")
            knockout_drops[key].append(
                get_knockout_perf_drop(model, circuit[key].heads, clean_tokens, metric)
            )

        previous_model = model

    return {
        "logit_diffs": torch.tensor(logit_diff_vals),
        "clean_baselines": torch.tensor(clean_ld_baselines),
        "corrupted_baselines": torch.tensor(corrupted_ld_baselines),
        "attn_head_vals": torch.stack(attn_head_vals, dim=-1),
        "value_patch_vals": torch.stack(value_patch_vals, dim=-1),
        "circuit_vals": circuit_vals,
        "knockout_drops": knockout_drops,
    }


