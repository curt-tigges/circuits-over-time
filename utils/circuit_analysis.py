import os
from functools import partial

from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union, Int
from typeguard import typechecked

from transformers import PreTrainedTokenizerBase
from transformer_lens.utils import get_attention_mask
from transformer_lens import ActivationCache, HookedTransformer

import torch
from torchtyping import TensorType as TT
from torch import Tensor
from jaxtyping import Float
from typing import Tuple
import einops

from fancy_einsum import einsum

import plotly.graph_objs as go
import torch
import ipywidgets as widgets
from IPython.display import display



if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"


def get_final_non_pad_token(
    logits: Float[Tensor, "batch pos vocab"],
    attention_mask: Int[Tensor, "batch pos"],
) -> Float[Tensor, "batch vocab"]:
    """Gets the final non-pad token from a tensor.

    Args:
        logits (torch.Tensor): Logits to use.
        attention_mask (torch.Tensor): Attention mask to use.

    Returns:
        torch.Tensor: Final non-pad token logits.
    """
    # Get the last non-pad token
    position_index = einops.repeat(
        torch.arange(logits.shape[1], device=logits.device),
        "pos -> batch pos",
        batch=logits.shape[0],
    )
    masked_position = torch.where(
        attention_mask == 0, torch.full_like(position_index, -1), position_index
    )
    last_non_pad_token = einops.reduce(
        masked_position, "batch pos -> batch", reduction="max"
    )
    assert (last_non_pad_token >= 0).all()
    # Get the final token logits
    final_token_logits = logits[
        torch.arange(logits.shape[0]), last_non_pad_token, :
    ]
    return final_token_logits


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


# =============== METRIC UTILS ===============

def get_logit_diff(
    logits: Float[Tensor, "batch pos vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"], 
    per_prompt: bool = False,
    per_completion: bool = False,
):
    """
    Gets the difference between the logits of the provided tokens 
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    logits = einops.repeat(
        logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    left_logits: Float[Tensor, "batch n_pairs"] = logits.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    )
    right_logits: Float[Tensor, "batch n_pairs"] = logits.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    )
    if per_completion:
        print(left_logits - right_logits)
    left_logits: Float[Tensor, "batch"] = left_logits.mean(dim=1)
    right_logits: Float[Tensor, "batch"] = right_logits.mean(dim=1)
    if per_prompt:
        return left_logits - right_logits

    return (left_logits - right_logits).mean()

def get_log_probs(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs"],
    per_prompt: bool = False,
) -> Float[Tensor, "batch n_pairs"]:
    
    n_pairs = answer_tokens.shape[1]
    if len(logits.shape) == 3:
        # Get final logits only
        logits: Float[Tensor, "batch vocab"] = logits[:, -1, :]
    assert len(answer_tokens.shape) == 2
    
    # convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # get the log probs for the answer tokens
    log_probs = einops.repeat(
        log_probs, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    answer_log_probs: Float[Tensor, "batch n_pairs"] = log_probs.gather(
        -1, answer_tokens.unsqueeze(-1)
    )
    # average over the answer tokens
    answer_log_probs: Float[Tensor, "batch"] = answer_log_probs.mean(dim=1)
    if per_prompt:
        return answer_log_probs
    else:
        return answer_log_probs.mean()


def log_prob_diff_noising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs"],
    flipped_log_prob: float,
    clean_log_prob: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()
    
def log_prob_diff_denoising(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch n_pairs 2"],
        flipped_log_prob: float,
        clean_log_prob: float,
        return_tensor: bool = False,
) -> Float[Tensor, ""]:
    """
    """
    log_prob = get_log_probs(logits, answer_tokens)
    ld = ((log_prob - flipped_log_prob) / (clean_log_prob  - flipped_log_prob))
    if return_tensor:
        return ld
    else:
        return ld.item()

def logit_diff_denoising(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch n_pairs 2"],
    flipped_logit_diff: float,
    clean_logit_diff: float,
    return_tensor: bool = False,
) -> Float[Tensor, ""]:
    '''
    Linear function of logit diff, calibrated so that it equals 0 when performance is
    same as on flipped input, and 1 when performance is same as on clean input.
    '''
    patched_logit_diff = get_logit_diff(logits, answer_tokens)
    ld = ((patched_logit_diff - flipped_logit_diff) / (clean_logit_diff  - flipped_logit_diff))
    if return_tensor:
        return ld
    else:
        return ld.item()


def logit_diff_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float,
        corrupted_logit_diff: float,
        answer_tokens: Float[Tensor, "batch n_pairs 2"],
        return_tensor: bool = False,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = get_logit_diff(logits, answer_tokens)
        ld = ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

        if return_tensor:
            return ld
        else:
            return ld.item()


# =============== LOGIT LENS UTILS ===============


def residual_stack_to_logit_diff(residual_stack, logit_diff_directions, prompts, cache):
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / len(prompts)


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


def get_final_token_logits(
    logits: Float[Tensor, "batch *pos vocab"],
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, "batch vocab"]:
    if tokenizer is None and logits.ndim == 3:
        final_token_logits = logits[:, -1, :]
    elif tokenizer is None and logits.ndim == 2:
        final_token_logits = logits
    else:
        mask = get_attention_mask(
            tokenizer, tokens, prepend_bos=False
        )
        final_token_logits = get_final_non_pad_token(logits, mask)
    return final_token_logits


@typechecked
def get_prob_diff(
    logits: Float[Tensor, "batch *pos vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"], 
    per_prompt: bool = False,
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
) -> Float[Tensor, "*batch"]:
    """
    Gets the difference between the softmax probabilities of the provided tokens 
    e.g., the correct and incorrect tokens in IOI

    Args:
        logits (torch.Tensor): Logits to use.
        answer_tokens (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the softmax probs of the provided tokens.
    """
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    final_token_logits: Float[Tensor, "batch vocab"] = get_final_token_logits(
        logits, tokens=tokens, tokenizer=tokenizer
    )
    repeated_logits: Float[Tensor, "batch n_pairs d_vocab"] = einops.repeat(
        final_token_logits, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    probs: Float[Tensor, "batch n_pairs vocab"] = repeated_logits.softmax(dim=-1)
    left_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 0].unsqueeze(-1)
    ).squeeze(-1)
    right_probs: Float[Tensor, "batch n_pairs"] = probs.gather(
        -1, answer_tokens[:, :, 1].unsqueeze(-1)
    ).squeeze(-1)
    left_probs_batch: Float[Tensor, "batch"] = left_probs.mean(dim=1)
    right_probs_batch: Float[Tensor, "batch"] = right_probs.mean(dim=1)
    if per_prompt:
        return left_probs_batch - right_probs_batch

    return (left_probs_batch - right_probs_batch).mean()


def get_log_probs(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"],
    tokens: Optional[Int[Tensor, "batch pos"]] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    per_prompt: bool = False,
) -> Float[Tensor, "batch n_pairs"]:
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    n_pairs = answer_tokens.shape[1]
    logits: Float[Tensor, "batch vocab"] = get_final_token_logits(
        logits, tokens=tokens, tokenizer=tokenizer
    )
    assert len(answer_tokens.shape) == 2
    
    # convert logits to log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # get the log probs for the answer tokens
    log_probs_repeated = einops.repeat(
        log_probs, "batch vocab -> batch n_pairs vocab", n_pairs=n_pairs
    )
    answer_log_probs: Float[Tensor, "batch n_pairs"] = log_probs_repeated.gather(
        -1, answer_tokens.unsqueeze(-1)
    )
    # average over the answer tokens
    answer_log_probs_batch: Float[Tensor, "batch"] = answer_log_probs.mean(dim=1)
    if per_prompt:
        return answer_log_probs_batch
    else:
        return answer_log_probs_batch.mean()
    

def center_logit_diffs(
    logit_diffs: Float[Tensor, "batch"],
    answer_tokens: Int[Tensor, "batch *n_pairs 2"], 
) -> Tuple[Float[Tensor, "batch"], float]:
    """
    Useful to debias a model when using as a binary classifier
    """
    device = logit_diffs.device
    if answer_tokens.ndim == 2:
        answer_tokens = answer_tokens.unsqueeze(1)
    is_positive = (
        answer_tokens[:, 0, 0] == 
        answer_tokens[0, 0, 0]
    ).to(device=device)
    bias = torch.where(
        is_positive, logit_diffs, -logit_diffs
    ).mean().to(device=device)
    debiased = (
        logit_diffs - torch.where(is_positive, bias, -bias)
    )
    return debiased, bias.item()


def get_accuracy_from_logit_diffs(
    logit_diffs: Float[Tensor, "batch"]
):
    return (logit_diffs > 0).float().mean()