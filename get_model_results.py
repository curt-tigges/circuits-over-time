import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import yaml
import gc

import einops
from fancy_einsum import einsum

from datasets import load_dataset
from transformers import pipeline

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"

import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from huggingface_hub import notebook_login

import transformer_lens
import transformer_lens.utils as utils
import transformer_lens.patching as patching
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

from functools import partial

from torchtyping import TensorType as TT

torch.set_grad_enabled(False)
DO_SLOW_RUNS = True

# define the model names
model_name = "pythia-1.4b"
model_tl_name = "pythia-1.3b"

model_full_name = f"EleutherAI/{model_name}"
model_tl_full_name = f"EleutherAI/{model_tl_name}"


def clear_gpu_memory(model):
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_hf_name, model_tl_name, revision, cache_dir=None):

    if cache_dir == None:
        cache_dir = f"/fsx/home-curt/saved_models/{model_name}/{revision}"

    # Download model from HuggingFace
    source_model = AutoModelForCausalLM.from_pretrained(
        model_hf_name, revision=revision, cache_dir=cache_dir
    )

    # Load model into TransformerLens
    model = HookedTransformer.from_pretrained(
        model_tl_name,
        hf_model=source_model,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
    )

    return model


model = load_model(model_full_name, model_tl_full_name, revision="step143000")


# define circuit
from collections import namedtuple

CircuitComponent = namedtuple(
    "CircuitComponent", ["heads", "position", "receiver_type"]
)

circuit = {
    "name-movers": CircuitComponent(
        [(12, 15), (13, 1), (13, 6), (15, 15), (16, 13), (17, 7)], -1, "hook_q"
    ),
    "s2-inhibition": CircuitComponent([(10, 7)], 10, "hook_v"),
    # "duplicate-name": CircuitComponent([(7, 15), (9, 1)], 10, 'head_v'),
    # "induction": CircuitComponent([], 10, 'head_v')
}

# set up data
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]


def set_up_data(model, prompts, answers):
    clean_tokens = model.to_tokens(prompts)
    # Swap each adjacent pair, with a hacky list comprehension
    corrupted_tokens = clean_tokens[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]
    # print("Clean string 0", model.to_string(clean_tokens[0]))
    # print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

    answer_token_indices = torch.tensor(
        [
            [model.to_single_token(answers[i][j]) for j in range(2)]
            for i in range(len(answers))
        ],
        device=model.cfg.device,
    )
    # print("Answer token indices", answer_token_indices)
    return clean_tokens, corrupted_tokens, answer_token_indices


clean_tokens, corrupted_tokens, answer_token_indices = set_up_data(
    model, prompts, answers
)

# activation patching
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()


clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff


def ioi_metric(
    logits,
    clean_baseline=CLEAN_BASELINE,
    corrupted_baseline=CORRUPTED_BASELINE,
    answer_token_indices=answer_token_indices,
):
    return (get_logit_diff(logits, answer_token_indices) - corrupted_baseline) / (
        clean_baseline - corrupted_baseline
    )


clean_baseline_ioi = ioi_metric(clean_logits, CLEAN_BASELINE, CORRUPTED_BASELINE)
corrupted_baseline_ioi = ioi_metric(
    corrupted_logits, CLEAN_BASELINE, CORRUPTED_BASELINE
)

print(
    f"Clean Baseline is 1: {ioi_metric(clean_logits, CLEAN_BASELINE, CORRUPTED_BASELINE).item():.4f}"
)
print(
    f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits, CLEAN_BASELINE, CORRUPTED_BASELINE).item():.4f}"
)


clear_gpu_memory(model)


def patch_pos_head_vector(
    orig_head_vector: TT["batch", "pos", "head_index", "d_head"],
    hook,
    pos,
    head_index,
    patch_cache,
):
    # print(patch_cache.keys())
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
    orig_head_vector[:, :, head_index, :] = patch_cache[hook.name][:, :, head_index, :]
    return orig_head_vector


def path_patching(
    model,
    patch_tokens,
    orig_tokens,
    sender_heads,
    receiver_hooks,
    positions=-1,
):
    """
    Patch in the effect of `sender_heads` on `receiver_hooks` only
    (though MLPs are "ignored" if `freeze_mlps` is False so are slight confounders in this case - see Appendix B of https://arxiv.org/pdf/2211.00593.pdf)

    TODO fix this: if max_layer < model.cfg.n_layers, then let some part of the model do computations (not frozen)
    """

    def patch_positions(z, source_act, hook, positions=["end"], verbose=False):
        for pos in positions:
            z[torch.arange(orig_tokens.N), orig_tokens.word_idx[pos]] = source_act[
                torch.arange(patch_tokens.N), patch_tokens.word_idx[pos]
            ]
        return z

    # process arguments
    sender_hooks = []
    for layer, head_idx in sender_heads:
        if head_idx is None:
            sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))

        else:
            sender_hooks.append((f"blocks.{layer}.attn.hook_z", head_idx))

    sender_hook_names = [x[0] for x in sender_hooks]
    receiver_hook_names = [x[0] for x in receiver_hooks]
    receiver_hook_heads = [x[1] for x in receiver_hooks]
    # Forward pass A (in https://arxiv.org/pdf/2211.00593.pdf)
    source_logits, sender_cache = model.run_with_cache(patch_tokens)

    # Forward pass B
    target_logits, target_cache = model.run_with_cache(orig_tokens)

    # Forward pass C
    # Cache the receiver hooks
    # (adding these hooks first means we save values BEFORE they are overwritten)
    receiver_cache = model.add_caching_hooks(lambda x: x in receiver_hook_names)

    # "Freeze" intermediate heads to their orig_tokens values
    # q, k, and v will get frozen, and then if it's a sender head, this will get undone
    # z, attn_out, and the MLP will all be recomputed and added to the residual stream
    # however, the effect of the change on the residual stream will be overwritten by the
    # freezing for all non-receiver components
    pass_c_hooks = []
    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                hook_name = hook_template.format(layer)
                if (hook_name, head_idx) not in receiver_hooks:
                    # print(f"Freezing {hook_name}")
                    hook = partial(
                        patch_head_vector, head_index=head_idx, patch_cache=target_cache
                    )
                    pass_c_hooks.append((hook_name, hook))
                else:
                    pass
                    # print(f"Not freezing {hook_name}")

    # These hooks will overwrite the freezing, for the sender heads
    # We also carry out pass C
    for hook_name, head_idx in sender_hooks:
        assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
            hook_name,
            head_idx,
        )
        hook = partial(
            patch_pos_head_vector,
            pos=positions,
            head_index=head_idx,
            patch_cache=sender_cache,
        )
        pass_c_hooks.append((hook_name, hook))

    receiver_logits = model.run_with_hooks(orig_tokens, fwd_hooks=pass_c_hooks)
    # Add (or return) all the hooks needed for forward pass D
    pass_d_hooks = []

    for hook_name, head_idx in receiver_hooks:
        # for pos in positions:
        # if torch.allclose(
        #     receiver_cache[hook_name][torch.arange(orig_tokens.N), orig_tokens.word_idx[pos]],
        #     target_cache[hook_name][torch.arange(orig_tokens.N), orig_tokens.word_idx[pos]],
        # ):
        #     warnings.warn("Torch all close for {}".format(hook_name))
        hook = partial(
            patch_pos_head_vector,
            pos=positions,
            head_index=head_idx,
            patch_cache=receiver_cache,
        )
        pass_d_hooks.append((hook_name, hook))

    return pass_d_hooks


def get_path_patching_results(
    model,
    clean_baseline,
    corrupted_baseline,
    receiver_heads,
    receiver_type="hook_q",
    sender_heads=None,
    position=-1,
):

    metric_delta_results = torch.zeros(
        model.cfg.n_layers, model.cfg.n_heads, device="cuda:0"
    )

    for layer in range(model.cfg.n_layers):
        for head_idx in range(model.cfg.n_heads):
            pass_d_hooks = path_patching(
                model=model,
                patch_tokens=corrupted_tokens,
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
            iot_metric_res = ioi_metric(
                path_patched_logits, clean_baseline, corrupted_baseline
            )
            metric_delta_results[layer, head_idx] = (
                -(clean_baseline_ioi - iot_metric_res) / clean_baseline_ioi
            )
    return metric_delta_results


def ablate_top_head_hook(
    z: TT["batch", "pos", "head_index", "d_head"], hook, head_idx=0
):
    z[:, -1, head_idx, :] = 0
    return z


def get_knockout_perf_drop(model, heads_to_ablate, clean_baseline, corrupted_baseline):
    # Adds a hook into global model state
    for layer, head in heads_to_ablate:
        ablate_head_hook = partial(ablate_top_head_hook, head_idx=head)
        model.blocks[layer].attn.hook_z.add_hook(ablate_head_hook)

    ablated_logits, ablated_cache = model.run_with_cache(clean_tokens)
    ablated_ioi_metric = ioi_metric(
        ablated_logits,
        clean_baseline=clean_baseline,
        corrupted_baseline=corrupted_baseline,
    )

    return ablated_ioi_metric


def get_chronological_circuit_data(
    model_hf_name, model_tl_name, start_ckpt, end_ckpt, ckpt_interval, metric, circuit
):
    ckpt_count = (end_ckpt - start_ckpt) / ckpt_interval
    metric_vals = []
    attn_head_vals = []
    value_patch_vals = []
    circuit_vals = {key: [] for key in circuit.keys()}
    activation_patching_vals = {key: [] for key in circuit.keys()}
    knockout_drops = {key: [] for key in circuit.keys()}
    # Loop through all checkpoints in range, getting metrics for each
    # for ckpt in range(start_ckpt, end_ckpt, ckpt_interval):

    # Powers of 2 up to 143000, rounded to the nearest thousand after 1000
    ckpts = [
        round((2**i) / 1000) * 1000 if 2**i > 1000 else 2**i for i in range(18)
    ]
    # ckpts = [2 ** i for i in range(10)] + [i * 1000 for i in range(1, 144)]

    previous_model = None

    for ckpt in ckpts:

        # Get model
        if previous_model is not None:
            clear_gpu_memory(previous_model)
        print(f"Loading model {model_hf_name} from checkpoint {ckpt}")
        model = load_model(model_hf_name, model_tl_name, f"step{ckpt}")

        # Get metric values (relative to final performance)
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

        clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
        corrupted_logit_diff = get_logit_diff(
            corrupted_logits, answer_token_indices
        ).item()

        metric = partial(
            metric, clean_baseline=CLEAN_BASELINE, corrupted_baseline=CORRUPTED_BASELINE
        )
        metric_val = metric(clean_logits)
        metric_vals.append(metric_val)

        metric = partial(
            metric,
            clean_baseline=clean_logit_diff,
            corrupted_baseline=corrupted_logit_diff,
        )

        # Get attention pattern patching metrics
        attn_head_out_all_pos_act_patch_results = (
            patching.get_act_patch_attn_head_pattern_all_pos(
                model, corrupted_tokens, clean_cache, metric
            )
        )
        attn_head_vals.append(attn_head_out_all_pos_act_patch_results)

        # Get value patching metrics
        value_patch_results = patching.get_act_patch_attn_head_v_all_pos(
            model, corrupted_tokens, clean_cache, metric
        )
        value_patch_vals.append(value_patch_results)

        # Get path patching metrics for specific circuit parts
        for key in circuit.keys():
            # Get path patching results
            path_patching_results = get_path_patching_results(
                model,
                clean_logit_diff,
                corrupted_logit_diff,
                circuit[key].heads,
                receiver_type=circuit[key].receiver_type,
                position=circuit[key].position,
            )
            circuit_vals[key].append(path_patching_results)

            # Get knockout performance drop
            knockout_drops[key].append(
                get_knockout_perf_drop(
                    model, circuit[key].heads, clean_logit_diff, corrupted_logit_diff
                )
            )

        previous_model = model

    return (
        torch.tensor(metric_vals),
        torch.stack(attn_head_vals, dim=-1),
        torch.stack(value_patch_vals, dim=-1),
        circuit_vals,
        knockout_drops,
    )


(
    overall_perf,
    attn_head_perf,
    value_perf,
    circuit_vals,
    knockout_drops,
) = get_chronological_circuit_data(
    model_full_name,
    model_tl_full_name,
    start_ckpt=1000,
    end_ckpt=50000,
    ckpt_interval=1000,
    metric=ioi_metric,
    circuit=circuit,
)

ckpts = [round((2**i) / 1000) * 1000 if 2**i > 1000 else 2**i for i in range(18)]

import pickle

# load the saved data
overall_perf = torch.load(f"results/{model_name}-no-dropout/overall_perf.pt")
attn_head_perf = torch.load(f"results/{model_name}-no-dropout/attn_head_perf.pt")
value_perf = torch.load(f"results/{model_name}-no-dropout/value_perf.pt")
with open(f"results/{model_name}-no-dropout/circuit_vals.pkl", "rb") as f:
    circuit_vals = pickle.load(f)
with open(f"results/{model_name}-no-dropout/knockout_drops.pkl", "rb") as f:
    knockout_drops = pickle.load(f)
