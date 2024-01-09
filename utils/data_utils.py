import os
import pathlib
from typing import List, Optional, Union

import torch
import numpy as np
import pandas as pd
import yaml

from torch import Tensor
from jaxtyping import Float

import einops
from fancy_einsum import einsum

from datasets import load_dataset
from transformers import pipeline
import plotly.io as pio
import plotly.express as px

# import pysvelte
from IPython.display import HTML

import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display

import transformers
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
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

from path_patching_cm.path_patching import Node, IterNode, path_patch, act_patch
from path_patching_cm.ioi_dataset import IOIDataset, NAMES
from utils.metrics import _logits_to_mean_logit_diff, _ioi_metric_noising

from functools import partial

from torchtyping import TensorType as TT

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def read_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    prompts_str, answers_str = content.split("\n\n")
    prompts = prompts_str.split("\n")  # Remove the last empty item
    answers = [
        tuple(answer.split(",")) for answer in answers_str.split(";")[:-1]
    ]  # Remove the last empty item

    return prompts, answers





def generate_data_and_caches(model: HookedTransformer, N: int, verbose: bool = False, seed: int = 42):

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=False,
        seed=seed,
        device=str(device)
    )

    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA")

    model.reset_hooks(including_permanent=True)

    ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
    abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

    ioi_average_logit_diff = _logits_to_mean_logit_diff(ioi_logits_original, ioi_dataset).item()
    abc_average_logit_diff = _logits_to_mean_logit_diff(abc_logits_original, ioi_dataset).item()

    if verbose:
        print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
        print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

    ioi_metric_noising = partial(
        _ioi_metric_noising,
        clean_logit_diff=ioi_average_logit_diff,
        corrupted_logit_diff=abc_average_logit_diff,
        ioi_dataset=ioi_dataset,
    )

    return ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising
