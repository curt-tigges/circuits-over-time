import os
import pathlib
from pathlib import Path
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

from data.greater_than_dataset import YearDataset, get_valid_years, get_year_indices
from data.sentiment_datasets import get_dataset, PromptType

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


def set_up_data(model, prompts, answers):
    """Sets up data for a given model, prompts, and answers.

    Args:
        model (HookedTransformer): Model to set up data for.
        prompts (List[str]): List of prompts to use.
        answers (List[List[str]]): List of answers to use.

    Returns:
        Tuple[List[str], List[str], torch.Tensor]: Clean tokens, corrupted tokens, and answer token indices.
    """
    clean_tokens = model.to_tokens(prompts)
    # Swap each adjacent pair of tokens
    corrupted_tokens = clean_tokens[
        [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
    ]

    answer_token_indices = torch.tensor(
        [
            [model.to_single_token(answers[i][j]) for j in range(2)]
            for i in range(len(answers))
        ],
        device=model.cfg.device,
    )

    return clean_tokens, corrupted_tokens, answer_token_indices

def _logits_to_mean_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average. Used only for legacy IOI dataset generation code.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def _ioi_metric_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float,
        corrupted_logit_diff: float,
        ioi_dataset: IOIDataset,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = _logits_to_mean_logit_diff(logits, ioi_dataset)
        return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()


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


def prepare_indices_for_prob_diff(tokenizer, years):
    """
    Prepares two tensors for use with the compute_probability_diff function in 'groups' mode.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to convert years to token indices.
        years (torch.Tensor): Tensor containing the year for each prompt in the batch.

    Returns:
        torch.Tensor, torch.Tensor: Two tensors, one for token IDs and one for correct/incorrect flags.
    """

    # Get the indices for years 00 to 99
    year_indices = get_year_indices(tokenizer)  # Tensor of size 100 with token IDs for years

    # Prepare tensors to store token IDs and correct/incorrect flags
    token_ids_tensor = year_indices.repeat(years.size(0), 1)  # Repeat the year_indices for each batch item
    flags_tensor = torch.zeros_like(token_ids_tensor)  # Initialize the flags tensor with zeros

    for i, year in enumerate(years):
        # Mark years greater than the given year as correct (1)
        flags_tensor[i, year + 1:] = 1
        # Mark years less than or equal to the given year as incorrect (-1)
        flags_tensor[i, :year + 1] = -1

    return token_ids_tensor, flags_tensor


class UniversalPatchingDataset():

    def __init__(
            self,
            toks: torch.Tensor,
            flipped_toks: torch.Tensor,
            answer_toks: torch.Tensor,
            max_seq_len: int,
            positions: Optional[torch.Tensor] = None,
            group_flags: Optional[torch.Tensor] = None
    ) -> None:
        self.toks = toks
        self.flipped_toks = flipped_toks
        self.answer_toks = answer_toks
        self.max_seq_len = max_seq_len

        # optional attributes
        self.positions = positions # used for IOI
        self.group_flags = group_flags # used for greater_than, optional for others

    def __len__(self):
        return len(self.toks)
    
    def __getitem__(self, idx):
        data = {
            'toks': self.toks[idx],
            'flipped_toks': self.flipped_toks[idx],
            'answer_toks': self.answer_toks[idx]
        }
        if self.positions is not None:
            data['positions'] = self.positions[idx]
        if self.group_flags is not None:
            data['group_flags'] = self.group_flags[idx]
        return data

    @classmethod
    def from_ioi(cls, model, size: int = 70):
        ioi_dataset, abc_dataset, _, _, _ = generate_data_and_caches(model, size, verbose=True)
        answer_tokens = torch.cat((torch.Tensor(ioi_dataset.io_tokenIDs).unsqueeze(1), torch.Tensor(ioi_dataset.s_tokenIDs).unsqueeze(1)), dim=1).to(device)
        answer_tokens = answer_tokens.long()

        return cls(ioi_dataset.toks, abc_dataset.toks, answer_tokens, 21, ioi_dataset.word_idx["end"])

    @classmethod
    def from_greater_than(cls, model, size: int = 1000):
        ds = YearDataset(get_valid_years(model.tokenizer, 1100, 1800), size, Path("data/potential_nouns.txt"), model.tokenizer)
        answer_tokens, group_flags = prepare_indices_for_prob_diff(model.tokenizer, torch.Tensor(ds.years_YY))

        return cls(ds.good_toks, ds.bad_toks, answer_tokens, 12, group_flags=group_flags)

    @classmethod
    def from_sentiment(cls, model, task_type: str):
        if task_type == "cont":
            ds_type = PromptType.COMPLETION_2
        elif task_type == "class":
            ds_type = PromptType.CLASSIFICATION_4
        else:
            raise ValueError(f"task_type must be 'cont' or 'class', got {task_type}")

        ds = get_dataset(model, device, prompt_type=ds_type)
        
        return cls(ds.clean_tokens, ds.corrupted_tokens, ds.answer_tokens, 28)

    @classmethod
    def from_mood_sentiment():
        pass