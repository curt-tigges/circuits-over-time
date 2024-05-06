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

from datasets import load_from_disk
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

from data.sva.create_dataset import create_dataset as create_sva_dataset
from data.sva.utils import get_singular_and_plural
from data.greater_than_dataset import YearDataset, get_valid_years, get_year_indices
#from data.sentiment_datasets import get_dataset, PromptType

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


def generate_data_and_caches(model: HookedTransformer, N: int, verbose: bool = False, seed: int = 42, prepend_bos: bool = False):

    ioi_dataset = IOIDataset(
        prompt_type="mixed",
        N=N,
        tokenizer=model.tokenizer,
        prepend_bos=prepend_bos,
        seed=seed,
        device=str(device)
    )

    abc_dataset = ioi_dataset.gen_flipped_prompts("ABB->ABA, BAB->BAA")

    model.reset_hooks(including_permanent=True)

    #ioi_logits_original, ioi_cache = model.run_with_cache(ioi_dataset.toks)
    #abc_logits_original, abc_cache = model.run_with_cache(abc_dataset.toks)

    #ioi_average_logit_diff = _logits_to_mean_logit_diff(ioi_logits_original, ioi_dataset).item()
    #abc_average_logit_diff = _logits_to_mean_logit_diff(abc_logits_original, ioi_dataset).item()

    #if verbose:
    #    print(f"Average logit diff (IOI dataset): {ioi_average_logit_diff:.4f}")
    #    print(f"Average logit diff (ABC dataset): {abc_average_logit_diff:.4f}")

    # ioi_metric_noising = partial(
    #     _ioi_metric_noising,
    #     clean_logit_diff=ioi_average_logit_diff,
    #     corrupted_logit_diff=abc_average_logit_diff,
    #     ioi_dataset=ioi_dataset,
    # )

    return ioi_dataset, abc_dataset #, ioi_cache, abc_cache, ioi_metric_noising


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


def prepare_indices_for_sva(model, pluralities):
    """
    Prepares two tensors for use with the compute_probability_diff function in 'groups' mode.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to convert verbs to token indices.
        pluralities (torch.Tensor): Tensor containing the plurality fo the subject for each prompt in the batch.

    Returns:
        torch.Tensor, torch.Tensor: Two tensors, one for token IDs and one for correct/incorrect flags.
    """

    # Get the indices for present-tense verbs
    singular_verb_indices, plural_verb_indices = get_singular_and_plural(model, strict=False)  # Tensor with token IDs for verbs
    all_verb_indices = torch.cat((singular_verb_indices, plural_verb_indices), dim=0)
    singular_index = len(singular_verb_indices)

    # Prepare tensors to store token IDs and correct/incorrect flags
    token_ids_tensor = all_verb_indices.repeat(pluralities.size(0), 1)  # Repeat the year_indices for each batch item
    flags_tensor = torch.zeros_like(token_ids_tensor)  # Initialize the flags tensor with zeros

    for i, plurality in enumerate(pluralities):
        # plural case
        if plurality:
            # Mark plural verbs correct
            flags_tensor[i, singular_index:] = 1
            # Mark singular verbs incorrect
            flags_tensor[i, :singular_index] = -1
        # singular case
        else:
            # Mark plural verbs incorrect
            flags_tensor[i, singular_index:] = -1
            # Mark singular verbs correct
            flags_tensor[i, :singular_index] = 1

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
        item = {
            'toks': self.toks[idx],
            'flipped_toks': self.flipped_toks[idx],
            'answer_toks': self.answer_toks[idx]
        }
        if self.positions is not None:
            item['positions'] = self.positions[idx]
        if self.group_flags is not None:
            item['flags_tensor'] = self.group_flags[idx]
        return item

    @classmethod
    def from_ioi(cls, model, size: int = 70):
        ioi_dataset, abc_dataset = generate_data_and_caches(model, size, verbose=True)
        answer_tokens = torch.cat((torch.Tensor(ioi_dataset.io_tokenIDs).unsqueeze(1), torch.Tensor(ioi_dataset.s_tokenIDs).unsqueeze(1)), dim=1).to(device)
        answer_tokens = answer_tokens.long()

        return cls(ioi_dataset.toks, abc_dataset.toks, answer_tokens, 21, ioi_dataset.word_idx["end"])

    @classmethod
    def from_greater_than(cls, model, size: int = 1000):
        ds = YearDataset(get_valid_years(model.tokenizer, 1100, 1800), size, None, model.tokenizer)
        answer_tokens, group_flags = prepare_indices_for_prob_diff(model.tokenizer, torch.Tensor(ds.years_YY))

        good_lens = ds.good_attn.sum(-1)
        bad_lens = ds.bad_attn.sum(-1)
        assert torch.all(good_lens == bad_lens)

        max_len = good_lens.max()
        end_idx = good_lens - 1

        return cls(ds.good_toks, ds.bad_toks, answer_tokens, max_len, positions=end_idx, group_flags=group_flags)
    
    @classmethod
    def from_csv(cls, model: HookedTransformer, filename, clean_col, corrupted_col, clean_label_col, corrupted_label_col, size: int = 1000):
        df = pd.read_csv(filename).sample(frac=1).head(size)

        good_toks = model.tokenizer(df[clean_col].tolist(), return_tensors='pt',padding='longest')
        bad_toks = model.tokenizer(df[corrupted_col].tolist(), return_tensors='pt',padding='longest')

        good_lens = good_toks['attention_mask'].sum(-1)
        bad_lens = bad_toks['attention_mask'].sum(-1)
        assert torch.all(good_lens == bad_lens)

        max_len = good_lens.max()
        end_idx = good_lens - 1

        clean_label_idx = torch.tensor(df[clean_label_col].tolist())
        corrupted_label_idx = torch.tensor(df[corrupted_label_col].tolist())
        answer_tokens = torch.stack([clean_label_idx, corrupted_label_idx], dim=1)

        return cls(good_toks['input_ids'], bad_toks['input_ids'], answer_tokens, max_len, end_idx)

    @classmethod
    def from_capital_country(cls, model: HookedTransformer, size: int = 1000):
        return cls.from_csv(model, 'data/capital-country.csv', 'clean', 'corrupted', 'country_idx', 'corrupted_country_idx', size=size)
    
    @classmethod
    def from_country_capital(cls, model: HookedTransformer, size: int = 1000):
        return cls.from_csv(model, 'data/country-capital.csv', 'clean', 'corrupted', 'capital_idx', 'corrupted_capital_idx', size=size)
        
    @classmethod
    def from_gender_bias(cls, model: HookedTransformer, size: int = 1000):
        return cls.from_csv(model, 'data/gender-bias.csv', 'clean', 'corrupted', 'clean_answer_idx', 'corrupted_answer_idx', size=size)
    
    @classmethod
    def from_gender_pronoun(cls, model: HookedTransformer, size: int = 1000):
        return cls.from_csv(model, 'data/gender-pronoun.csv', 'clean', 'corrupted', 'clean_answer_idx', 'corrupted_answer_idx', size=size)

    @classmethod
    def from_gender_both(cls, model: HookedTransformer, size: int = 1000):
        return cls.from_csv(model, 'data/gender-both.csv', 'clean', 'corrupted', 'clean_answer_idx', 'corrupted_answer_idx', size=size)

    @classmethod
    def from_sva(cls, model, size: int = 1000):
        clean, corrupted, labels, max_len, positions = create_sva_dataset(model.tokenizer, size)
        answer_tokens, group_flags = prepare_indices_for_sva(model, labels)

        return cls(clean, corrupted, answer_tokens, max_len, positions=positions, group_flags=group_flags)

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
    def from_sst(cls, model, size: int = 1000):
        ds = load_from_disk("sst_zero_shot_balanced_EleutherAI_pythia-2.8b")

        # Turn all items in ['tokens'] into a single tensor
        all_tokens = torch.cat([item['tokens'].unsqueeze(0) for item in ds], dim=0).to(device)
        all_answers = torch.cat([item['answers'].unsqueeze(0) for item in ds], dim=0).to(device)
        all_positions = torch.cat([item['final_pos_index'].unsqueeze(0) for item in ds], dim=0).to(device)

        return cls(all_tokens[:size], all_tokens[:size], all_answers[:size], 64, all_positions[:size])
        

    @classmethod
    def from_mood_sentiment():
        pass

    