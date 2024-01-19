
import os
from typing import List, Optional, Union, Dict, Tuple

import torch
from torch import Tensor
import numpy as np
import einops
from fancy_einsum import einsum
import circuitsvis as cv

import transformer_lens.utils as utils

from transformer_lens import HookedTransformer
import transformer_lens.patching as patching

from transformers import AutoModelForCausalLM

from torch import Tensor
from jaxtyping import Float
import plotly.express as px

from functools import partial

from torchtyping import TensorType as TT

from path_patching_cm.path_patching import Node, IterNode, path_patch, act_patch
from path_patching_cm.ioi_dataset import IOIDataset, NAMES
from neel_plotly import imshow as imshow_n

from utils.visualization import imshow_p, plot_attention_heads, plot_attention

from utils.visualization_utils import (
    plot_attention_heads,
    scatter_attention_and_contribution,
    get_attn_head_patterns
)

from utils.circuit_utils import (
    ComponentDict,
    get_components_to_swap,
    load_swapped_params
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

def test_component_dict():
    # Example: Test for layer 9, heads 0, 1, 2
    component_dict = ComponentDict(layer_heads=[(9, 0), (9, 1), (9, 2)], include_ln=False, include_mlps=False)
    component_specs = component_dict.get_component_specs()

    # Check for the presence of correct keys and slices
    assert 'gpt_neox.layers.9.attention.query_key_value.weight' in component_specs
    assert 'gpt_neox.layers.9.attention.query_key_value.bias' in component_specs
    assert len(component_specs['gpt_neox.layers.9.attention.query_key_value.weight']) == 3  # 3 heads
    assert len(component_specs['gpt_neox.layers.9.attention.query_key_value.bias']) == 3  # 3 heads

    # Add more assertions as needed to test the slices

    print("Test ComponentDict: Passed")


def test_get_components_to_swap():
    source_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    component_dict = ComponentDict(layer_heads=[(9, 0), (9, 1)], include_ln=False, include_mlps=False)
    component_params = get_components_to_swap(source_model, component_dict, cache_dir="./")

    hidden_size = source_model.config.hidden_size
    num_heads = source_model.config.num_attention_heads
    head_size = hidden_size // num_heads

    for name, (concatenated_param, slices) in component_params.items():
        if "query_key_value.weight" in name or "query_key_value.bias" in name:
            for i, (start, end) in enumerate(slices):
                expected_slice = source_model.get_parameter(name).data[:, start:end] if "weight" in name else source_model.get_parameter(name).data[start:end]
                actual_slice = concatenated_param[:, i*head_size:(i+1)*head_size] if concatenated_param.ndim == 2 else concatenated_param[i*head_size:(i+1)*head_size]
                assert torch.allclose(expected_slice, actual_slice), f"Slice mismatch in {name} for head {i}"

    print("Test get_components_to_swap: Passed")


def test_load_swapped_params():
    source_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    target_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    component_dict = ComponentDict(layer_heads=[(9, 0), (9, 1)], include_ln=False, include_mlps=False)
    

    hidden_size = target_model.config.hidden_size
    num_heads = target_model.config.num_attention_heads
    head_size = hidden_size // num_heads

    # Modify source model's specified components
    for name, param in source_model.named_parameters():
        if name in component_dict.get_component_specs():
            param.data.add_(0.123)  # Arbitrary modification for testing

    # Perform parameter swapping
    component_params = get_components_to_swap(source_model, component_dict, cache_dir="./")
    load_swapped_params(target_model, component_params)

    # Verify the parameters are correctly updated in the target model
    for name, (concatenated_param, slices) in component_params.items():
        if "query_key_value.weight" in name or "query_key_value.bias" in name:
            target_param = target_model.get_parameter(name)
            for i, (start, end) in enumerate(slices):
                target_slice = target_param.data[:, start:end] if target_param.ndim == 2 else target_param.data[start:end]
                source_slice = concatenated_param[:, i*head_size:(i+1)*head_size] if concatenated_param.ndim == 2 else concatenated_param[i*head_size:(i+1)*head_size]
                assert torch.allclose(target_slice, source_slice), f"Mismatch in parameters for {name} head {i}"

    print("Test load_swapped_params: Passed")
