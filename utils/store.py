import numpy as np
import os
from transformer_lens import HookedTransformer
from typing import Union
import torch
import plotly.graph_objects as go
import pickle


def clean_label(label: str) -> str:
    label = label.replace('.npy', '')
    label = label.replace('.html', '')
    label = label.replace('data/', '')
    label = label.replace('.csv', '')
    label = label.replace('.txt', '')
    label = label.replace('.pkl', '')
    label = label.replace('.pdf', '')
    assert "/" not in label, "Label must not contain slashes"
    return label


def get_model_name(model: Union[HookedTransformer, str]) -> str:
    if isinstance(model, HookedTransformer):
        assert len(model.name) > 0, "Model must have a name"
        model = model.name
    model = model.replace('EleutherAI/', '')
    return model


def save_array(
        array: Union[np.ndarray, torch.Tensor], 
        label: str, 
        model: Union[HookedTransformer, str]
    ):
    model: str = get_model_name(model)
    if isinstance(array, torch.Tensor):
        array = array.cpu().detach().numpy()
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'wb') as f:
        np.save(f, array)
    return path


def load_array(label: str, model: Union[HookedTransformer, str]) -> np.ndarray:
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    path = os.path.join(model_path, label + '.npy')
    with open(path, 'rb') as f:
        array = np.load(f)
    return array


def save_html(
        fig: go.Figure,
        label: str, 
        model: Union[HookedTransformer, str]
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    path = os.path.join(model_path, label + '.html')
    fig.write_html(path)
    return path


def get_model_name(model: Union[HookedTransformer, str]) -> str:
    if isinstance(model, HookedTransformer):
        assert len(model.cfg.model_name) > 0, "Model must have a name"
        model = model.cfg.model_name
    model = model.replace('EleutherAI/', '')
    if model == 'gpt2':
        model = 'gpt2-small'
    return model


def load_pickle(
    label: str,
    model: Union[HookedTransformer, str],
):
    model: str = get_model_name(model)
    label = clean_label(label)
    model_path = os.path.join('data', model)
    path = os.path.join(model_path, label + '.pkl')
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj