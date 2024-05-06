from typing import Dict, Any, Tuple, List, Optional, Literal, Union, Iterable
from transformer_lens import HookedTransformer, utils
import einops
from jaxtyping import Int, Float, Bool
import torch as t
from torch import Tensor
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from IPython.display import display, clear_output, HTML

def get_effective_embedding(model: HookedTransformer, use_codys_without_attention_changes=True) -> Float[Tensor, "d_vocab d_model"]:
    # TODO - implement Neel's variation; attention to self from the token
    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid_pre = W_E.unsqueeze(0)

    if not use_codys_without_attention_changes:
        pre_attention = model.blocks[0].ln1(resid_pre)
        attn_out = einops.einsum(
            pre_attention, 
            model.W_V[0],
            model.W_O[0],
            "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
        )
        resid_mid = attn_out + resid_pre
        del pre_attention, attn_out
    else:
        resid_mid = resid_pre

    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    del resid_pre, resid_mid, normalized_resid_mid, mlp_out
    t.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
        "W_U": W_U.T,
    }

def devices_are_equal(device_1: Union[str, t.device], device_2: Union[str, t.device]):
    '''
    Helper function, because devices "cuda:0" and "cuda" are actually the same.
    '''
    device_set = set([str(device_1), str(device_2)])
    
    return (len(device_set) == 1) or (device_set == {"cuda", "cuda:0"})

def first_occurrence(array_1D):
    series = pd.Series(array_1D)
    duplicates = series.duplicated(keep='first')
    inverted = ~duplicates
    return inverted.values

def first_occurrence_2d(tensor_2D):
    device = tensor_2D.device
    array_2D = utils.to_numpy(tensor_2D)
    return t.from_numpy(np.array([first_occurrence(row) for row in array_2D])).to(device)


def concat_dicts(d1: Dict[str, Tensor], d2: Dict[str, Tensor]) -> Dict[str, Tensor]:
    '''
    Given 2 dicts, return the dict of concatenated tensors along the zeroth dimension.

    Special case: if d1 is empty, we just return d2.

    Also, we make sure that d2 tensors are moved to cpu.
    '''
    if len(d1) == 0: return d2
    assert d1.keys() == d2.keys()
    return {k: t.cat([d1[k], d2[k]], dim=0) for k in d1.keys()}


def concat_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def kl_div(
    logits1: Float[Tensor, "... d_vocab"],
    logits2: Float[Tensor, "... d_vocab"],
):
    '''
    Estimates KL divergence D_KL( logits1 || logits2 ), i.e. where logits1 is the "ground truth".

    Each tensor is assumed to have all dimensions be the batch dimension, except for the last one
    (which is a distribution over the vocabulary).

    In our use-cases, logits1 will be the non-ablated version of the model.
    '''

    logprobs1 = logits1.log_softmax(dim=-1)
    logprobs2 = logits2.log_softmax(dim=-1)
    logprob_diff = logprobs1 - logprobs2
    probs1 = logits1.softmax(dim=-1)

    return einops.reduce(
        probs1 * logprob_diff,
        "... d_vocab -> ...",
        reduction = "sum",
    )


def make_list_correct_length(L, K, pad_tok: Optional[str] = None):
    '''
    If len(L) < K, pad list L with its last element until it is of length K.
    If len(L) > K, truncate.

    Special case when len(L) == 0, we just put the BOS token in it.
    '''
    if len(L) == 0:
        L = ["<|endoftext|>"]

    if pad_tok is None:
        pad_tok = L[-1]

    if len(L) <= K:
        L = L + [pad_tok] * (K - len(L))
    else:
        L = L[:K]

    assert len(L) == K
    return L


def parse_str(s: str):
    doubles = "“”"
    singles = "‘’"
    for char in doubles: s = s.replace(char, '"')
    for char in singles: s = s.replace(char, "'")
    return s

def parse_str_tok_for_printing(s: str):
    s = s.replace("\n", "\\n")
    return s

def parse_str_toks_for_printing(s: List[str]):
    return list(map(parse_str_tok_for_printing, s))


def get_webtext(seed: int = 420, dataset="stas/openwebtext-10k") -> List[str]:
    """Get 10,000 sentences from the OpenWebText dataset"""

    # Let's see some WEBTEXT
    raw_dataset = load_dataset(dataset)
    train_dataset = raw_dataset["train"]
    dataset = [train_dataset[i]["text"] for i in range(len(train_dataset))]

    # Shuffle the dataset (I don't want the Hitler thing being first so use a seeded shuffle)
    np.random.seed(seed)
    np.random.shuffle(dataset)

    return dataset


def process_webtext(
    seed: int,
    batch_size: int,
    seq_len: int,
    model: HookedTransformer,
    verbose: bool = False,
    return_indices: bool = False,
    use_tqdm: bool = False,
    prepend_bos: bool = True,
) -> Tuple[Int[Tensor, "batch seq"], List[List[str]]]:
    
    DATA_STR_ALL = get_webtext(seed=seed, dataset="stas/openwebtext-10k" if "llama" not in model.cfg.model_name.lower() else "stas/c4-en-10k") # A large part of LLAMA training data is common crawl 
    DATA_STR_ALL = [parse_str(s) for s in DATA_STR_ALL]
    DATA_STR = []

    count = 0
    indices = []
    iter = tqdm(range(len(DATA_STR_ALL))) if use_tqdm else range(len(DATA_STR_ALL))
    for i in iter:
        num_toks = len(model.to_tokens(DATA_STR_ALL[i], prepend_bos=prepend_bos).squeeze())
        if num_toks > seq_len:
            DATA_STR.append(DATA_STR_ALL[i])
            indices.append(i)
            count += 1
        if count == batch_size:
            break
    else:
        raise Exception("Couldn't find enough sequences of sufficient length. Inly" + str(count) + "found.")

    DATA_TOKS = model.to_tokens(DATA_STR)
    DATA_STR_TOKS = model.to_str_tokens(DATA_STR)

    if seq_len < 1024:
        DATA_TOKS = DATA_TOKS[:, :seq_len]
        DATA_STR_TOKS = [str_toks[:seq_len] for str_toks in DATA_STR_TOKS]

    DATA_STR_TOKS_PARSED = list(map(parse_str_toks_for_printing, DATA_STR_TOKS))

    clear_output()
    if verbose:
        print(f"Shape = {DATA_TOKS.shape}\n")
        print("First prompt:\n" + "".join(DATA_STR_TOKS[0]))

    if return_indices: 
        return DATA_TOKS, DATA_STR_TOKS_PARSED, indices

    return DATA_TOKS, DATA_STR_TOKS_PARSED