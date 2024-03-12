import sys, os
import gc
import torch
import warnings
from typing import Dict, Any, Tuple, List, Optional, Literal, Union, Iterable
import nltk
from transformer_lens import HookedTransformer, utils
from functools import partial
import einops
from dataclasses import dataclass, field, InitVar
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
from jaxtyping import Int, Float, Bool
import torch as t
from collections import defaultdict
import time
from torch import Tensor
import pandas as pd
import numpy as np
from copy import copy
import math
from tqdm.auto import tqdm
import itertools
import pickle

from utils.cspa_extra_utils import (
    get_effective_embedding,
    devices_are_equal,
    first_occurrence_2d,
    concat_dicts,
    kl_div
)

Head = Tuple[int, int]


FUNCTION_STR_TOKS =  [
    '\x00',
    '\x01',
    '\x02',
    '\x03',
    '\x04',
    '\x05',
    '\x06',
    '\x07',
    '\x08',
    '\t',
    '\n',
    '\x0b',
    '\x0c',
    '\r',
    '\x0e',
    '\x0f',
    '\x10',
    '\x11',
    '\x12',
    '\x13',
    '\x14',
    '\x15',
    '\x16',
    '\x17',
    '\x18',
    '\x19',
    '\x1a',
    '\x1b',
    '\x1c',
    '\x1d',
    '\x1e',
    '\x1f',
    '\x7f',
    ' a',
    ' the',
    ' an',
    ' to',
    ' of',
    ' in',
    ' that',
    ' for',
    ' it',
    ' with',
    ' as',
    ' was',
    ' The',
    ' are',
    ' by',
    ' have',
    ' this',
    'The',
    ' will',
    ' they',
    ' their',
    ' which',
    ' about',
    ' In',
    ' like',
    ' them',
    ' some',
    ' when',
    ' It',
    ' what',
    ' its',
    ' only',
    ' how',
    ' most',
    ' This',
    ' these',
    ' very',
    ' much',
    ' those',
    ' such',
    ' But',
    ' You',
    ' If',
    ' take',
    'It',
    ' As',
    ' For',
    ' They',
    'the',
    ' That',
    'But',
    ' When',
    ' To',
    'As',
    ' almost',
    ' With',
    ' However',
    'When',
    ' These',
    'That',
    'To',
    ' By',
    ' takes',
    ' While',
    ' whose',
    'With',
    ' Of',
    ' THE',
    ' From',
    ' aren',
    'While',
    ' perhaps',
    'By',
    ' whatever',
    'with',
    ' Like',
    'However',
    'Of',
    ' Their',
    ' Those',
    ' Its',
    ' Thus',
    ' Such',
    'Those',
    ' Much',
    ' Between',
    'Their',
    ' meanwhile',
    ' nevertheless',
    'Its',
    ' at', ' of', 'to', ' now', "'s", 'The', ".", ",", "?", "!", " '", "'", ' "', '"', 'you', 'your', ' you', ' your', ' once', ' and', ' all', 'Now', ' He',
    ' her', ' my', ' your',
    '<|endoftext|>',
    # Need to go through these words and add more to them, I suspect this list is minimal
]

def get_first_letter(tok: str):
    assert isinstance(tok, str)
    if tok[0] != " " or len(tok) == 1:
        return tok[0]
    return tok[1]

def begins_with_capital_letter(tok: str):
    str_tok = get_first_letter(tok)
    return ord("A") <= ord(str_tok) <= ord("Z")

def get_proper_nouns(model):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    proper_nouns = torch.zeros(model.cfg.d_vocab).bool()
    print("Computing proper nouns...")
    for i in tqdm(range(model.cfg.d_vocab)):
        s = model.to_single_str_token(i)
        tokens = nltk.word_tokenize(s)
        tagged = nltk.pos_tag(tokens)
        proper_nouns = [word for word, pos in tagged if pos == 'NNP']
        if len(proper_nouns)>0: 
            proper_nouns[i] = True
    if "gpt" in model.cfg.model_name:
        proper_nouns[50256] = False # BOS is not a proper noun ... hacky
    return proper_nouns


def get_performance_recovered(cspa_results: Dict[str, t.Tensor], metric: str = "kl_div_cspa_to_orig", verbose=False):
    '''Calculate the performance recovered with some metric'''

    numerator = cspa_results[metric]
    if "loss" in metric:
        numerator -= cspa_results["loss"]
    if verbose: print(f"numerator = {numerator.mean().item():.4f}")
    denominator = cspa_results[metric.replace("cspa", "ablated")]
    if "loss" in metric:
        numerator -= cspa_results["loss"]

    assert numerator.shape==denominator.shape, f"numerator.shape: {numerator.shape}, denominator.shape: {denominator.shape}"
    return 1 - numerator.mean().item() / denominator.mean().item()


def rescale_to_retain_bos(
    att_probs: Float[t.Tensor, "batch seqQ seqK"],
    old_bos_probs: Float[t.Tensor, "batch seqQ"]
):
    new_att_probs = att_probs.clone()  # Kinda scared of modifying this in place
    rest_of_attention_probs = new_att_probs[:, 1:, 1:].sum(dim=-1)
    scale_factor = (-old_bos_probs[:, 1:] + 1.0) / rest_of_attention_probs  # scale_factor * (sum of non BOS probs) + new BOS probs = 1.0
    new_att_probs[:, 1:, 1:] *= scale_factor.unsqueeze(-1)
    new_att_probs[:, 1:, 0] = old_bos_probs[:, 1:]
    return new_att_probs

def gram_schmidt(basis: Float[Tensor, "... d num"], device=None) -> Float[Tensor, "... d num"]:
    '''
    Performs Gram-Schmidt orthonormalization on a batch of vectors, returning a basis.

    `basis` is a batch of vectors. If it was 2D, then it would be `num` vectors each with length
    `d`, and I'd want a basis for these vectors in d-dimensional space. If it has more dimensions 
    at the start, then I want to do the same thing for all of them (i.e. get multiple independent
    bases).

    If the vectors aren't linearly independent, then some of the basis vectors will be zero (this is
    so we can still batch our projections, even if the subspace rank for each individual projection
    is not equal.

    If device is not None, do computation on the device (to try and reduce memory strain)
    '''
    # Make a copy of the vectors

    if device is not None:
        original_device = basis.device
        basis = basis.to(device).clone()

    else:
        basis = basis.clone()

    num_vectors = basis.shape[-1]
    
    # Iterate over each vector in the batch, starting from the zeroth
    for i in range(num_vectors):

        # Project the i-th vector onto the space orthogonal to the previous ones
        for j in range(i):
            projection = einops.einsum(basis[..., i], basis[..., j], "... d, ... d -> ...")
            basis[..., i] = basis[..., i] - einops.einsum(projection, basis[..., j], "..., ... d -> ... d")
        
        # Normalize this vector (we can set it to zero if it's not linearly independent)
        basis_vec_norm = basis[..., i].norm(dim=-1, keepdim=True)
        basis[..., i] = t.where(
            basis_vec_norm > 1e-6,
            basis[..., i] / basis_vec_norm,
            0.0,
        )
        
    if device is not None:
        basis = basis.to(original_device)

    return basis


def project(
    vectors: Float[Tensor, "... d"],
    proj_directions: Float[Tensor, "... d num"],
    only_keep: Optional[Literal["pos", "neg"]] = None,
    gs: bool = True,
    return_coeffs: bool = False,
    device = None,
):
    '''
    `vectors` is a batch of vectors, with last dimension `d` and all earlier dimensions as batch dims.

    `proj_directions` is either the same shape as `vectors`, or has an extra dim at the end.

    If they have the same shape, we project each vector in `vectors` onto the corresponding direction
    in `proj_directions`. If `proj_directions` has an extra dim, then the last dimension is another 
    batch dim, i.e. we're projecting each vector onto a subspace rather than a single vector.
    
    If device is not None, do Gram-Schmidt on that device to try and reduce memory strain
    '''

    # If we're only projecting onto one direction, add a dim at the end (for consistency)
    if proj_directions.shape == vectors.shape:
        proj_directions = proj_directions.unsqueeze(-1)
    # Check shapes
    assert proj_directions.shape[:-1] == vectors.shape
    assert not((proj_directions.shape[-1] > 20) and gs), "Shouldn't do too many vectors, GS orth might be computationally heavy I think"

    # We might want to have done G-S orthonormalization first
    proj_directions_basis = gram_schmidt(proj_directions, device=device) if gs else proj_directions

    components_in_proj_dir = einops.einsum(
        vectors, proj_directions_basis,
        "... d, ... d num -> ... num"
    )
    if return_coeffs: return components_in_proj_dir
    
    if only_keep == "pos":
        components_in_proj_dir = t.where(components_in_proj_dir > 0, components_in_proj_dir, 0.0)
    elif only_keep == "neg":
        components_in_proj_dir = t.where(components_in_proj_dir < 0, components_in_proj_dir, 0.0)

    vectors_projected = einops.einsum(
        components_in_proj_dir,
        proj_directions_basis,
        "... num, ... d num -> ... d"
    )

    return vectors_projected



def model_fwd_pass_from_resid_pre(
    model: HookedTransformer, 
    resid_pre: Float[Tensor, "batch seq d_model"],
    layer: int,
    return_type: Literal["logits", "resid_post", "loss"] = "logits",
    toks: Optional[Int[Tensor, "batch seq"]] = None,
) -> Float[Tensor, "batch seq d_vocab"]:
    '''
    Performs a forward pass starting from an intermediate point.

    For instance, if layer=10, this will apply the TransformerBlocks from
    layers 10, 11 respectively, then ln_final and unembed.

    Also, if return_resid_post = True, then it just returns the final value
    of the residual stream, i.e. omitting ln_final and unembed.
    '''
    assert return_type in ["logits", "resid_post", "loss"]

    resid = resid_pre
    for i in range(layer, model.cfg.n_layers):
        resid = model.blocks[i](resid)
    
    if (return_type == "resid_post"): return resid

    resid_scaled = model.ln_final(resid)
    logits = model.unembed(resid_scaled)

    if return_type == "logits":
        return logits
    elif return_type == "resid_post":
        return resid
    elif return_type == "loss":
        assert toks is not None
        return model.loss_fn(logits, toks, per_token=True)

@dataclass
class QKProjectionConfig:
    q_direction: Optional[str] = None
    actually_project: bool = True # Very rarely, we may want to just set the Q input to the q_direction, not project onto it
    k_direction: Optional[str] = None
    q_input_multiplier: float = 1.0 # Make this >1.0 as a hack --- the unembedding is ~1/3 of total attention score so this can be pretty helpful
    # When calculating softmax over attention scores, should we use the denominator from the original forward pass? Note this means attention no longer sums to 1.0!
    use_same_scaling: bool = False 
    # Note: projection_directions has a cursed type as we use the first for earlier_heads, and second for use_copying_as_query
    projection_directions: Optional[Union[List[Float[torch.Tensor, "batch seq d_model"]], Float[torch.Tensor, "batch seqQ seqK d_model"]]] = None
    mantain_bos_attention: bool = True
    model: Optional[HookedTransformer] = None # model is required for precomputation
    heads: Optional[List[Tuple[int, int]]] = field(default_factory=lambda: [ # Heads also required for precomputation. SVD is probably O(heads^2) and so let's do it for a subset for now
        (8, 1), 
        (8, 8),
        (9, 2),
        (9, 6),
        (9, 9),
        (9, 3), 
        (8, 6),
    ])
    W_EE: Optional[torch.Tensor] = None
    use_semantic: bool = False
    save_scores: bool = False
    scores: Optional[Float[torch.Tensor, "batch seqQ seqK"]] = None
    save_scaled_resid_pre: bool = False
    scaled_resid_pre: Optional[Float[torch.Tensor, "batch seq d_model"]] = None
    swap_model_and_our_max_attention: bool = False # Testing whether we are wrong because we just get our top attention wrong. Let's hope so!
    swap_model_and_our_max_scores: bool = False # Same for scores
    query_bias_multiplier: float = 1.0 # Do we want to multiply query bias up???
    key_bias_multiplier: float = 1.0 # Do we want to multiply key bias up???
    capital_adder: Optional[float] = None # Do we want to add attention scores on capital letters? (Note that timesing didn't work)
    proper_noun_adder: Optional[float] = None # Do we want to add attention scores on proper nouns? (Note that timesing didn't work)
    proper_nouns: Optional[Float[torch.Tensor, "n_vocab"]] = None # Stores the token IDs of proper nouns

    save_q_remove_unembed: bool = False # Save what gets left over after doing
    q_remove_unembed: Optional[Float[torch.Tensor, "batch seq d_model"]] = None
    another_direction: Optional[Float[torch.Tensor, "d_model"]] = None # If we want to use an additional direction for the query
    save_query_input_dotter: bool = False
    query_input_dotter: Optional[Float[torch.Tensor, "batch seqK d_model"]] = None
    fix_bos_input: bool = True # Keep BOS attention score the same

    def __post_init__(self):
        
        assert self.q_direction == "layer9_heads" or self.actually_project, "If we're not projecting, we need to be using layer9_heads"

        if self.q_direction == "earlier_heads":
            """Precompute some OV circuit for earlier heads... this apprach didn't work well I think <60% KL and other heads in the model outcompeted the L10H7"""

            W_EE = get_effective_embedding(self.model, use_codys_without_attention_changes=True)["W_E (including MLPs)"]
            assert self.model is not None, "Need to pass model to QKProjectionConfig if you want to use earlier_heads"
            assert self.heads is not None, "Need to pass heads to QKProjectionConfig if you want to use earlier_heads"

            head_projection_matrices: Dict[
                Tuple[int, int], Float["d_vocab d_model"]
            ] = {}

            W_EE = get_effective_embedding(self.model, use_codys_without_attention_changes=True)["W_E (including MLPs)"]

            for layer_idx, head_idx in self.heads:
                W_OV = self.model.W_V[layer_idx, head_idx] @ self.model.W_O[layer_idx, head_idx]
                queryside_matrix = W_EE @ W_OV
                head_projection_matrices[(layer_idx, head_idx)] = queryside_matrix
                del queryside_matrix
                del W_OV 
                gc.collect()
                torch.cuda.empty_cache()

            self.projection_directions = head_projection_matrices

        if self.k_direction == "effective_embedding":
            W_EE = get_effective_embedding(self.model, use_codys_without_attention_changes=True)["W_E (including MLPs)"]
            self.W_EE = W_EE

        if self.proper_noun_adder is not None:
            warnings.warn("Proper noun adder seems to miss lots of things that the model considers proper nounds, e.g Vog, Qualcomm, Aurora")
            print("Downloading NLTK...")
            self.proper_nouns = get_proper_nouns(
                self.model,
            )


    def compute_copying_as_query_directions(self, cache: ActivationCache, negative_head):
        """This was another idea that didn't really work, recovering pretty bad KL"""

        # Let's compute what the directions need be from the cache
        assert self.q_direction.startswith("use_copying_as_query")
        if self.q_direction == "use_copying_as_query_testing":
            same_place_hook_names = ["hook_pos_embed", "hook_embed"] + [utils.get_act_name("mlp_out", layer) for layer in range(negative_head[0])]
            residual = sum([cache[name] for name in same_place_hook_names]) # [batch seq d_model]

            for layer in range(negative_head[0]):
                v = cache[utils.get_act_name("v", layer)] # [batch seq d_model]
                pattern = cache[utils.get_act_name("pattern", layer)] # [batch head seqQ seqK]
                results = einops.einsum(
                    v,
                    pattern,
                    self.model.W_O[layer],
                    "batch seqK head d_head, batch head seqQ seqK, head d_head d_model -> batch seqQ d_model", # batch query_pos head_index d_head
                )
                residual += results
                residual += self.model.b_O[layer]

            torch.testing.assert_close(
                residual,
                cache["blocks.10.hook_resid_pre"],
                atol=1e-3,
                rtol=1e-3,
            )
            print("Passed!")
        
        batch_size, seq_len, d_model = cache[f"blocks.{negative_head[0]}.hook_resid_pre"].shape
        self.projection_directions = torch.zeros(batch_size, seq_len, seq_len, d_model) # [batch seqQ seqK d_model]

        original_model_device = self.model.cfg.device
        self.model = self.model.to("cpu")

        for layer in range(negative_head[0]): # 20% not 30% when restricting to last two heads
            current_resid = einops.einsum(
                cache[utils.get_act_name("v", layer)].cpu(), # TODO see if speedup w/ CUDA possible
                cache[utils.get_act_name("pattern", layer)].cpu(),
                self.model.W_O[layer].cpu(),
                "batch seqK head d_head, batch head seqQ seqK, head d_head d_model -> batch seqQ seqK d_model",
            )
            self.projection_directions += current_resid

        self.model = self.model.to(original_model_device)

@dataclass 
class OVProjectionConfig:
    pass # Currently we only have the Unembedding projection option


def get_cspa_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_head: Tuple[int, int],
    interventions: List[str],
    qk_projection_config: Optional[QKProjectionConfig] = None,
    ov_projection_config: Optional[OVProjectionConfig] = None,
    K_unembeddings: Optional[Union[int, float]] = None,
    K_semantic: int = 10,
    global_range: Optional[Tuple[int,int]] = None,
    only_keep_negative_components: bool = False,
    semantic_dict: dict = {},
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    return_dla: bool = False,
    return_logits: bool = False,
    verbose: bool = False,
    keep_self_attn: bool = True,
    computation_device = None,
) -> Tuple[
        Dict[str, Float[Tensor, "batch seq-1"]],
        Optional[Int[Tensor, "n 4"]],
        Optional[Float[Tensor, "n"]],
        Optional[Int[Tensor, "batch seqK K_semantic"]],
        float,
    ]:
    '''
    Short explpanation of the copy-suppression preserving ablation (hereby CSPA), with
    the following arguments:

        interventions = ["qk", "ov"]
        K_semantic = 3
        K_unembeddings = 5

        For every souce token s, we pick the 3 tokens which it's most semantically
        related to - call this set S*.

        For every destination token d, we look at the union of all semantically similar
        tokens S* for the source tokens before it in context, and pick the 5 which
        are most predicted (i.e. the highest logit lens). This gives us 5 pairs (s, s*).

        For each source token s which is in one of these pairs, we take the result vector
        which is moved from s -> d, and project it onto the unembeddings of all s* which
        it appears in a pair with. If a source token doesn't appear in one of these pairs,
        we mean ablate that vector.

            Summary: information is moved from s -> d if and only if s is semantically similar 
            to some token s* which is being predicted at d, and in this case the information
            which is moved is restricted to the subspace of the unembedding of s*.

        If instead we just had interventions = ["qk"], then we'd filter for the pairs (s, s*)
        where s* is predicted at d, but we wouldn't project their output vectors onto
        unembeddings. If instead we just had ["ov"], then we'd project all vectors from s onto 
        the span of unembeddings of their s*, but we wouldn't also filter for pairs (s, s*) 
        where s* is predicted.

    
    A few notes / explanations:

        > result_mean is the vector we'll use for ablation, if supplied. It'll map e.g. 
          (10, 7) to the mean result vector for each seqpos (hopefully seqpos is larger than 
          that for toks).
        > If K_unembeddings is a float rather than an int, it's converted to an int as follows:
          ceil(K_unembeddings * destination_position).

    Return type:

        > A dictionary of the results, with "loss", "loss_ablated", and "loss_cspa", plus the
          logits & KL divergences, as keys.
        > A dict mapping (batch_idx, d) to the list of (s, s*) which we preserve in our
          ablation.
    '''

    if ov_projection_config is None and "ov" in interventions:
        ov_projection_config = OVProjectionConfig()
        interventions = interventions[:] # Otherwise successive calls to the function will also be edited
        interventions.remove("ov")

    # ====================================================================
    # ! STEP 0: Define setup vars, move things onto the correct device, get semantically similar toks
    # ====================================================================

    LAYER, HEAD = negative_head
    batch_size, seq_len = toks.shape
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()
    
    W_U = model.W_U
    FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()


    semantically_similar_toks = toks.unsqueeze(-1) # [batch seqK 1]



    # ====================================================================
    # ! STEP 1: Perform clean and ablated model forward passes (while caching clean activations, for use later)
    # ====================================================================
    
    # Get all hook names
    resid_hook_name = utils.get_act_name("resid_pre", LAYER)
    resid_final_hook_name = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    pre_result_hook_name = utils.get_act_name("result", LAYER-1)
    result_hook_name = utils.get_act_name("result", LAYER)
    v_hook_name = utils.get_act_name("v", LAYER)
    pattern_hook_name = utils.get_act_name("pattern", LAYER)
    scores_hook_name = f"blocks.{LAYER}.attn.hook_attn_scores"
    scale_hook_name = utils.get_act_name("scale", LAYER, "ln1")
    scale_final_hook_name = utils.get_act_name("scale")
    hook_names_to_cache = [pre_result_hook_name, scale_final_hook_name, scores_hook_name, v_hook_name, pattern_hook_name, scale_hook_name, resid_hook_name, resid_final_hook_name, result_hook_name]

    t_clean_and_ablated = time.time()

    # * Get clean results (also use this to get residual stream before layer 10)
    model.reset_hooks()
    logits, cache = model.run_with_cache(
        toks,
        return_type = "logits",
        names_filter = lambda name: name in hook_names_to_cache
    )
    loss = model.loss_fn(logits, toks, per_token=True)

    resid_post_final = cache[resid_final_hook_name] # [batch seqQ d_model]
    resid_pre = cache[resid_hook_name] # [batch seqK d_model]
    # TODO odd thing; sometimes it seems like scale isn't recorded as different across heads. why?
    scale = cache[scale_hook_name]
    if scale.ndim == 4:
        scale = cache[scale_hook_name][:, :, HEAD] # [batch seqK 1]
    head_result_orig = cache[result_hook_name][:, :, HEAD] # [batch seqQ d_model]
    final_scale = cache[scale_final_hook_name] # [batch seq 1]
    v = cache[v_hook_name][:, :, HEAD] # [batch seqK d_head]
    pattern = cache[pattern_hook_name][:, HEAD] # [batch seqQ seqK]      
    scores = cache[scores_hook_name][:, HEAD] # [batch seqQ seqK]
    scaled_resid_pre = cache[resid_hook_name].clone() / cache[scale_hook_name]
    pre_head_result_orig = cache[pre_result_hook_name] # [batch seq n_heads d_model]

    del cache

    # * Perform complete ablation (via a direct calculation)
    if batch_size * seq_len < 1000:
        assert result_mean is not None, "You should be using an externally supplied mean ablation vector for such a small dataset."
    if result_mean is None:
        head_result_orig_mean_ablated = einops.reduce(head_result_orig, "batch seqQ d_model -> seqQ d_model", "mean")
    else:
        head_result_orig_mean_ablated = result_mean[(LAYER, HEAD)][:seq_len]
    resid_post_final_mean_ablated = resid_post_final + (head_result_orig_mean_ablated - head_result_orig) # [batch seq d_model]
    logits_mean_ablated = (resid_post_final_mean_ablated / final_scale) @ model.W_U + model.b_U
    loss_mean_ablated = model.loss_fn(logits_mean_ablated, toks, per_token=True)
    model.reset_hooks()

    t_clean_and_ablated = time.time() - t_clean_and_ablated

    # ====================================================================
    # ! STEP 3: Run projections
    # ====================================================================



    # ====================================================================
    # ! STEP 4: Get CSPA results (this is the hard part!)
    # ====================================================================    

    # Multiply by output matrix, then by attention probabilities
    output = v @ model.W_O[LAYER, HEAD] # [batch seqK d_model]
    output_attn = einops.einsum(output, pattern, "batch seqK d_model, batch seqQ seqK -> batch seqQ seqK d_model")
    
    # We might want to use the results supplied for mean ablation

    output_attn_mean_ablated = einops.einsum(
        result_mean[(LAYER, HEAD)][:seq_len], pattern,
        "seqQ d_model, batch seqQ seqK -> batch seqQ seqK d_model"
    )

    assert ("qk" in interventions) == (K_unembeddings != 1.0), "Either do a QK intervention, or we must all unembeddings used"

    if "qk" in interventions:

        # Get the top predicted semantically similar tokens (this everything with seqQ<=seqK if we're not doing QK filtering)
        # TODO probably refactor this because I expect us to rarely be needing this full function now, it's mostly a no op
        t0 = time.time()
        # Get the unembeddings we'll be projecting onto (also get the dict of (s, s*) pairs and store in context)
        # Most of the elements in `semantically_similar_unembeddings` will be zero
        semantically_similar_unembeddings, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, top_K_and_Ksem_mask = get_top_predicted_semantically_similar_tokens(
            toks=toks,
            resid_pre=resid_pre,
            semantically_similar_toks=semantically_similar_toks,
            K_unembeddings=K_unembeddings,
            function_toks=FUNCTION_TOKS,
            model=model,
            final_scale=final_scale,
            keep_self_attn=keep_self_attn,
        )
        if verbose: print(f"Fraction of unembeddings we keep = {(semantically_similar_unembeddings.abs() > 1e-6).float().mean():.4f}")
        time_for_sstar = time.time() - t0

        if ov_projection_config is not None:
            # We project the output onto the unembeddings we got from the code above (which will either be all unembeddings,
            # or those which were filtered for being predicted on the destination side).
            if only_keep_negative_components:
                assert K_semantic == 1, "Can't use semantic similarity if we're only keeping negative components."
            output_attn_cspa = project(
                vectors = output_attn - output_attn_mean_ablated,
                proj_directions = semantically_similar_unembeddings,
                only_keep = "neg" if only_keep_negative_components else None, 
                device = computation_device,
            ) + output_attn_mean_ablated


        else:
            # In this case, we assume we are filtering for QK (cause we're doing at least one). We want to set the output to be the mean-ablated
            # output at all source positions which are not in the top predicted semantically similar tokens.
            def any_reduction(tensor: Tensor, dims: tuple):
                assert dims == (3,)
                return tensor.any(dims[0])
            top_K_and_Ksem_mask_any = einops.reduce(
                top_K_and_Ksem_mask, 
                "batch seqQ seqK K_semantic -> batch seqQ seqK",
                reduction = any_reduction # dims will be supplied as (3,) I think
            )
            output_attn_cspa = t.where(
                top_K_and_Ksem_mask_any.unsqueeze(-1),
                output_attn,
                output_attn_mean_ablated,
            )

    else:
        output_attn_cspa = output_attn

    # Sum over key-side vectors to get new head result
    # ? (don't override the BOS token attention, because it's more appropriate to preserve this information I think)
    # output_attn_cspa[:, :, 0, :] = output_attn[:, :, 0, :]
    head_result_cspa = einops.reduce(output_attn_cspa, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")


    # Get DLA, logits, and loss
    dla_cspa = ((head_result_cspa - head_result_orig_mean_ablated) / final_scale) @ model.W_U
    resid_post_final_cspa = resid_post_final + (head_result_cspa - head_result_orig) # [batch seq d_model]
    logits_cspa = (resid_post_final_cspa / final_scale) @ model.W_U + model.b_U
    loss_cspa = model.loss_fn(logits_cspa, toks, per_token=True)

    gc.collect()
    t.cuda.empty_cache()

    cspa_results = {
        "loss": loss.cpu(),
        "loss_cspa": loss_cspa.cpu(),
        "loss_ablated": loss_mean_ablated.cpu(),
        # "dla": dla_cspa,
        # "logits": logits_cspa,
        # "logits_orig": logits,
        # "logits_ablated": logits_mean_ablated,
        "kl_div_ablated_to_orig": kl_div(logits.cpu(), logits_mean_ablated.cpu()),
        "kl_div_cspa_to_orig": kl_div(logits.cpu(), logits_cspa.cpu()),
        "pattern": pattern.detach().cpu(),
        "normal_scores": scores.detach().cpu(),
    }
    if return_dla: 
        cspa_results["dla"] = dla_cspa
    if return_logits:
        cspa_results["logits_cspa"] = logits_cspa
        cspa_results["logits_orig"] = logits
        cspa_results["logits_ablated"] = logits_mean_ablated

    if "qk" in interventions:
        return cspa_results, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, semantically_similar_toks, time_for_sstar 

    else:
        return cspa_results, None, None, None, 0.0


def get_cspa_results_batched(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    max_batch_size: int,    
    negative_head: Tuple[int, int],
    interventions: List[str],
    qk_projection_config: Optional[QKProjectionConfig] = None,
    ov_projection_config: Optional[OVProjectionConfig] = None,
    K_unembeddings: Optional[Union[int, float]] = None,
    K_semantic: int = 10,
    only_keep_negative_components: bool = False,
    semantic_dict: dict = {},
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    use_cuda: bool = False,
    verbose: bool = False,
    compute_s_sstar_dict: bool = False,
    return_dla: bool = False,
    return_logits: bool = False,
    keep_self_attn: bool = True,
    computation_device = None, 
    do_running_updates: bool = False,
) -> Dict[str, Float[Tensor, "batch seq-1"]]:
    '''
    Gets results from CSPA, by splitting the tokens along batch dimension and running it several 
    times. This allows me to use batch sizes of 1000+ without getting CUDA errors.

    See the `get_cspa_results` docstring for more info.
    '''
    if "ov" in interventions:

        warnings.warn("WARNING: since the OV move is really a projection, we now use OVProjectionConfig. Please add ov_projection_config argument in future rather than using the interventions list")        
        ov_projection_config = OVProjectionConfig()

        interventions = interventions[:] # Otherwise successive calls to the function will also be edited
        interventions.remove("ov")

    batch_size, seq_len = toks.shape
    chunks = toks.shape[0] // max_batch_size

    device = t.device("cuda" if use_cuda else "cpu")
    result_mean = {k: v.to(device) for k, v in result_mean.items()}

    orig_model_device = str(next(iter(model.parameters())).device)
    orig_toks_device = str(toks.device)
    target_device = "cuda" if use_cuda else "cpu"
    if not devices_are_equal(orig_model_device, target_device):
        model = model.to(target_device)
    if not devices_are_equal(orig_toks_device, target_device):
        toks = toks.to(target_device)

    CSPA_RESULTS = {}
    TOP_K_AND_KSEM_PER_DEST_TOKEN = t.empty((0, 4), dtype=t.long, device=target_device) if "qk" in interventions else None
    LOGIT_LENS_FOR_TOP_K_KSEM = t.empty((0,), dtype=t.float, device=target_device) if "qk" in interventions else None
    SEMANTICALLY_SIMILAR_TOKS = t.empty((0, seq_len, K_semantic), dtype=t.long, device=target_device) if "qk" in interventions else None

    for i, _toks in enumerate(t.chunk(toks, chunks=chunks)):
        if verbose and i == 0:
            bar = tqdm(total=chunks, desc=f"Batch {i+1}/{chunks}, shape {_toks.shape}")
        
        # Get new results
        t_get = time.time()

        current_indices_lower = max_batch_size * i
        current_indices_upper = current_indices_lower + _toks.shape[0]

        cspa_results, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, semantically_similar_toks, t_sstar = get_cspa_results(
            model = model,
            toks = _toks,
            global_range = (current_indices_lower, current_indices_upper),
            negative_head = negative_head,
            interventions = interventions,
            qk_projection_config=qk_projection_config,
            ov_projection_config=ov_projection_config,
            K_unembeddings = K_unembeddings,
            K_semantic = K_semantic,
            only_keep_negative_components = only_keep_negative_components,
            semantic_dict = semantic_dict,
            result_mean = result_mean,
            keep_self_attn = keep_self_attn,
            return_dla = return_dla,
            return_logits = return_logits,
            computation_device = computation_device,
        )
        t_get = time.time() - t_get

        # Add them to all accumulated results
        t_agg = time.time()
        CSPA_RESULTS = concat_dicts(CSPA_RESULTS, cspa_results)

        if "qk" in interventions:
            TOP_K_AND_KSEM_PER_DEST_TOKEN = t.cat([TOP_K_AND_KSEM_PER_DEST_TOKEN, top_K_and_Ksem_per_dest_token], dim=0)
            LOGIT_LENS_FOR_TOP_K_KSEM = t.cat([LOGIT_LENS_FOR_TOP_K_KSEM, logit_lens_for_top_K_Ksem], dim=0)
            SEMANTICALLY_SIMILAR_TOKS = t.cat([SEMANTICALLY_SIMILAR_TOKS, semantically_similar_toks], dim=0)
            del top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, semantically_similar_toks
        
        del cspa_results
        t.cuda.empty_cache()
        t_agg = time.time() - t_agg

        if verbose:
            bar.update()
            bar.set_description(f"Batch {i+1}/{chunks}, shape = {tuple(_toks.shape)}, times = [get = {t_get-t_sstar:.2f}, s* = {t_sstar:.2f}, aggregate = {t_agg:.2f}]")

    to_return = CSPA_RESULTS

    if not devices_are_equal(orig_model_device, target_device):
        model = model.to(orig_model_device)
    if not devices_are_equal(orig_toks_device, target_device):
        toks = toks.to(orig_toks_device)
    
    return to_return



def get_top_predicted_semantically_similar_tokens(
    toks: Int[Tensor, "batch seq"],
    resid_pre: Float[Tensor, "batch seqK d_model"],
    semantically_similar_toks: Int[Tensor, "batch seq K_semantic"],
    K_unembeddings: Optional[Union[int, float]],
    function_toks: Int[Tensor, "tok"],
    model: HookedTransformer,
    final_scale: Optional[Float[Tensor, "batch seqQ 1"]] = None,
    keep_self_attn: bool = True,
) -> Tuple[
    Float[Tensor, "batch seqQ seqK d_model K_semantic"], 
    Int[Tensor, "n 4"], 
    Float[Tensor, "n"], 
    Bool[Tensor, "batch seqQ seqK K_semantic"],
]:
    '''
    Arguments:

        toks: [batch seq]
            The source tokens.
        
        resid_pre: [batch seqK d_model]
            The residual stream before the head, which we'll use to figure out which tokens in `toks`
            are predicted with highest probability.

        semantically_similar_toks: [batch seqK K_semantic]
            Semantically similar tokens for each source token.
        
        function_toks: [tok]
            1D array of function tokens (we filter for pairs (s, s*) where s is not a function token).

    Returns:

        semantically_similar_unembeddings: [batch seqQ seqK d_model K_semantic]
            The unembeddings of the semantically similar tokens, with all the vectors except the
            ones we're actually using set to zero.
        
        top_K_and_Ksem_per_dest_token: [n 4]
            The i-th row is the (b, sQ, sK, K_s) indices of the i-th top predicted semantically similar
            token.

        logit_lens_for_top_K_and_Ksem_per_dest_token: [n]
            The i-th element is the logits for the corresponding prediction in the previous tensor. This
            is used to make sure that the eventual s_star dictionary we create is sorted correctly.
        
        mask: [batch seqQ seqK K_semantic]
            The mask which we'll be applying once we project the unembeddings.
    '''
    semantically_similar_unembeddings = model.W_U.T[semantically_similar_toks].transpose(-1, -2) # [batch seqK d_model K_semantic]
    batch_size, seq_len = toks.shape
    
    # If K_unembeddings is None, then we have no restrictions, and we use all the top K_semantic toks for each seqK
    if K_unembeddings is None:
        semantically_similar_unembeddings = einops.repeat(
            semantically_similar_unembeddings,
            "batch seqK d_model K_semantic -> batch seqQ seqK d_model K_semantic",
            seqQ = seq_len,
        )

    # Else, we filter down the sem similar unembeddings to only those that are predicted
    else:
        # Get logit lens from current value of residual stream
        # logit_lens[b, sQ, sK, K_s] = prediction at destination token (b, sQ), for the K_s'th semantically similar token to source token (b, sK)
        resid_pre_scaled = resid_pre if (final_scale is None) else resid_pre / final_scale
        logit_lens = einops.einsum(
            resid_pre_scaled, semantically_similar_unembeddings,
            "batch seqQ d_model, batch seqK d_model K_semantic -> batch seqQ seqK K_semantic",
        )

        # * MASK: make sure function words are never the source token (because we've observed that the QK circuit has managed to learn this)
        is_fn_word = (toks[:, :, None] == function_toks).any(dim=-1) # [batch seqK]
        logit_lens = t.where(einops.repeat(is_fn_word, "batch seqK -> batch 1 seqK 1"), -1e9, logit_lens)
        # * MASK: apply causal mask (this might be strict if keep_self_attn is False)
        seqQ_idx = einops.repeat(t.arange(seq_len), "seqQ -> 1 seqQ 1 1").to(logit_lens.device)
        seqK_idx = einops.repeat(t.arange(seq_len), "seqK -> 1 1 seqK 1").to(logit_lens.device)
        causal_mask = (seqQ_idx < seqK_idx) if keep_self_attn else (seqQ_idx <= seqK_idx)
        logit_lens = t.where(causal_mask, -1e9, logit_lens)
        # * MASK: each source token should only be counted at its first instance
        # Note, we apply this mask to get our topk values (so no repetitions), but we don't want to apply it when we're choosing pairs to keep
        first_occurrence_mask = einops.repeat(first_occurrence_2d(toks), "batch seqK -> batch 1 seqK 1")
        logit_lens_for_topk = t.where(~first_occurrence_mask, -1e9, logit_lens)

        # Get the top predicted src-semantic-neighbours s* for each destination token d
        # (this might be different for each destination posn, if K_unembeddings is a float)
        if isinstance(K_unembeddings, int):
            top_K_and_Ksem_per_dest_token_values = logit_lens_for_topk.flatten(-2, -1).topk(K_unembeddings, dim=-1).values[..., [[-1]]] # [batch seqQ 1 1]
        else:
            top_K_and_Ksem_per_dest_token_values = t.full((batch_size, seq_len, 1, 1), fill_value=-float("inf"), device=logit_lens.device)
            for dest_posn in range(seq_len):
                K_u_dest = math.ceil(K_unembeddings * (dest_posn + 1))
                top_K_and_Ksem_per_dest_token_values[:, dest_posn] = logit_lens_for_topk[:, dest_posn].flatten(-2, -1).topk(K_u_dest, dim=-1).values[..., [[-1]]]
        
        # Later we'll be computing the list of (s, s*) for analysis afterwards
        top_K_and_Ksem_mask = (logit_lens + 1e-6 >= top_K_and_Ksem_per_dest_token_values) # [batch seqQ seqK K_s]
        top_K_and_Ksem_per_dest_token = t.nonzero(top_K_and_Ksem_mask) # [n 4 = (batch, seqQ, seqK, K_s)], n >= batch * seqQ * K_u (more if we're double-counting source tokens)
        b, sQ, sK, K_s = top_K_and_Ksem_per_dest_token.T.tolist()
        logit_lens_for_top_K_and_Ksem_per_dest_token = logit_lens[b, sQ, sK, K_s]

        # Use this boolean mask to set some of the unembedding vectors to zero
        unembeddings = einops.repeat(semantically_similar_unembeddings, "batch seqK d_model K_semantic -> batch 1 seqK d_model K_semantic")
        top_K_and_Ksem_mask_repeated = einops.repeat(top_K_and_Ksem_mask, "batch seqQ seqK K_semantic -> batch seqQ seqK 1 K_semantic")
        semantically_similar_unembeddings = unembeddings * top_K_and_Ksem_mask_repeated.float()

    return semantically_similar_unembeddings, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_and_Ksem_per_dest_token, top_K_and_Ksem_mask




def convert_top_K_and_Ksem_to_dict(
    top_K_and_Ksem_per_dest_token: Int[Tensor, "n 4"], # each row is (batch, seqQ, seqK, K_s)
    logit_lens_for_top_K_Ksem: Float[Tensor, "n"], # each element is logits (we keep it for sorting purposes)
    toks: Int[Tensor, "batch seq"],
    semantically_similar_toks: Int[Tensor, "batch seq s_K"],
    model: HookedTransformer,
):
    '''
    Making this function because it's more efficient to do this all at once (model.to_str_tokens is slow!).
    '''
    s_sstar_pairs = defaultdict(list)

    # Get all batch indices, dest pos indices, src pos indices, and semantically similar indices
    b, sQ, sK, K_s = top_K_and_Ksem_per_dest_token.T.tolist()

    # Get the string representations of (s, s*) that we'll be using in the html viz (s comes with its position)
    s_str_toks = model.to_str_tokens(toks[b, sK], prepend_bos=False)
    s_repr = [f"[{s_posn}] {repr(s_str_tok)}" for s_posn, s_str_tok in zip(sK, s_str_toks)]
    sstar_str_toks = model.to_str_tokens(semantically_similar_toks[b, sK, K_s], prepend_bos=False)
    sstar_repr = [repr(sstar_str_tok) for sstar_str_tok in sstar_str_toks]

    # Add them all to our dict
    for _b, _sQ, _s, _sstar, _LL in zip(b, sQ, s_repr, sstar_repr, logit_lens_for_top_K_Ksem):
        s_sstar_pairs[(_b, _sQ)].append((_LL, _s, _sstar))
    
    # Make sure we rank order the entries in each dictionary by how much they're being predicted
    for (b, sQ), s_star_list in s_sstar_pairs.items():
        s_sstar_pairs[(b, sQ)] = sorted(s_star_list, key = lambda x: x[0], reverse=True)
    
    return s_sstar_pairs


def get_result_mean(
    head_list: Optional[List[Tuple[int, int]]],
    toks: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    minibatch_size: int = 10,
    keep_seq_dim: bool = True,
    include_mlps: bool = False,
    verbose: bool = False,
) -> Dict[Tuple[int, int], Float[Tensor, "*seq d_model"]]:
    '''
    Returns the mean result vector for a given set of inputs. 

    Useful for doing mean ablation, because we need the mean over the entire batch before we run any forward passes.
    '''
    # Get list of heads to iterate through, and list of all layers we need
    batch_size, seq_len = toks.shape
    if head_list is None:
        head_list = list(itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads)))
    layers = list(set([head[0] for head in head_list]))
    
    # Get dict to store results in: all heads, and possibly MLPs
    result_mean = {head: t.zeros((seq_len, model.cfg.d_model)).to(toks.device) for head in head_list}
    if include_mlps:
        for layer in layers: result_mean[layer] = t.zeros((seq_len, model.cfg.d_model)).to(toks.device)

    # Get tokens to iterate through
    toks_chunked = toks.split(minibatch_size, dim=0)
    toks_chunked_wrapper: Iterable[Tensor] = tqdm(toks_chunked) if verbose else toks_chunked
    num_seq_seen = 0

    # Get names of activations we want to cache
    hook_names = [utils.get_act_name("result", layer) for layer in layers]
    if include_mlps: hook_names += [utils.get_act_name("mlp_out", layer) for layer in layers]

    # Go through the iterator, and cache them
    for _toks in toks_chunked_wrapper:
        batch_size, seq_len = _toks.shape
        
        # Run with cache, getting all the activations we need
        _, cache = model.run_with_cache(
            _toks,
            return_type=None,
            names_filter=lambda name: name in hook_names
        )

        # Get mean for every head
        for layer, head in head_list:
            new_result_mean = cache["result", layer][:, :, head].mean(0) # [seq d_model]
            # Update formula for mean, e.g. R -> R first time, then R -> (R+R_new)/2, then R -> (2*R+R_new)/3, etc
            result_mean[(layer, head)] = (num_seq_seen * result_mean[(layer, head)] + batch_size * new_result_mean) / (num_seq_seen + batch_size)

        # If required, also get mean for MLP
        if include_mlps:
            new_mlp_out_mean = cache["mlp_out", layer].mean(0) # [seq d_model]
            result_mean[layer] = (num_seq_seen * result_mean[layer] + batch_size * new_mlp_out_mean) / (num_seq_seen + batch_size)

        num_seq_seen += batch_size

    # Possibly take mean over sequence dimension
    if not keep_seq_dim: result_mean = {k: v.mean(0) for k, v in result_mean.items()}
    
    return result_mean
