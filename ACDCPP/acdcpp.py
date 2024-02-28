#%%
from pathlib import Path 

import numpy as np
import torch
from transformer_lens import HookedTransformer
from einops import einsum, rearrange
import matplotlib.pyplot as plt
from ACDCPP.graph import Graph, InputNode, LogitNode, AttentionNode, MLPNode
from ACDCPP.attribute_vectorized import attribute_vectorized 
from ACDCPP.evaluate_graph import evaluate_graph

#%%

def batch(iterable, n:int=1):
   current_batch = []
   for item in iterable:
       current_batch.append(item)
       if len(current_batch) == n:
           yield current_batch
           current_batch = []
   if current_batch:
       yield current_batch


def get_acdcpp_results(model, clean_data, corrupted_data, batch_size, t, metric):
    #%%
    print("start get_acdcpp_results")
    clean = list(batch(clean_data.toks, batch_size))
    corrupted = list(batch(corrupted_data.toks, batch_size))
    io_tokenIDs = list(batch(clean_data.io_tokenIDs, batch_size))
    s_tokenIDs = list(batch(clean_data.s_tokenIDs, batch_size))
    word_idx = list(batch(clean_data.word_idx['end'], batch_size))
    answers = torch.stack((torch.tensor(clean_data.io_tokenIDs), torch.tensor(clean_data.s_tokenIDs)), dim = -1)

    # %%
    # Instantiate a graph with a model
    g = Graph.from_model(model)
    # Attribute using the model, graph, clean / corrupted data (as lists of lists of strs), your metric, and your labels (batched)
    attribute_vectorized(model, g, clean, corrupted, metric, io_tokenIDs, s_tokenIDs, word_idx)
    # Apply a threshold
    g.apply_threshold(t, absolute=True)
    g.prune_dead_nodes(prune_childless=True, prune_parentless=False)
    #performance = evaluate_graph(model, g, clean_data, corrupted_data, answers, metric)

    return g, g.get_nodes(), g.get_edges(), g.get_logits()

'''
gz = g.to_graphviz()
gz.draw('graph.png', prog='dot')
'''
