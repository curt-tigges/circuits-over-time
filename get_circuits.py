#%%
from functools import partial

from transformer_lens import HookedTransformer
import torch
from torch.utils.data import Dataset, DataLoader

from edge_attribution_patching.graph import  Graph
from edge_attribution_patching.attribute_mem import attribute 
from edge_attribution_patching.evaluate_graph import evaluate_graph, evaluate_baseline
from edge_attribution_patching.utils import kl_div

from utils.model_utils import load_model
from utils.data_utils import UniversalPatchingDataset
from utils.metrics import (
    CircuitMetric,
    compute_logit_diff,
    compute_probability_diff,
)
#%%    
def collate_fn(xs):
    toks, flipped_toks, answer_toks, positions, flags_tensor = zip(*xs)
    toks = torch.stack(toks)
    flipped_toks = torch.stack(flipped_toks)
    answer_toks = torch.stack(answer_toks)
    positions = torch.stack(positions)
    batch = {'toks': toks, 'flipped_toks': flipped_toks, 'answer_toks': answer_toks}
    if not torch.all(positions == -1):
        batch['positions'] = positions 
    if not flags_tensor[0] is None:
        batch["flags_tensor"] = torch.stack(flags_tensor)

    return batch


def get_data_and_metrics(
        model: HookedTransformer,
        task_name: str,
        eap: bool=True,
    ):
    assert task_name in ["ioi", "greater_than", "sentiment_cont", "sentiment_class", "mood_sentiment"]

    if task_name == "ioi":
        ds = UniversalPatchingDataset.from_ioi(model, 70)
        logit_diff_metric = partial(compute_logit_diff,mode='simple')
        metric = CircuitMetric("logit_diff", logit_diff_metric, eap = eap)

    elif task_name == "greater_than":
        # Get data
        ds = UniversalPatchingDataset.from_greater_than(model, 200)
        prob_diff_metric = partial(
            compute_probability_diff, 
            mode="group_sum"
        )
        metric = CircuitMetric("prob_diff", prob_diff_metric, eap = eap)

    elif task_name == "sentiment_cont":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "cont")
        logit_diff_metric = partial(compute_logit_diff, mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric, eap = eap)

    elif task_name == "sentiment_class":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "class")
        logit_diff_metric = partial(compute_logit_diff,  mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric, eap = eap)

    return ds, metric
#%%
batch_size = 8
model_name = 'pythia-160m'
model_tl_name = model_name

model_full_name = f"EleutherAI/{model_name}"
model_tl_full_name = f"EleutherAI/{model_tl_name}"

cache_dir = "~/.cache/huggingface/transformers"

model = HookedTransformer.from_pretrained(model_name,center_writing_weights=False,
    center_unembed=False,
    fold_ln=False,
    device='cuda',
)
# %%
task = 'ioi'
ds, metric = get_data_and_metrics(model, task, eap=True)
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
graph = Graph.from_model(model)
#%%
dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
#%%
baseline = evaluate_baseline(model, dataloader, metric).mean()
print(baseline)
# %%
attribute(model, graph, dataloader, partial(metric, loss=True), integrated_gradients=30)
# %%
graph.apply_greedy(400)
graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
results = evaluate_graph(model, graph, dataloader, metric).mean()
print(results)
graph.to_json(f'graphs/{task}.json')
# %%
gz = graph.to_graphviz()
gz.draw(f'images/{task}.png', prog='dot')
# %%
