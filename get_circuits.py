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
class EAPDataset(Dataset):
    def __init__(self, ds: UniversalPatchingDataset):
        self.ds = ds
    def __len__(self):
        return len(self.ds.toks)
    def __getitem__(self, idx):
        position = None if self.ds.positions is None else self.ds.positions[idx]
        group_flag = None if self.ds.group_flags is None else self.ds.group_flags[idx]
        return self.ds.toks[idx], self.ds.flipped_toks[idx], self.ds.answer_toks[idx], position, group_flag
    
def collate_fn(xs):
    toks, flipped_toks, answer_toks, positions, flags_tensor = zip(*xs)
    toks = torch.stack(toks)
    flipped_toks = torch.stack(flipped_toks)
    answer_toks = torch.stack(answer_toks)
    positions = torch.stack(positions)
    flags_tensor = None if flags_tensor[0] is None else torch.stack(flags_tensor)
    return toks, flipped_toks, answer_toks, positions, flags_tensor


def get_data_and_metrics(
        model: HookedTransformer,
        task_name: str,
    ):
    assert task_name in ["ioi", "greater_than", "sentiment_cont", "sentiment_class", "mood_sentiment"]

    if task_name == "ioi":
        ds = UniversalPatchingDataset.from_ioi(model, 70)
        logit_diff_metric = partial(compute_logit_diff,mode='simple')
        metric = CircuitMetric("logit_diff", logit_diff_metric)

    elif task_name == "greater_than":
        # Get data
        ds = UniversalPatchingDataset.from_greater_than(model, 1000)
        prob_diff_metric = partial(
            compute_probability_diff, 
            mode="group_sum"
        )
        metric = CircuitMetric("prob_diff", prob_diff_metric)

    elif task_name == "sentiment_cont":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "cont")
        logit_diff_metric = partial(compute_logit_diff, mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric)

    elif task_name == "sentiment_class":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "class")
        logit_diff_metric = partial(compute_logit_diff,  mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric)
        
    def wrap_metric(metr):
        def wrapped_fn(logits, clean_logits, label, positions, flags_tensor):
            return metr(logits, label, positions, flags_tensor)
        return wrapped_fn
    return ds, wrap_metric(metric)
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
ds, metric = get_data_and_metrics(model, 'ioi')
model.cfg.use_split_qkv_input = True
model.cfg.use_attn_result = True
model.cfg.use_hook_mlp_in = True
graph = Graph.from_model(model)
#%%
dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)
#%%
baseline = evaluate_baseline(model, dataloader, metric).mean()
# %%
attribute(model, graph, dataloader, lambda *args: -metric(*args), integrated_gradients=30)
# %%
graph.apply_greedy(400)
graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
results = evaluate_graph(model, graph, dataloader, metric)
# %%
gz = graph.to_graphviz()
gz.draw('images/ioi.png', prog='dot')
# %%
