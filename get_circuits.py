#%%
from functools import partial
import os
import argparse
import yaml

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
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & assess model checkpoints")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="ioi",
        help="Name of task dataset for which to find the circuit",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="pythia-160m",
        help="Name of model to load",
    )
    parser.add_argument(
        "-e",
        "--eval_metric",
        default="logit_diff",
        help="Name of metric to use for EAP evaluation",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "-l",
        "--large_model",
        default=False,
        help="Whether to load a large model",
    )
    parser.add_argument(
        "-cp",
        "--ckpt",
        default=143000,
        help="Checkpoint to load",
    )
    parser.add_argument(    
        "-cd",
        "--cache_dir",
        default="model_cache",
        help="Directory for cache",
    )   
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def process_args():
    # Returns a namespace of arguments either from a config file or from the command line
    args = get_args()
    if args.config is not None:
        config = read_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)
    # Placeholder to revisit when we want to add different model seed variants
    setattr(args, "canonical_model", True)
    return args


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict


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
args = process_args()
batch_size = args.batch_size
model_name = args.model
model_tl_name = args.model

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
os.makedirs(f"results/graphs/pythia-160m", exist_ok=True)
os.makedirs(f"results/images/pythia-160m", exist_ok=True)
graph.to_json(f'results/graphs/pythia-160m/{task}.json')
# %%
gz = graph.to_graphviz()
gz.draw(f'results/images/pythia-160m/{task}.png', prog='dot')
# %%
