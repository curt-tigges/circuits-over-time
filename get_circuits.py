#%%
from functools import partial
import argparse

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

def get_args():
    parser = argparse.ArgumentParser(description="Download & assess model checkpoints")
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
    return parser.parse_args()    


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict

def metric_mapper(metric_name):
    if metric_name == "logit_diff":
        return compute_logit_diff
    elif metric_name == "prob_diff":
        return compute_probability_diff
    #elif metric_name == "kl_div":
    #    return compute_kl_divergence
    #elif metric_name == "js_div":
    #    return compute_js_divergence
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")
                         

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
def main(args):
    model = HookedTransformer.from_pretrained(
        args.model,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        device='cuda',
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    
    # Set up for task 
    task = args.task
    ds, metric = get_data_and_metrics(model, task, eap=True)
    graph = Graph.from_model(model)
    dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # Evaluate baseline and graph
    baseline = evaluate_baseline(model, dataloader, metric).mean()
    print(baseline)
    attribute(model, graph, dataloader, metric, integrated_gradients=30)
    graph.apply_greedy(400)
    graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
    results = evaluate_graph(model, graph, dataloader, metric).mean()
    print(results)

    # Save graph and results
    graph.to_json(f'graphs/{task}.json')
    gz = graph.to_graphviz()
    gz.draw(f'images/{task}.png', prog='dot')
    return graph, results

if __name__ == "__main__":
    args = get_args()
    main(args)

